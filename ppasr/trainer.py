import io
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from datetime import timedelta

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.static import InputSpec
from tqdm import tqdm
from visualdl import LogWriter

from ppasr.data_utils.collate_fn import collate_fn
from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer
from ppasr.data_utils.normalizer import FeatureNormalizer
from ppasr.data_utils.reader import PPASRDataset
from ppasr.data_utils.sampler import SortagradBatchSampler, SortagradDistributedBatchSampler
from ppasr.decoders.ctc_greedy_decoder import greedy_decoder_batch
from ppasr.model_utils.deepspeech2.model import deepspeech2, deepspeech2_big
from ppasr.model_utils.utils import DeepSpeech2ModelExport
from ppasr.utils.metrics import cer, wer
from ppasr.utils.utils import create_manifest, create_noise, count_manifest, compute_mean_std
from ppasr.utils.utils import labels_to_string
from ppasr.utils.model_summary import summary


class PPASRTrainer(object):
    def __init__(self,
                 use_model='deepspeech2',
                 feature_method='linear',
                 mean_std_path='dataset/mean_std.npz',
                 train_manifest='dataset/manifest.train',
                 test_manifest='dataset/manifest.test',
                 dataset_vocab='dataset/vocabulary.txt',
                 num_workers=8,
                 alpha=2.2,
                 beta=4.3,
                 beam_size=300,
                 num_proc_bsearch=10,
                 cutoff_prob=0.99,
                 cutoff_top_n=40,
                 decoder='ctc_greedy',
                 metrics_type='cer',
                 lang_model_path='lm/zh_giga.no_cna_cmn.prune01244.klm'):
        """
        PPASR集成工具类
        :param use_model: 所使用的模型
        :param feature_method: 所使用的预处理方法
        :param mean_std_path: 数据集的均值和标准值的npy文件路径
        :param train_manifest: 训练数据的数据列表路径
        :param test_manifest: 测试数据的数据列表路径
        :param dataset_vocab: 数据字典的路径
        :param num_workers: 读取数据的线程数量
        :param alpha: 集束搜索的LM系数
        :param beta: 集束搜索的WC系数
        :param beam_size: 集束搜索的大小，范围:[5, 500]
        :param num_proc_bsearch: 集束搜索方法使用CPU数量
        :param cutoff_prob: 剪枝的概率
        :param cutoff_top_n: 剪枝的最大值
        :param metrics_type: 计算错误方法
        :param decoder: 结果解码方法，支持ctc_beam_search和ctc_greedy
        :param lang_model_path: 语言模型文件路径
        """
        self.use_model = use_model
        self.feature_method = feature_method
        self.mean_std_path = mean_std_path
        self.train_manifest = train_manifest
        self.test_manifest = test_manifest
        self.dataset_vocab = dataset_vocab
        self.num_workers = num_workers
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size
        self.num_proc_bsearch = num_proc_bsearch
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.decoder = decoder
        self.metrics_type = metrics_type
        self.lang_model_path = lang_model_path
        self.beam_search_decoder = None

    def create_data(self,
                    annotation_path='dataset/annotation/',
                    noise_manifest_path='dataset/manifest.noise',
                    noise_path='dataset/audio/noise',
                    num_samples=1000000,
                    count_threshold=2,
                    is_change_frame_rate=True,
                    max_test_manifest=10000):
        """
        创建数据列表和词汇表
        :param annotation_path: 标注文件的路径
        :param noise_manifest_path: 噪声数据列表的路径
        :param noise_path: 噪声音频存放的文件夹路径
        :param num_samples: 用于计算均值和标准值得音频数量，当为-1使用全部数据
        :param count_threshold: 字符计数的截断阈值，0为不做限制
        :param is_change_frame_rate: 是否统一改变音频为16000Hz，这会消耗大量的时间
        :param max_test_manifest: 生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据
        """
        print('开始生成数据列表...')
        create_manifest(annotation_path=annotation_path,
                        train_manifest_path=self.train_manifest,
                        test_manifest_path=self.test_manifest,
                        is_change_frame_rate=is_change_frame_rate,
                        max_test_manifest=max_test_manifest)
        print('=' * 70)
        print('开始生成噪声数据列表...')
        create_noise(path=noise_path,
                     noise_manifest_path=noise_manifest_path,
                     is_change_frame_rate=is_change_frame_rate)
        print('=' * 70)

        print('开始生成数据字典...')
        counter = Counter()
        count_manifest(counter, self.train_manifest)

        count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        with open(self.dataset_vocab, 'w', encoding='utf-8') as fout:
            fout.write('<blank>\t-1\n')
            for char, count in count_sorted:
                if char == ' ': char = '<space>'
                # 跳过指定的字符阈值，超过这大小的字符都忽略
                if count < count_threshold: break
                fout.write('%s\t%d\n' % (char, count))
        print('数据字典生成完成！')

        print('=' * 70)
        print('开始抽取{}条数据计算均值和标准值...'.format(num_samples))
        compute_mean_std(feature_method=self.feature_method,
                         manifest_path=self.train_manifest,
                         output_path=self.mean_std_path,
                         num_samples=num_samples,
                         num_workers=self.num_workers)

    def evaluate(self,
                 batch_size=32,
                 min_duration=0,
                 max_duration=-1,
                 resume_model='models/deepspeech2/best_model/'):
        """
        评估模型
        :param batch_size: 评估的批量大小
        :param min_duration: 过滤最短的音频长度
        :param max_duration: 过滤最长的音频长度，当为-1的时候不限制长度
        :param resume_model: 所使用的模型
        :return: 评估结果
        """
        # 获取测试数据
        test_dataset = PPASRDataset(data_list=self.test_manifest,
                                    vocab_filepath=self.dataset_vocab,
                                    mean_std_filepath=self.mean_std_path,
                                    min_duration=min_duration,
                                    max_duration=max_duration,
                                    feature_method=self.feature_method)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=self.num_workers,
                                 use_shared_memory=False)

        # 获取模型
        if self.use_model == 'deepspeech2':
            model = deepspeech2(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        elif self.use_model == 'deepspeech2_big':
            model = deepspeech2_big(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.use_model))
        # 打印模型
        input_data = [paddle.rand([1, 161, 900], dtype=paddle.float32),
                      paddle.to_tensor(200, dtype=paddle.int64)]
        summary(net=model, input=input_data)
        assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "模型不存在！"
        model.set_state_dict(paddle.load(os.path.join(resume_model, 'model.pdparams')))
        model.eval()

        c = []
        for inputs, labels, input_lens, _ in tqdm(test_loader()):
            # 执行识别
            outs, out_lens = model(inputs, input_lens)
            outs = paddle.nn.functional.softmax(outs, 2)
            # 解码获取识别结果
            out_strings = self.decoder_result(outs.numpy(), out_lens, test_dataset.vocab_list)
            labels_str = labels_to_string(labels.numpy(), test_dataset.vocab_list)
            for out_string, label in zip(*(out_strings, labels_str)):
                # 计算字错率或者词错率
                if self.metrics_type == 'wer':
                    c.append(wer(out_string, label))
                else:
                    c.append(cer(out_string, label))
        cer_result = float(sum(c) / len(c))
        return cer_result

    def train(self,
              batch_size=32,
              min_duration=0,
              max_duration=20,
              num_epoch=65,
              learning_rate=5e-5,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              augment_conf_path='conf/augmentation.json'):
        """
        训练模型
        :param batch_size: 训练的批量大小
        :param min_duration: 过滤最短的音频长度
        :param max_duration: 过滤最长的音频长度，当为-1的时候不限制长度
        :param num_epoch: 训练的轮数
        :param learning_rate: 初始学习率的大小
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        :param augment_conf_path: 数据增强的配置文件，为json格式
        """
        # 获取有多少张显卡训练
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')
        if nranks > 1:
            # 初始化Fleet环境
            fleet.init(is_collective=True)

        # 获取训练数据
        if augment_conf_path is not None and os.path.exists(augment_conf_path):
            augmentation_config = io.open(augment_conf_path, mode='r', encoding='utf8').read()
        else:
            if augment_conf_path is not None and not os.path.exists(augment_conf_path):
                print('[{}] 数据增强配置文件{}不存在'.format(datetime.now(), augment_conf_path), file=sys.stderr)
            augmentation_config = '{}'
        train_dataset = PPASRDataset(data_list=self.train_manifest,
                                     vocab_filepath=self.dataset_vocab,
                                     feature_method=self.feature_method,
                                     mean_std_filepath=self.mean_std_path,
                                     min_duration=min_duration,
                                     max_duration=max_duration,
                                     augmentation_config=augmentation_config)
        # 设置支持多卡训练
        if nranks > 1:
            train_batch_sampler = SortagradDistributedBatchSampler(train_dataset,
                                                                   batch_size=batch_size,
                                                                   sortagrad=True,
                                                                   drop_last=True,
                                                                   shuffle=True)
        else:
            train_batch_sampler = SortagradBatchSampler(train_dataset,
                                                        batch_size=batch_size,
                                                        sortagrad=True,
                                                        drop_last=True,
                                                        shuffle=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_sampler=train_batch_sampler,
                                  num_workers=self.num_workers)
        # 获取测试数据
        test_dataset = PPASRDataset(data_list=self.test_manifest,
                                    vocab_filepath=self.dataset_vocab,
                                    feature_method=self.feature_method,
                                    mean_std_filepath=self.mean_std_path,
                                    min_duration=min_duration,
                                    max_duration=max_duration)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=self.num_workers)

        # 获取模型
        if self.use_model == 'deepspeech2':
            model = deepspeech2(feat_size=train_dataset.feature_dim, vocab_size=train_dataset.vocab_size)
        elif self.use_model == 'deepspeech2_big':
            model = deepspeech2_big(feat_size=train_dataset.feature_dim, vocab_size=train_dataset.vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.use_model))
        # 打印模型
        input_data = [paddle.rand([1, 161, 900], dtype=paddle.float32),
                      paddle.to_tensor(200, dtype=paddle.int64)]
        summary(net=model, input=input_data)
        # 设置优化方法
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=3.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=learning_rate, gamma=0.93)
        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
                                           learning_rate=scheduler,
                                           weight_decay=1e-6,
                                           grad_clip=grad_clip)

        # 设置支持多卡训练
        if nranks > 1:
            optimizer = fleet.distributed_optimizer(optimizer)
            model = fleet.distributed_model(model)

        print('[{}] 训练数据：{}'.format(datetime.now(), len(train_dataset)))

        # 加载预训练模型
        if pretrained_model is not None:
            model_dict = model.state_dict()
            model_state_dict = paddle.load(os.path.join(pretrained_model, 'model.pdparams'))
            # 特征层
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if weight.shape != list(model_state_dict[name].shape):
                        print('{} not used, shape {} unmatched with {} in model.'.
                              format(name, list(model_state_dict[name].shape), weight.shape))
                        model_state_dict.pop(name, None)
                else:
                    print('Lack weight: {}'.format(name))
            model.set_state_dict(model_state_dict)
            print('[{}] 成功加载预训练模型：{}'.format(datetime.now(), pretrained_model))

        # 加载恢复模型
        last_epoch = 0
        last_model_dir = os.path.join(save_model_path, self.use_model, 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
            model.set_state_dict(paddle.load(os.path.join(resume_model, 'model.pdparams')))
            optimizer.set_state_dict(paddle.load(os.path.join(resume_model, 'optimizer.pdopt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                last_epoch = json.load(f)['last_epoch'] - 1
            print('[{}] 成功恢复模型参数和优化方法参数：{}'.format(datetime.now(), resume_model))

        # 获取损失函数
        ctc_loss = paddle.nn.CTCLoss(reduction='none')

        test_step, train_step = 0, 0
        best_test_cer = 1
        train_times = []
        sum_batch = len(train_loader) * num_epoch
        train_batch_sampler.epoch = last_epoch
        if local_rank == 0:
            writer.add_scalar('Train/lr', scheduler.last_lr, last_epoch)
        try:
            # 开始训练
            for epoch in range(last_epoch, num_epoch):
                epoch += 1
                start_epoch = time.time()
                start = time.time()
                for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(train_loader()):
                    out, out_lens = model(inputs, input_lens)
                    out = paddle.transpose(out, perm=[1, 0, 2])

                    # 计算损失
                    loss = ctc_loss(out, labels, out_lens, label_lens)
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()
                    train_times.append((time.time() - start) * 1000)
                    # 多卡训练只使用一个进程打印
                    if batch_id % 100 == 0 and local_rank == 0:
                        eta_sec = (sum(train_times) / len(train_times)) * (sum_batch - (epoch - 1) * len(train_loader) - batch_id)
                        eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                        print('[{}] Train epoch: [{}/{}], batch: [{}/{}], loss: {:.5f}, learning rate: {:>.8f}, eta: {}'.format(
                                datetime.now(), epoch, num_epoch, batch_id, len(train_loader), loss.numpy()[0], scheduler.get_lr(), eta_str))
                        writer.add_scalar('Train/Loss', loss, train_step)
                        train_step += 1
                        train_times = []
                    # 固定步数也要保存一次模型
                    if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                        self.save_model(save_model_path=save_model_path, use_model=self.use_model, epoch=epoch,
                                        model=model, optimizer=optimizer)
                    start = time.time()

                # 多卡训练只使用一个进程执行评估和保存模型
                if local_rank == 0:
                    # 执行评估
                    model.eval()
                    print('\n', '=' * 70)
                    c, l = self.__test(model, test_loader, test_dataset.vocab_list, ctc_loss)
                    print('[{}] Test epoch: {}, time/epoch: {}, loss: {:.5f}, {}: {:.5f}'.format(
                        datetime.now(), epoch, str(timedelta(seconds=(time.time() - start_epoch))), l, self.metrics_type, c))
                    print('=' * 70, '\n')
                    writer.add_scalar('Test/{}'.format(self.metrics_type), c, test_step)
                    writer.add_scalar('Test/Loss', l, test_step)
                    test_step += 1
                    model.train()

                    # 记录学习率
                    writer.add_scalar('Train/lr', scheduler.last_lr, epoch)
                    # 保存最优模型
                    if c <= best_test_cer:
                        best_test_cer = c
                        self.save_model(save_model_path=save_model_path, use_model=self.use_model, model=model,
                                        optimizer=optimizer, epoch=epoch, error_type=self.metrics_type, error_rate=c, test_loss=l, best_model=True)
                    # 保存模型
                    self.save_model(save_model_path=save_model_path, use_model=self.use_model, epoch=epoch, model=model,
                                    error_type=self.metrics_type, error_rate=c, test_loss=l, optimizer=optimizer)
                scheduler.step()
        except KeyboardInterrupt:
            # Ctrl+C退出时保存模型
            if local_rank == 0:
                print('请等一下，正在保存模型...')
                self.save_model(save_model_path=save_model_path, use_model=self.use_model, epoch=epoch, model=model,
                                optimizer=optimizer)

    # 评估模型
    @paddle.no_grad()
    def __test(self, model, test_loader, vocabulary, ctc_loss):
        cer_result, test_loss = [], []
        for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(test_loader()):
            # 执行识别
            outs, out_lens = model(inputs, input_lens)
            out = paddle.transpose(outs, perm=[1, 0, 2])
            # 计算损失
            loss = ctc_loss(out, labels, out_lens, label_lens)
            loss = loss.mean().numpy()[0]
            test_loss.append(loss)
            outs = paddle.nn.functional.softmax(outs, 2)
            # 解码获取识别结果
            out_strings = self.decoder_result(outs.numpy(), out_lens, vocabulary)
            labels_str = labels_to_string(labels.numpy(), vocabulary)
            cer_batch = []
            for out_string, label in zip(*(out_strings, labels_str)):
                # 计算字错率或者词错率
                if self.metrics_type == 'wer':
                    c = wer(out_string, label)
                else:
                    c = cer(out_string, label)
                cer_result.append(c)
                cer_batch.append(c)
            if batch_id % 10 == 0:
                print('[{}] Test batch: [{}/{}], loss: {:.5f}, '
                      '{}: {:.5f}'.format(datetime.now(), batch_id, len(test_loader),loss,self.metrics_type,
                                          float(sum(cer_batch) / len(cer_batch))))
        cer_result = float(sum(cer_result) / len(cer_result))
        test_loss = float(sum(test_loss) / len(test_loss))
        return cer_result, test_loss

    # 保存模型
    @staticmethod
    def save_model(save_model_path, use_model, epoch, model, optimizer, error_type='cer', error_rate=-1., test_loss=-1., best_model=False):
        if not best_model:
            model_path = os.path.join(save_model_path, use_model, 'epoch_{}'.format(epoch))
            os.makedirs(model_path, exist_ok=True)
            paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
                f.write('{"last_epoch": %d, "test_%s": %f, "test_loss": %f}' % (epoch, error_type, error_rate, test_loss))
            last_model_path = os.path.join(save_model_path, use_model, 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, use_model, 'epoch_{}'.format(epoch - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        else:
            model_path = os.path.join(save_model_path, use_model, 'best_model')
            paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
                f.write('{"last_epoch": %d, "test_%s": %f, "test_loss": %f}' % (epoch, error_type, error_rate, test_loss))
        print('[{}] 已保存模型：{}'.format(datetime.now(), model_path))

    def decoder_result(self, outs, outs_lens, vocabulary):
        # 集束搜索方法的处理
        if self.decoder == "ctc_beam_search" and self.beam_search_decoder is None:
            try:
                from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
                self.beam_search_decoder = BeamSearchDecoder(self.alpha, self.beta, self.lang_model_path, vocabulary)
            except ModuleNotFoundError:
                print('\n==================================================================', file=sys.stderr)
                print('缺少 paddlespeech-ctcdecoders 库，请安装，如果是Windows系统，只能使用ctc_greedy。', file=sys.stderr)
                print('【注意】已自动切换为ctc_greedy解码器。', file=sys.stderr)
                print('==================================================================\n', file=sys.stderr)
                self.decoder = 'ctc_greedy'

        # 执行解码
        outs = [outs[i, :l, :] for i, l in enumerate(outs_lens)]
        if self.decoder == 'ctc_greedy':
            result = greedy_decoder_batch(outs, vocabulary)
        else:
            result = self.beam_search_decoder.decode_batch_beam_search(probs_split=outs,
                                                                       beam_alpha=self.alpha,
                                                                       beam_beta=self.beta,
                                                                       beam_size=self.beam_size,
                                                                       cutoff_prob=self.cutoff_prob,
                                                                       cutoff_top_n=self.cutoff_top_n,
                                                                       vocab_list=vocabulary,
                                                                       num_processes=self.num_proc_bsearch)
        return result

    def export(self, save_model_path='models/', resume_model='models/deepspeech2/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取训练数据
        audio_featurizer = AudioFeaturizer(feature_method=self.feature_method)
        text_featurizer = TextFeaturizer(self.dataset_vocab)
        featureNormalizer = FeatureNormalizer(mean_std_filepath=self.mean_std_path, feature_method=self.feature_method)

        # 获取模型
        if self.use_model == 'deepspeech2':
            base_model = deepspeech2(feat_size=audio_featurizer.feature_dim, vocab_size=text_featurizer.vocab_size)
        elif self.use_model == 'deepspeech2_big':
            base_model = deepspeech2_big(feat_size=audio_featurizer.feature_dim, vocab_size=text_featurizer.vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.use_model))
        # 打印模型
        input_data = [paddle.rand([1, 161, 900], dtype=paddle.float32),
                      paddle.to_tensor(200, dtype=paddle.int64)]
        summary(net=base_model, input=input_data)
        # 加载预训练模型
        resume_model_path = os.path.join(resume_model, 'model.pdparams')
        assert os.path.exists(resume_model_path), "恢复模型不存在！"
        base_model.set_state_dict(paddle.load(resume_model_path))
        print('[{}] 成功恢复模型参数和优化方法参数：{}'.format(datetime.now(), resume_model_path))

        # 获取模型
        if 'deepspeech2' in self.use_model:
            model = DeepSpeech2ModelExport(model=base_model, feature_mean=featureNormalizer.mean, feature_std=featureNormalizer.std)
            input_spec = [InputSpec(shape=(-1, audio_featurizer.feature_dim, -1), dtype=paddle.float32),
                          InputSpec(shape=(-1,), dtype=paddle.int64),
                          InputSpec(shape=(base_model.num_rnn_layers, -1, base_model.rnn_size), dtype=paddle.float32),
                          InputSpec(shape=(base_model.num_rnn_layers, -1, base_model.rnn_size), dtype=paddle.float32)]
        else:
            raise Exception('没有该模型：{}'.format(self.use_model))

        infer_model_dir = os.path.join(save_model_path, self.use_model, 'infer')
        os.makedirs(infer_model_dir, exist_ok=True)
        infer_model_path = os.path.join(infer_model_dir, 'model')
        paddle.jit.save(layer=model, path=infer_model_path, input_spec=input_spec)
        print("预测模型已保存：{}".format(infer_model_dir))
