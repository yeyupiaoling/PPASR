import io
import json
import os
import platform
import shutil
import time
from collections import Counter
from datetime import timedelta

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.static import InputSpec
from tqdm import tqdm
from visualdl import LogWriter

from ppasr import SUPPORT_MODEL
from ppasr.data_utils.collate_fn import collate_fn
from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer
from ppasr.data_utils.normalizer import FeatureNormalizer
from ppasr.data_utils.reader import PPASRDataset
from ppasr.data_utils.sampler import SortagradBatchSampler, SortagradDistributedBatchSampler
from ppasr.decoders.ctc_greedy_decoder import greedy_decoder_batch
from ppasr.model_utils.deepspeech2.model import deepspeech2, deepspeech2_big
from ppasr.model_utils.deepspeech2_no_stream.model import deepspeech2_no_stream, deepspeech2_big_no_stream
from ppasr.model_utils.utils import DeepSpeech2ModelExport, DeepSpeech2NoStreamModelExport
from ppasr.utils.logger import setup_logger
from ppasr.utils.metrics import cer, wer
from ppasr.utils.utils import create_manifest, create_noise, count_manifest, dict_to_object, merge_audio
from ppasr.utils.utils import labels_to_string
from ppasr.utils.model_summary import summary

logger = setup_logger(__name__)


class PPASRTrainer(object):
    def __init__(self, configs):
        """PPASR集成工具类

        :param configs: 配置字典
        """
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.beam_search_decoder = None

    def create_data(self,
                    annotation_path='dataset/annotation/',
                    noise_path='dataset/audio/noise',
                    num_samples=1000000,
                    count_threshold=2,
                    is_change_frame_rate=True,
                    max_test_manifest=10000,
                    is_merge_audio=False,
                    save_audio_path='dataset/audio/merge_audio',
                    max_duration=600):
        """
        创建数据列表和词汇表
        :param annotation_path: 标注文件的路径
        :param noise_path: 噪声音频存放的文件夹路径
        :param num_samples: 用于计算均值和标准值得音频数量，当为-1使用全部数据
        :param count_threshold: 字符计数的截断阈值，0为不做限制
        :param is_change_frame_rate: 是否统一改变音频为16000Hz，这会消耗大量的时间
        :param max_test_manifest: 生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据
        :param is_merge_audio: 是否将多个短音频合并成长音频，以减少音频文件数量，注意自动删除原始音频文件
        :param save_audio_path: 合并音频的保存路径
        :param max_duration: 合并音频的最大长度，单位秒
        """
        if is_merge_audio:
            logger.info('开始合并音频...')
            merge_audio(annotation_path=annotation_path, save_audio_path=save_audio_path, max_duration=max_duration)
            logger.info('合并音频已完成，原始音频文件和标注文件已自动删除，其他原始文件可手动删除！')

        logger.info('开始生成数据列表...')
        create_manifest(annotation_path=annotation_path,
                        train_manifest_path=self.configs.dataset.train_manifest,
                        test_manifest_path=self.configs.dataset.test_manifest,
                        is_change_frame_rate=is_change_frame_rate,
                        max_test_manifest=max_test_manifest)
        logger.info('=' * 70)
        logger.info('开始生成噪声数据列表...')
        create_noise(path=noise_path,
                     noise_manifest_path=self.configs.dataset.noise_manifest_path,
                     is_change_frame_rate=is_change_frame_rate)
        logger.info('=' * 70)

        logger.info('开始生成数据字典...')
        counter = Counter()
        count_manifest(counter, self.configs.dataset.train_manifest)

        count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        with open(self.configs.dataset.dataset_vocab, 'w', encoding='utf-8') as fout:
            fout.write('<blank>\t-1\n')
            fout.write('<unk>\t-1\n')
            for char, count in count_sorted:
                if char == ' ': char = '<space>'
                # 跳过指定的字符阈值，超过这大小的字符都忽略
                if count < count_threshold: break
                fout.write('%s\t%d\n' % (char, count))
            fout.write('<eos>\t-1\n')
        logger.info('数据字典生成完成！')

        logger.info('=' * 70)
        normalizer = FeatureNormalizer(mean_std_filepath=self.configs.dataset.mean_std_path)
        normalizer.compute_mean_std(manifest_path=self.configs.dataset.train_manifest,
                                    num_workers=self.configs.dataset.num_workers,
                                    preprocess_configs=self.configs.preprocess,
                                    num_samples=num_samples)
        print('计算的均值和标准值已保存在 %s！' % self.configs.dataset.mean_std_path)

    def evaluate(self, batch_size=32, resume_model='models/deepspeech2_fbank/best_model/'):
        """
        评估模型
        :param batch_size: 评估的批量大小
        :param resume_model: 所使用的模型
        :return: 评估结果
        """
        if not os.path.exists(self.configs.dataset.mean_std_path):
            raise Exception(f'归一化列表文件 {self.configs.dataset.mean_std_path} 不存在')
        # 获取测试数据
        test_dataset = PPASRDataset(preprocess_configs=self.configs.preprocess,
                                    data_manifest=self.configs.dataset.test_manifest,
                                    vocab_filepath=self.configs.dataset.dataset_vocab,
                                    mean_std_filepath=self.configs.dataset.mean_std_path,
                                    min_duration=self.configs.dataset.min_duration,
                                    max_duration=self.configs.dataset.max_duration)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=self.configs.dataset.num_workers)

        # 获取模型
        if self.configs.use_model == 'deepspeech2':
            model = deepspeech2(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        elif self.configs.use_model == 'deepspeech2_no_stream':
            model = deepspeech2_no_stream(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        elif self.configs.use_model == 'deepspeech2_big':
            model = deepspeech2_big(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        elif self.configs.use_model == 'deepspeech2_big_no_stream':
            model = deepspeech2_big_no_stream(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.configs.use_model))
        # 打印模型
        input_data = [paddle.rand([1, 900, test_dataset.feature_dim], dtype=paddle.float32),
                      paddle.to_tensor(200 if 'no_stream' not in self.configs.use_model else 0, dtype=paddle.int64)]
        summary(net=model, input=input_data)

        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pdparams')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model.set_state_dict(paddle.load(resume_model))
        logger.info(f'成功加载模型：{resume_model}')
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
                if self.configs.metrics_type == 'wer':
                    c.append(wer(out_string, label))
                else:
                    c.append(cer(out_string, label))
        cer_result = float(sum(c) / len(c))
        return cer_result

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              augment_conf_path='configs/augmentation.json'):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        :param augment_conf_path: 数据增强的配置文件，为json格式
        """
        # 训练只能用贪心解码，解码速度快
        self.configs.decoder = 'ctc_greedy'
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
                logger.error('数据增强配置文件{}不存在'.format(augment_conf_path))
            augmentation_config = '{}'
        if not os.path.exists(self.configs.dataset.mean_std_path):
            raise Exception(f'归一化列表文件 {self.configs.dataset.mean_std_path} 不存在')
        train_dataset = PPASRDataset(preprocess_configs=self.configs.preprocess,
                                     data_manifest=self.configs.dataset.train_manifest,
                                     vocab_filepath=self.configs.dataset.dataset_vocab,
                                     mean_std_filepath=self.configs.dataset.mean_std_path,
                                     min_duration=self.configs.dataset.min_duration,
                                     max_duration=self.configs.dataset.max_duration,
                                     augmentation_config=augmentation_config,
                                     train=True)
        # 设置支持多卡训练
        if nranks > 1:
            train_batch_sampler = SortagradDistributedBatchSampler(train_dataset,
                                                                   batch_size=self.configs.dataset.batch_size,
                                                                   sortagrad=True,
                                                                   drop_last=True,
                                                                   shuffle=True)
        else:
            train_batch_sampler = SortagradBatchSampler(train_dataset,
                                                        batch_size=self.configs.dataset.batch_size,
                                                        sortagrad=True,
                                                        drop_last=True,
                                                        shuffle=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  collate_fn=collate_fn,
                                  batch_sampler=train_batch_sampler,
                                  num_workers=self.configs.dataset.num_workers)
        # 获取测试数据
        test_dataset = PPASRDataset(preprocess_configs=self.configs.preprocess,
                                    data_manifest=self.configs.dataset.test_manifest,
                                    vocab_filepath=self.configs.dataset.dataset_vocab,
                                    mean_std_filepath=self.configs.dataset.mean_std_path,
                                    min_duration=self.configs.dataset.min_duration,
                                    max_duration=self.configs.dataset.max_duration)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.configs.dataset.batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=self.configs.dataset.num_workers)

        # 获取模型
        if self.configs.use_model == 'deepspeech2':
            model = deepspeech2(feat_size=train_dataset.feature_dim, vocab_size=train_dataset.vocab_size)
        elif self.configs.use_model == 'deepspeech2_no_stream':
            model = deepspeech2_no_stream(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        elif self.configs.use_model == 'deepspeech2_big':
            model = deepspeech2_big(feat_size=train_dataset.feature_dim, vocab_size=train_dataset.vocab_size)
        elif self.configs.use_model == 'deepspeech2_big_no_stream':
            model = deepspeech2_big_no_stream(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.configs.use_model))
        # 打印模型
        input_data = [paddle.rand([1, 900, train_dataset.feature_dim], dtype=paddle.float32),
                      paddle.to_tensor(200 if 'no_stream' not in self.configs.use_model else 0, dtype=paddle.int64)]
        summary(net=model, input=input_data)
        # 设置优化方法
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.configs.optimizer.clip_norm)
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=float(self.configs.optimizer.learning_rate),
                                                         gamma=self.configs.optimizer.gamma)
        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
                                           learning_rate=scheduler,
                                           weight_decay=float(self.configs.optimizer.weight_decay),
                                           grad_clip=grad_clip)

        # 设置支持多卡训练
        if nranks > 1:
            optimizer = fleet.distributed_optimizer(optimizer)
            model = fleet.distributed_model(model)

        logger.info('训练数据：{}'.format(len(train_dataset)))

        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            model_dict = model.state_dict()
            model_state_dict = paddle.load(pretrained_model)
            # 特征层
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if weight.shape != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), weight.shape))
                        model_state_dict.pop(name, None)
                else:
                    logger.info('Lack weight: {}'.format(name))
            model.set_state_dict(model_state_dict)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

        # 加载恢复模型
        last_epoch = 0
        best_error_rate = 1.0
        last_model_dir = os.path.join(save_model_path, f'{self.configs.use_model}_{self.configs.preprocess.feature_method}', 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pdparams')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
            model.set_state_dict(paddle.load(os.path.join(resume_model, 'model.pdparams')))
            optimizer.set_state_dict(paddle.load(os.path.join(resume_model, 'optimizer.pdopt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                if 'test_cer' in json_data.keys():
                    best_error_rate = abs(json_data['test_cer'])
                if 'test_wer' in json_data.keys():
                    best_error_rate = abs(json_data['test_wer'])
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))

        # 获取损失函数
        ctc_loss = paddle.nn.CTCLoss(reduction='none')

        test_step, train_step = 0, 0
        train_times = []
        sum_batch = len(train_loader) * self.configs.num_epoch
        train_batch_sampler.epoch = last_epoch
        if local_rank == 0:
            writer.add_scalar('Train/lr', scheduler.last_lr, last_epoch)
        try:
            # 开始训练
            for epoch in range(last_epoch, self.configs.num_epoch):
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
                    if batch_id % 100 == 0:
                        eta_sec = (sum(train_times) / len(train_times)) * (sum_batch - (epoch - 1) * len(train_loader) - batch_id)
                        eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                        logger.info('Train epoch: [{}/{}], batch: [{}/{}], loss: {:.5f}, learning rate: {:>.8f}, eta: {}'.format(
                                epoch, self.configs.num_epoch, batch_id, len(train_loader), loss.numpy()[0], scheduler.get_lr(), eta_str))
                        if local_rank == 0:
                            writer.add_scalar('Train/Loss', loss, train_step)
                        train_step += 1
                        train_times = []
                    # 固定步数也要保存一次模型
                    if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                        self.save_model(save_model_path=save_model_path, epoch=epoch,
                                        model=model, optimizer=optimizer)
                    start = time.time()

                # 执行评估
                model.eval()
                logger.info('=' * 70)
                c, l = self.__test(model, test_loader, test_dataset.vocab_list, ctc_loss)
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, {}: {:.5f}'.format(
                    epoch, str(timedelta(seconds=(time.time() - start_epoch))), l, self.configs.metrics_type, c))
                logger.info('=' * 70)
                test_step += 1
                model.train()
                # 多卡训练只使用一个进程执行评估和保存模型
                if local_rank == 0:
                    writer.add_scalar('Test/{}'.format(self.configs.metrics_type), c, test_step)
                    writer.add_scalar('Test/Loss', l, test_step)
                    # 记录学习率
                    writer.add_scalar('Train/lr', scheduler.last_lr, epoch)
                    # 保存最优模型
                    if c <= best_error_rate:
                        best_error_rate = c
                        self.save_model(save_model_path=save_model_path, model=model,
                                        optimizer=optimizer, epoch=epoch, error_rate=c, test_loss=l, best_model=True)
                    # 保存模型
                    self.save_model(save_model_path=save_model_path, epoch=epoch, model=model,
                                    error_rate=c, test_loss=l, optimizer=optimizer)
                scheduler.step()
        except KeyboardInterrupt:
            # 解决停止训练之后还在读取数据导致报错的问题
            del train_dataset, test_dataset
            # Ctrl+C退出时保存模型
            if local_rank == 0:
                try:
                    logger.info(f'请等一下，正在保存模型，当前损失值为：{l}')
                except NameError as e:
                    c, l = 1.0, 1e3
                self.save_model(save_model_path=save_model_path, epoch=epoch, model=model,
                                optimizer=optimizer, error_rate=c, test_loss=l)

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
                if self.configs.metrics_type == 'wer':
                    c = wer(out_string, label)
                else:
                    c = cer(out_string, label)
                cer_result.append(c)
                cer_batch.append(c)
            if batch_id % 10 == 0:
                logger.info('Test batch: [{}/{}], loss: {:.5f}, '
                            '{}: {:.5f}'.format(batch_id, len(test_loader), loss, self.configs.metrics_type,
                                                float(sum(cer_batch) / len(cer_batch))))
        cer_result = float(sum(cer_result) / len(cer_result))
        test_loss = float(sum(test_loss) / len(test_loss))
        return cer_result, test_loss

    # 保存模型
    def save_model(self, save_model_path, epoch, model, optimizer, error_rate=1., test_loss=1e3, best_model=False):
        if not best_model:
            model_path = os.path.join(save_model_path, f'{self.configs.use_model}_{self.configs.preprocess.feature_method}',
                                      'epoch_{}'.format(epoch))
            os.makedirs(model_path, exist_ok=True)
            paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
                f.write('{"last_epoch": %d, "test_%s": %f, "test_loss": %f}' % (
                epoch, self.configs.metrics_type, error_rate, test_loss))
            last_model_path = os.path.join(save_model_path, f'{self.configs.use_model}_{self.configs.preprocess.feature_method}', 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, f'{self.configs.use_model}_{self.configs.preprocess.feature_method}',
                                          'epoch_{}'.format(epoch - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        else:
            model_path = os.path.join(save_model_path, f'{self.configs.use_model}_{self.configs.preprocess.feature_method}', 'best_model')
            paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
            with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
                f.write('{"last_epoch": %d, "test_%s": %f, "test_loss": %f}' % (
                epoch, self.configs.metrics_type, error_rate, test_loss))
        logger.info('已保存模型：{}'.format(model_path))

    def decoder_result(self, outs, outs_lens, vocabulary):
        # 集束搜索方法的处理
        if self.configs.decoder == "ctc_beam_search" and self.beam_search_decoder is None:
            if platform.system() != 'Windows':
                try:
                    from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
                    self.beam_search_decoder = BeamSearchDecoder(vocab_list=vocabulary,
                                                                 **self.configs.ctc_beam_search_decoder)
                except ModuleNotFoundError:
                    logger.warning('==================================================================')
                    logger.warning('缺少 paddlespeech-ctcdecoders 库，请根据文档安装。')
                    logger.warning('【注意】已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                    logger.warning('==================================================================\n')
                    self.configs.decoder = 'ctc_greedy'
            else:
                logger.warning('==================================================================')
                logger.warning('【注意】Windows不支持ctc_beam_search，已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.configs.decoder = 'ctc_greedy'

        # 执行解码
        outs = [outs[i, :l, :] for i, l in enumerate(outs_lens)]
        if self.configs.decoder == 'ctc_greedy':
            result = greedy_decoder_batch(outs, vocabulary)
        else:
            result = self.beam_search_decoder.decode_batch_beam_search_offline(probs_split=outs)
        return result

    def export(self, save_model_path='models/', resume_model='models/deepspeech2_fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        # 获取训练数据
        audio_featurizer = AudioFeaturizer(**self.configs.preprocess)
        text_featurizer = TextFeaturizer(self.configs.dataset.dataset_vocab)
        if not os.path.exists(self.configs.dataset.mean_std_path):
            raise Exception(f'归一化列表文件 {self.configs.dataset.mean_std_path} 不存在')
        featureNormalizer = FeatureNormalizer(mean_std_filepath=self.configs.dataset.mean_std_path)

        # 获取模型
        if self.configs.use_model == 'deepspeech2':
            base_model = deepspeech2(feat_size=audio_featurizer.feature_dim, vocab_size=text_featurizer.vocab_size)
        elif self.configs.use_model == 'deepspeech2_no_stream':
            base_model = deepspeech2_no_stream(feat_size=audio_featurizer.feature_dim,
                                               vocab_size=text_featurizer.vocab_size)
        elif self.configs.use_model == 'deepspeech2_big':
            base_model = deepspeech2_big(feat_size=audio_featurizer.feature_dim, vocab_size=text_featurizer.vocab_size)
        elif self.configs.use_model == 'deepspeech2_big_no_stream':
            base_model = deepspeech2_big_no_stream(feat_size=audio_featurizer.feature_dim,
                                                   vocab_size=text_featurizer.vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.configs.use_model))
        base_model.eval()
        # 打印模型
        input_data = [paddle.rand([1, 900, audio_featurizer.feature_dim], dtype=paddle.float32),
                      paddle.to_tensor(200 if 'no_stream' not in self.configs.use_model else 0, dtype=paddle.int64)]
        summary(net=base_model, input=input_data)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pdparams')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        base_model.set_state_dict(paddle.load(resume_model))
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))

        # 获取导出模型
        if self.configs.use_model == 'deepspeech2' or self.configs.use_model == 'deepspeech2_big':
            model = DeepSpeech2ModelExport(model=base_model, feature_mean=featureNormalizer.mean,
                                           feature_std=featureNormalizer.std)
            input_spec = [InputSpec(shape=(-1, -1, audio_featurizer.feature_dim), dtype=paddle.float32),
                          InputSpec(shape=(-1,), dtype=paddle.int64),
                          InputSpec(shape=(base_model.num_rnn_layers, -1, base_model.rnn_size), dtype=paddle.float32),
                          InputSpec(shape=(base_model.num_rnn_layers, -1, base_model.rnn_size), dtype=paddle.float32)]
        elif self.configs.use_model == 'deepspeech2_no_stream' or self.configs.use_model == 'deepspeech2_big_no_stream':
            model = DeepSpeech2NoStreamModelExport(model=base_model, feature_mean=featureNormalizer.mean,
                                                   feature_std=featureNormalizer.std)
            input_spec = [InputSpec(shape=(-1, -1, audio_featurizer.feature_dim), dtype=paddle.float32),
                          InputSpec(shape=(-1,), dtype=paddle.int64)]
        else:
            raise Exception('没有该模型：{}'.format(self.configs.use_model))

        infer_model_dir = os.path.join(save_model_path, f'{self.configs.use_model}_{self.configs.preprocess.feature_method}', 'infer')
        os.makedirs(infer_model_dir, exist_ok=True)
        infer_model_path = os.path.join(infer_model_dir, 'model')
        paddle.jit.save(layer=model, path=infer_model_path, input_spec=input_spec)
        logger.info("预测模型已保存：{}".format(infer_model_dir))
