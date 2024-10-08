import json
import os
import platform
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import paddle
import yaml
from paddle.distributed import fleet
from paddle.io import DataLoader
from tqdm import tqdm
from visualdl import LogWriter
from loguru import logger

from ppasr.data_utils.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.collate_fn import collate_fn
from ppasr.data_utils.normalizer import FeatureNormalizer
from ppasr.data_utils.reader import PPASRDataset
from ppasr.data_utils.sampler import SortagradBatchSampler, SortagradDistributedBatchSampler
from ppasr.data_utils.tokenizer import PPASRTokenizer
from ppasr.data_utils.utils import create_manifest_binary
from ppasr.decoders.attention_rescoring import attention_rescoring
from ppasr.decoders.ctc_greedy_search import ctc_greedy_search
from ppasr.decoders.ctc_prefix_beam_search import ctc_prefix_beam_search
from ppasr.model_utils import build_model
from ppasr.optimizer import build_lr_scheduler, build_optimizer
from ppasr.utils.checkpoint import save_checkpoint, load_pretrained, load_checkpoint
from ppasr.utils.metrics import cer, wer, mer
from ppasr.utils.summary import summary
from ppasr.utils.utils import dict_to_object, print_arguments
from ppasr.data_utils.utils import create_manifest, merge_audio


class PPASRTrainer(object):
    def __init__(self,
                 configs,
                 use_gpu=True,
                 metrics_type="cer",
                 decoder="ctc_greedy_search",
                 decoder_configs=None,
                 data_augment_configs=None):
        """ PPASR集成工具类

        :param configs: 配置文件路径或者是yaml读取到的配置参数
        :type configs: dict or str
        :param use_gpu: 是否使用GPU训练模型
        :type use_gpu: bool
        :param metrics_type: 评估指标类型，中文用cer，英文用wer，中英混合用mer
        :type metrics_type: str
        :param decoder: 解码器，支持ctc_greedy、ctc_beam_search
        :type decoder: str
        :param decoder_configs: 解码器配置参数文件路径，支持yaml格式
        :type decoder_configs: dict or str
        :param data_augment_configs: 数据增强配置字典或者其文件路径
        :type data_augment_configs: dict or str
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        # 读取数据增强配置文件
        if isinstance(data_augment_configs, str):
            with open(data_augment_configs, 'r', encoding='utf-8') as f:
                data_augment_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=data_augment_configs, title='数据增强配置')
        self.data_augment_configs = dict_to_object(data_augment_configs)
        self.local_rank = 0
        self.use_gpu = use_gpu
        self.metrics_type = metrics_type
        self.decoder = decoder
        if self.configs.model_conf.model == 'DeepSpeech2Model':
            assert self.decoder == "ctc_greedy_search", f'{self.configs.model_conf.model}模型不支持{self.decoder}解码器'
        # 读取解码器配置文件
        if isinstance(decoder_configs, str) and os.path.exists(decoder_configs):
            with open(decoder_configs, 'r', encoding='utf-8') as f:
                decoder_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=decoder_configs, title='解码器参数配置')
        self.decoder_configs = decoder_configs if decoder_configs is not None else {}
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.audio_featurizer = None
        self.tokenizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None
        self.beam_search_decoder = None
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_eta_sec = None, None
        self.eval_best_error_rate = None
        self.eval_loss, self.eval_error_result = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False

    def __setup_dataloader(self, is_train=False, max_text_duration=None):
        """ 获取数据加载器

        :param is_train: 是否获取训练数据
        :type is_train: bool
        :param max_text_duration: 测试过滤的最大音频时长，如果不指定，则使用配置文件里面的max_duration
        :type max_text_duration: int
        """
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        self.tokenizer = PPASRTokenizer(**self.configs.tokenizer_conf)
        # 判断是否有归一化文件
        if not os.path.exists(self.configs.dataset_conf.mean_istd_path):
            raise Exception(f'归一化列表文件 {self.configs.dataset_conf.mean_istd_path} 不存在')

        dataset_args = self.configs.dataset_conf.get('dataset', {})
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        if is_train:
            self.train_dataset = PPASRDataset(data_manifest=self.configs.dataset_conf.train_manifest,
                                              audio_featurizer=self.audio_featurizer,
                                              tokenizer=self.tokenizer,
                                              aug_conf=self.data_augment_configs,
                                              mode='train',
                                              **dataset_args)
            # 设置支持多卡训练
            if paddle.distributed.get_world_size() > 1 and self.use_gpu:
                self.train_batch_sampler = SortagradDistributedBatchSampler(self.train_dataset,
                                                                            **self.configs.dataset_conf.batch_sampler)
            else:
                self.train_batch_sampler = SortagradBatchSampler(self.train_dataset,
                                                                 **self.configs.dataset_conf.batch_sampler)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           batch_sampler=self.train_batch_sampler,
                                           **data_loader_args)
        # 获取测试数据
        if max_text_duration is not None:
            dataset_args.max_duration = max_text_duration
        self.test_dataset = PPASRDataset(data_manifest=self.configs.dataset_conf.test_manifest,
                                         audio_featurizer=self.audio_featurizer,
                                         tokenizer=self.tokenizer,
                                         mode='eval',
                                         **dataset_args)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.configs.dataset_conf.batch_sampler.batch_size,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **data_loader_args)

    # 提取特征保存文件
    def extract_features(self, save_dir='dataset/features', max_duration=100):
        """ 提取特征保存文件

        :param save_dir: 保存路径
        :type save_dir: str
        :param max_duration: 提取特征的最大时长，单位秒
        :type max_duration: int
        """
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        # 读取训练数据列表和测试数据列表
        for data_list_file in [self.configs.dataset_conf.train_manifest, self.configs.dataset_conf.test_manifest]:
            save_dir1 = os.path.join(save_dir, data_list_file.split('.')[-1])
            os.makedirs(save_dir1, exist_ok=True)
            dataset_args = self.configs.dataset_conf.get('dataset', {})
            dataset_args.max_duration = max_duration
            test_dataset = PPASRDataset(data_manifest=data_list_file,
                                        audio_featurizer=self.audio_featurizer,
                                        mode='eval',
                                        **dataset_args)
            # 保存文件夹
            save_dir_num = f'{int(time.time())}'
            os.makedirs(os.path.join(str(save_dir1), save_dir_num), exist_ok=True)
            all_feature, time_sum, index = None, 0, 0
            # 新的数据列表文件
            save_data_list = data_list_file.replace('manifest', 'manifest_features')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset)), desc=f'[{data_list_file}]提取特征中...'):
                    feature = test_dataset[i].numpy()
                    data_list = test_dataset.get_one_list(idx=i)
                    time_sum += data_list['duration']
                    if all_feature is None:
                        index += 1
                        all_feature = feature
                        if index >= 1000:
                            index = 0
                            save_dir_num = f'{int(time.time())}'
                            os.makedirs(os.path.join(str(save_dir1), save_dir_num), exist_ok=True)
                        save_path = os.path.join(str(save_dir1), save_dir_num,
                                                 f'{int(time.time() * 1000)}.npy').replace('\\', '/')
                    else:
                        all_feature = np.concatenate((all_feature, feature), axis=0)
                    new_data_list = {"audio_filepath": save_path,
                                     "duration": data_list['duration'],
                                     "text": data_list['text'],
                                     "start_frame": all_feature.shape[0] - feature.shape[0],
                                     "end_frame": all_feature.shape[0]}
                    f.write(f'{json.dumps(new_data_list, ensure_ascii=False)}\n')
                    # 把每10分钟的特征合并在一起
                    if time_sum > 600:
                        np.save(save_path, all_feature)
                        all_feature, time_sum = None, 0
                if all_feature is not None:
                    np.save(save_path, all_feature)
            logger.info(f'[{data_list_file}]列表中的数据已提取特征完成，新列表为：[{save_data_list}]')

    def __setup_model(self, input_dim, tokenizer, is_train=False):
        """初始化模型

        :param input_dim: 模型的输入维度
        :type input_dim: int
        :param tokenizer: 文本分词器
        :type tokenizer: MASRTokenizer
        :param is_train: 是否是训练模式
        :type is_train: bool
        """
        # 获取模型
        self.model = build_model(input_size=input_dim,
                                 vocab_size=tokenizer.vocab_size,
                                 mean_istd_path=self.configs.dataset_conf.mean_istd_path,
                                 eos_id=tokenizer.eos_id,
                                 encoder_conf=self.configs.get('encoder_conf', None),
                                 decoder_conf=self.configs.get('decoder_conf', None),
                                 model_conf=self.configs.model_conf)
        # print(self.model)
        if is_train:
            summary(self.model, inputs=[paddle.rand([1, 260, self.audio_featurizer.feature_dim]),
                                        paddle.to_tensor([260], dtype=paddle.int64),
                                        paddle.randint(low=0, high=self.tokenizer.vocab_size - 1, shape=[1, 10],
                                                       dtype=paddle.int32),
                                        paddle.to_tensor([10], dtype=paddle.int64)])
            if self.configs.train_conf.enable_amp:
                # 自动混合精度训练，逻辑2，定义GradScaler
                self.amp_scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.configs.train_conf.grad_clip)
            # 学习率衰减函数
            self.scheduler = build_lr_scheduler(step_per_epoch=len(self.train_loader), configs=self.configs)
            # 获取优化方法
            self.optimizer = build_optimizer(params=self.model.parameters(), scheduler=self.scheduler,
                                             grad_clip=grad_clip, configs=self.configs)

    def __decoder_result(self, encoder_outs, ctc_probs, ctc_lens):
        """解码结果

        :param encoder_outs: 编码器输出
        :param ctc_probs: 模型输出的CTC概率
        :param ctc_lens: 模型输出的CTC长度
        :return: 解码结果
        """
        if self.decoder == "ctc_greedy_search":
            result = ctc_greedy_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens, blank_id=self.tokenizer.blank_id)
        elif self.decoder == "ctc_prefix_beam_search":
            decoder_args = self.decoder_configs.get('ctc_prefix_beam_search_args', {})
            result, _ = ctc_prefix_beam_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens,
                                               blank_id=self.tokenizer.blank_id, **decoder_args)
        elif self.decoder == "attention_rescoring":
            decoder_args = self.decoder_configs.get('attention_rescoring_args', {})
            result = attention_rescoring(model=self.model,
                                         ctc_probs=ctc_probs,
                                         ctc_lens=ctc_lens,
                                         blank_id=self.tokenizer.blank_id,
                                         encoder_outs=encoder_outs,
                                         encoder_lens=ctc_lens,
                                         **decoder_args)
        else:
            raise ValueError(f"不支持该解码器：{self.decoder}")
        text = self.tokenizer.ids2text([r for r in result])
        return text

    def __train_epoch(self, epoch_id, save_model_path, writer):
        """训练一轮数据

        :param epoch_id: 当前训练轮次
        :type epoch_id: int
        :param save_model_path: 保存模型路径
        :type save_model_path: str
        :param writer: 日志记录器
        :type writer: LogWriter
        """
        train_times, reader_times, batch_times, loss_sum = [], [], [], []
        start = time.time()
        for batch_id, batch in enumerate(self.train_loader()):
            if self.stop_train: break
            inputs, labels, input_lens, label_lens = batch
            reader_times.append((time.time() - start) * 1000)
            start_step = time.time()
            num_utts = label_lens.shape[0]
            if num_utts == 0:
                continue
            # 执行模型计算，是否开启自动混合精度
            with paddle.amp.auto_cast(enable=self.configs.train_conf.enable_amp, level='O1'):
                loss_dict = self.model(inputs, input_lens, labels, label_lens)
            # loss backward
            if paddle.distributed.get_world_size() > 1 and batch_id % self.configs.train_conf.accum_grad != 0:
                context = self.model.no_sync
            else:
                context = nullcontext
            with context():
                loss = loss_dict['loss'] / self.configs.train_conf.accum_grad
                # 是否开启自动混合精度
                if self.configs.train_conf.enable_amp:
                    # loss缩放，乘以系数loss_scaling
                    scaled = self.amp_scaler.scale(loss)
                    scaled.backward()
                else:
                    loss.backward()

            # 执行一次梯度计算
            if batch_id % self.configs.train_conf.accum_grad == 0:
                # 是否开启自动混合精度
                if self.configs.train_conf.enable_amp:
                    # 更新参数（参数梯度先除系数loss_scaling再更新参数）
                    self.amp_scaler.step(self.optimizer)
                    # 基于动态loss_scaling策略更新loss_scaling系数
                    self.amp_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.clear_grad()
                self.scheduler.step()
                self.train_step += 1
            loss_sum.append(float(loss))
            train_times.append((time.time() - start) * 1000)
            batch_times.append((time.time() - start_step) * 1000)
            self.train_step += 1

            # 多卡训练只使用一个进程打印
            if batch_id % self.configs.train_conf.log_interval == 0:
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.batch_sampler.batch_size / (
                            sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, '
                            f'learning_rate: {self.scheduler.get_lr():>.8f}, '
                            f'reader_cost: {(sum(reader_times) / len(reader_times) / 1000):.4f}, '
                            f'batch_cost: {(sum(batch_times) / len(batch_times) / 1000):.4f}, '
                            f'ips: {train_speed:.4f} speech/sec, '
                            f'eta: {eta_str}')
                if self.local_rank == 0:
                    # 记录学习率
                    writer.add_scalar('Train/lr', self.scheduler.get_lr(), self.train_log_step)
                    writer.add_scalar('Train/Loss', float(loss), self.train_log_step)
                self.train_log_step += 1
                train_times, reader_times, batch_times, loss_sum = [], [], [], []
            # 固定步数也要保存一次模型
            if batch_id % 10000 == 0 and batch_id != 0 and self.local_rank == 0:
                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id)
            start = time.time()

    def create_data(self,
                    annotation_path='dataset/annotation/',
                    num_samples=1000000,
                    max_test_manifest=10000,
                    is_merge_audio=False,
                    only_build_vocab=False,
                    save_audio_path='dataset/audio/merge_audio',
                    max_duration=600):
        """创建数据列表和词汇表模型等前期数据处理

        :param annotation_path: 标注文件的路径
        :type annotation_path: str
        :param num_samples: 用于计算均值和标准值得音频数量，当为-1使用全部数据
        :type num_samples: int
        :param max_test_manifest: 生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据
        :type max_test_manifest: int
        :param only_build_vocab: 是否只生成词汇表模型文件，不进行其他操作
        :type only_build_vocab: bool
        :param is_merge_audio: 是否将多个短音频合并成长音频，以减少音频文件数量，注意自动删除原始音频文件
        :type is_merge_audio: bool
        :param save_audio_path: 合并音频的保存路径
        :type save_audio_path: str
        :param max_duration: 合并音频的最大长度，单位秒
        :type max_duration: int
        """
        if is_merge_audio:
            logger.info('开始合并音频...')
            merge_audio(annotation_path=annotation_path, save_audio_path=save_audio_path, max_duration=max_duration,
                        target_sr=self.configs.dataset_conf.dataset.sample_rate)
            logger.info('合并音频已完成，原始音频文件和标注文件已自动删除，其他原始文件可手动删除！')

        if not only_build_vocab:
            logger.info('开始生成数据列表...')
            create_manifest(annotation_path=annotation_path,
                            train_manifest_path=self.configs.dataset_conf.train_manifest,
                            test_manifest_path=self.configs.dataset_conf.test_manifest,
                            max_test_manifest=max_test_manifest)
            logger.info('=' * 70)

            normalizer = FeatureNormalizer(mean_istd_filepath=self.configs.dataset_conf.mean_istd_path)
            normalizer.compute_mean_istd(manifest_path=self.configs.dataset_conf.train_manifest,
                                         preprocess_conf=self.configs.preprocess_conf,
                                         data_loader_conf=self.configs.dataset_conf.dataLoader,
                                         num_samples=num_samples)
            print('计算的均值和标准值已保存在 %s！' % self.configs.dataset_conf.mean_istd_path)

        logger.info('=' * 70)
        logger.info('开始生成数据字典...')
        tokenizer = PPASRTokenizer(is_build_vocab=True, **self.configs.tokenizer_conf)
        tokenizer.build_vocab(manifest_paths=[self.configs.dataset_conf.train_manifest,
                                              self.configs.dataset_conf.test_manifest])
        logger.info('数据字典生成完成！')

        if self.configs.dataset_conf.dataset.manifest_type == 'binary':
            logger.info('=' * 70)
            logger.info('正在生成数据列表的二进制文件...')
            create_manifest_binary(train_manifest_path=self.configs.dataset_conf.train_manifest,
                                   test_manifest_path=self.configs.dataset_conf.test_manifest)
            logger.info('数据列表的二进制文件生成完成！')

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
        paddle.jit.enable_to_static(False)
        # 获取有多少张显卡训练
        nranks = paddle.distributed.get_world_size()
        self.local_rank = paddle.distributed.get_rank()
        writer = None
        if self.local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # 初始化Fleet环境
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)

        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_dim=self.audio_featurizer.feature_dim,
                           tokenizer=self.tokenizer,
                           is_train=True)
        # 加载预训练模型
        self.model = load_pretrained(model=self.model, pretrained_model=pretrained_model)
        # 加载恢复模型
        self.model, self.optimizer, self.amp_scaler, self.scheduler, last_epoch, self.eval_best_error_rate = \
            load_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                            amp_scaler=self.amp_scaler, scheduler=self.scheduler, step_epoch=len(self.train_loader),
                            save_model_path=save_model_path, resume_model=resume_model)
        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.model = fleet.distributed_model(self.model)
        if self.local_rank == 0:
            logger.info(f'训练数据：{len(self.train_dataset)}，词汇表大小：{self.tokenizer.vocab_size}')

        self.train_loss, self.eval_loss = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.train_batch_sampler.epoch = last_epoch
        if self.local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_lr(), last_epoch)
        # 最大步数
        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, save_model_path=save_model_path, writer=writer)
            # 多卡训练只使用一个进程执行评估和保存模型
            logger.info('=' * 70)
            if self.stop_eval: continue
            self.eval_loss, self.eval_error_result = self.evaluate()
            logger.info(
                f'Test epoch: {epoch_id}, time/epoch: {str(timedelta(seconds=(time.time() - start_epoch)))}, '
                f'loss: {self.eval_loss:.5f}, {self.metrics_type}: {self.eval_error_result:.5f}, '
                f'best {self.metrics_type}: '
                f'{self.eval_error_result if self.eval_error_result <= self.eval_best_error_rate else self.eval_best_error_rate:.5f}')
            logger.info('=' * 70)
            writer.add_scalar(f'Test/{self.metrics_type}', self.eval_error_result, self.test_log_step)
            writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
            self.test_log_step += 1
            self.model.train()
            # 保存最优模型
            if self.eval_error_result <= self.eval_best_error_rate:
                self.eval_best_error_rate = self.eval_error_result
                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                error_rate=self.eval_error_result, metrics_type=self.metrics_type,
                                best_model=True)
            # 保存模型
            save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                            amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                            error_rate=self.eval_error_result, metrics_type=self.metrics_type)

    def evaluate(self, resume_model=None, display_result=False, max_text_duration=None):
        """评估模型

        :param resume_model: 所使用的模型
        :type resume_model: str
        :param display_result: 是否打印识别结果
        :type display_result: bool
        :param max_text_duration: 测试过滤的最大音频时长，如果不指定，则使用配置文件里面的max_duration
        :type max_text_duration: int
        :return: 评估结果
        """
        paddle.jit.enable_to_static(False)
        if self.test_loader is None:
            # 获取测试数据
            self.__setup_dataloader(max_text_duration=max_text_duration)
        if self.model is None:
            # 获取模型
            self.__setup_model(input_dim=self.audio_featurizer.feature_dim, tokenizer=self.tokenizer)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pdparams')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = paddle.load(resume_model)
            self.model.set_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, paddle.DataParallel):
            eval_model = self.model._layers
        else:
            eval_model = self.model

        error_results, losses = [], []
        with paddle.no_grad():
            for batch_id, batch in enumerate(tqdm(self.test_loader())):
                if self.stop_eval: break
                inputs, labels, input_lens, label_lens = batch
                loss_dict = self.model(inputs, input_lens, labels, label_lens)
                losses.append(float(loss_dict['loss']) / self.configs.train_conf.accum_grad)
                # 获取模型编码器输出
                encoder_outs, ctc_probs, ctc_lens = eval_model.get_encoder_out(inputs, input_lens)
                out_strings = self.__decoder_result(encoder_outs=encoder_outs, ctc_probs=ctc_probs, ctc_lens=ctc_lens)
                labels = labels.numpy().tolist()
                # 移除每条数据的-1值
                labels = [list(filter(lambda x: x != -1, label)) for label in labels]
                labels_str = self.tokenizer.ids2text(labels)
                for out_string, label in zip(*(out_strings, labels_str)):
                    # 计算字错率或者词错率
                    if self.metrics_type == 'wer':
                        error_rate = wer(label, out_string)
                    elif self.metrics_type == 'mer':
                        error_rate = mer(label, out_string)
                    else:
                        error_rate = cer(label, out_string)
                    error_results.append(error_rate)
                    if display_result:
                        logger.info(f'预测结果为：{out_string}')
                        logger.info(f'实际标签为：{label}')
                        logger.info(f'这条数据的{self.metrics_type}：{round(error_rate, 6)}，'
                                    f'当前{self.metrics_type}：{round(sum(error_results) / len(error_results), 6)}')
                        logger.info('-' * 70)
        loss = float(sum(losses) / len(losses)) if len(losses) > 0 else -1
        error_result = float(sum(error_results) / len(error_results)) if len(error_results) > 0 else -1
        self.model.train()
        return loss, error_result

    def export(self,
               save_model_path='models/',
               resume_model='models/conformer_streaming_fbank/best_model/',
               save_quant=False):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :param save_quant: 是否保存量化模型
        :return:
        """
        paddle.jit.enable_to_static(True)
        # 获取训练数据
        audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        tokenizer = PPASRTokenizer(**self.configs.tokenizer_conf)
        if not os.path.exists(self.configs.dataset_conf.mean_istd_path):
            raise Exception(f'归一化列表文件 {self.configs.dataset_conf.mean_istd_path} 不存在')
        # 获取模型
        self.__setup_model(input_dim=audio_featurizer.feature_dim, tokenizer=tokenizer)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pdparams')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = paddle.load(resume_model)
        self.model.set_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        # 保存模型的路径
        save_feature_method = self.configs.preprocess_conf.feature_method
        save_model_name = f'{self.configs.model_conf.model}_{save_feature_method}/inference_model'
        save_model_dir = os.path.join(save_model_path, save_model_name)
        infer_model_path = os.path.join(save_model_dir, 'inference')
        shutil.rmtree(save_model_dir, ignore_errors=True)
        os.makedirs(save_model_dir, exist_ok=True)
        paddle.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
        # 保存量化模型
        if save_quant:
            import paddleslim
            paddle.enable_static()
            paddleslim.quant.quant_post_dynamic(model_dir=os.path.dirname(infer_model_path),
                                                save_model_dir=os.path.dirname(infer_model_path),
                                                model_filename='inference.pdmodel',
                                                params_filename='inference.pdiparams',
                                                save_model_filename='inference.pdmodel',
                                                save_params_filename='inference.pdiparams')
            logger.info("量化模型已保存：{}".format(os.path.join(os.path.dirname(infer_model_path), 'quantized_model')))
        # 复制词汇表模型
        shutil.copytree(tokenizer.vocab_model_dir, os.path.join(save_model_dir, 'vocab_model'))
        # 保存配置信息
        with open(os.path.join(save_model_path, save_model_name, 'inference.json'), 'w', encoding="utf-8") as f:
            self.configs.tokenizer_conf.token_list = tokenizer.vocab_list
            inference_config = {
                'model_name': self.configs.model_conf.model,
                'streaming': self.configs.model_conf.model_args.streaming,
                'sample_rate': self.configs.dataset_conf.dataset.sample_rate,
                'preprocess_conf': self.configs.preprocess_conf
            }
            json.dump(inference_config, f, indent=4, ensure_ascii=False)
