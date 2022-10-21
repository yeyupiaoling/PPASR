import os
import platform

import numpy as np
import paddle
from ppasr import SUPPORT_MODEL
from ppasr.data_utils.audio import AudioSegment
from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer
from ppasr.decoders.ctc_greedy_decoder import greedy_decoder, greedy_decoder_chunk
from ppasr.model_utils.conformer.model import ConformerModelOffline, ConformerModelOnline
from ppasr.model_utils.deepspeech2.model import DeepSpeech2ModelOffline, DeepSpeech2ModelOnline
from ppasr.utils.logger import setup_logger
from ppasr.utils.utils import dict_to_object

logger = setup_logger(__name__)


class Predictor:
    def __init__(self,
                 configs,
                 model_path='models/conformer_online_fbank/best_model/',
                 use_pun=False,
                 pun_model_dir='models/pun_models/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param model_path: 导出的预测模型文件夹路径
        :param use_pun: 是否使用加标点符号的模型
        :param pun_model_dir: 给识别结果加标点符号的模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        self.configs = dict_to_object(configs)
        self.running = False
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self.pun_executor = None
        self._text_featurizer = TextFeaturizer(vocab_filepath=self.configs.dataset_conf.dataset_vocab)
        self._audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        # 流式解码参数
        self.output_state_h = None
        self.output_state_c = None
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'

        # 集束搜索方法的处理
        if self.configs.decoder == "ctc_beam_search":
            if platform.system() != 'Windows':
                try:
                    from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
                    self.beam_search_decoder = BeamSearchDecoder(vocab_list=self._text_featurizer.vocab_list,
                                                                 **self.configs.ctc_beam_search_decoder_conf)
                except ModuleNotFoundError:
                    logger.warning('==================================================================')
                    logger.warning('缺少 paddlespeech-ctcdecoders 库，请根据文档安装。')
                    logger.warning('【注意】已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                    logger.warning('==================================================================\n')
                    self.configs.decoder = 'ctc_greedy'
            else:
                logger.warning('==================================================================')
                logger.warning(
                    '【注意】Windows不支持ctc_beam_search，已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.configs.decoder = 'ctc_greedy'

        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 根据 config 创建 predictor
        if self.configs.use_model == 'conformer_online':
            self.predictor = ConformerModelOnline(configs=self.configs,
                                                  input_dim=self._audio_featurizer.feature_dim,
                                                  vocab_size=self._text_featurizer.vocab_size,
                                                  **self.configs.model_conf)
        elif self.configs.use_model == 'conformer_offline':
            self.predictor = ConformerModelOffline(configs=self.configs,
                                                   input_dim=self._audio_featurizer.feature_dim,
                                                   vocab_size=self._text_featurizer.vocab_size,
                                                   **self.configs.model_conf)
        elif self.configs.use_model == 'deepspeech2_online':
            self.predictor = DeepSpeech2ModelOnline(configs=self.configs,
                                                    input_dim=self._audio_featurizer.feature_dim,
                                                    vocab_size=self._text_featurizer.vocab_size)
        elif self.configs.use_model == 'deepspeech2_offline':
            self.predictor = DeepSpeech2ModelOffline(configs=self.configs,
                                                     input_dim=self._audio_featurizer.feature_dim,
                                                     vocab_size=self._text_featurizer.vocab_size, )
        else:
            raise Exception('没有该模型：{}'.format(self.configs.use_model))
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pdparams')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        model_state_dict = paddle.load(model_path)
        self.predictor.set_state_dict(model_state_dict)
        logger.info(f'成功加载模型：{model_path}')
        self.predictor.eval()

        # 加标点符号
        if use_pun:
            from ppasr.utils.text_utils import PunctuationExecutor
            self.pun_executor = PunctuationExecutor(model_dir=pun_model_dir,
                                                    use_gpu=use_gpu,
                                                    gpu_mem=gpu_mem,
                                                    num_threads=num_threads)

        # 预热
        warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(134240,))
        self.predict(audio_ndarray=warmup_audio, is_itn=False)

    # 解码模型输出结果
    def decode(self, output_data, use_pun, is_itn):
        """
        解码模型输出结果
        :param output_data: 模型输出结果
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :return:
        """
        # 执行解码
        if self.configs.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            result = self.beam_search_decoder.decode_beam_search_offline(probs_split=output_data)
        else:
            # 贪心解码策略
            result = greedy_decoder(probs_seq=output_data, vocabulary=self._text_featurizer.vocab_list)

        score, text = result[0], result[1]
        # 加标点符号
        if use_pun and len(text) > 0:
            if self.pun_executor is not None:
                text = self.pun_executor(text)
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)
        return score, text

    # 预测音频
    def predict(self,
                audio_path=None,
                audio_bytes=None,
                audio_ndarray=None,
                use_pun=False,
                is_itn=True):
        """
        预测函数，只预测完整的一句话。
        :param audio_path: 需要预测音频的路径
        :param audio_bytes: 需要预测的音频wave读取的字节流
        :param audio_ndarray: 需要预测的音频未预处理的numpy值
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :return: 识别的文本结果和解码的得分数
        """
        assert audio_path is not None or audio_bytes is not None or audio_ndarray is not None, \
            'audio_path，audio_bytes和audio_ndarray至少有一个不为None！'
        # 加载音频文件，并进行预处理
        if audio_path is not None:
            audio_data = AudioSegment.from_file(audio_path)
        elif audio_ndarray is not None:
            audio_data = AudioSegment.from_ndarray(audio_ndarray)
        else:
            audio_data = AudioSegment.from_wave_bytes(audio_bytes)
        audio_feature = self._audio_featurizer.featurize(audio_data)
        audio_data = np.array(audio_feature).astype(np.float32)[np.newaxis, :]
        audio_len = np.array([audio_data.shape[1]]).astype(np.int64)

        audio_data = paddle.to_tensor(audio_data, dtype=paddle.float32)
        audio_len = paddle.to_tensor(audio_len)

        if self.use_gpu:
            audio_data = audio_data.cuda()
            audio_len = audio_len.cuda()

        # 运行predictor
        output_data = self.predictor.get_encoder_out(audio_data, audio_len).numpy()[0]

        # 解码
        score, text = self.decode(output_data=output_data, use_pun=use_pun, is_itn=is_itn)
        return score, text

    def predict_chunk(self, x_chunk, x_chunk_lens):
        audio_data = paddle.to_tensor(x_chunk, dtype=paddle.float32)
        audio_len = paddle.to_tensor(x_chunk_lens)

        if self.use_gpu:
            audio_data = audio_data.cuda()
            audio_len = audio_len.cuda()

        # 运行predictor
        output_chunk_probs, output_lens, self.output_state_h, self.output_state_c = \
            self.predictor.get_encoder_out_chunk(audio_data, audio_len, self.output_state_h, self.output_state_c)
        output_chunk_probs = output_chunk_probs.cpu().detach().numpy()
        output_lens = output_lens.cpu().detach().numpy()
        return output_chunk_probs, output_lens

    # 预测音频
    def predict_stream(self,
                       audio_bytes=None,
                       audio_ndarray=None,
                       is_end=False,
                       use_pun=False,
                       is_itn=True):
        """
        预测函数，流式预测，通过一直输入音频数据，实现实时识别。
        :param audio_bytes: 需要预测的音频wave读取的字节流
        :param audio_ndarray: 需要预测的音频未预处理的numpy值
        :param is_end: 是否结束语音识别
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :return: 识别的文本结果和解码的得分数
        """
        if self.configs.use_model != 'deepspeech2_online':
            raise Exception(f"不支持：{self.configs.use_model}")
        assert audio_bytes is not None or audio_ndarray is not None, \
            'audio_bytes和audio_ndarray至少有一个不为None！'
        # 加载音频文件，并进行预处理
        if audio_ndarray is not None:
            audio_data = AudioSegment.from_ndarray(audio_ndarray)
        else:
            audio_data = AudioSegment.from_wave_bytes(audio_bytes)

        if self.remained_wav is None:
            self.remained_wav = audio_data
        else:
            self.remained_wav = AudioSegment(np.concatenate([self.remained_wav.samples, audio_data.samples]),
                                             audio_data.sample_rate)

        # 预处理语音块
        x_chunk = self._audio_featurizer.featurize(self.remained_wav)
        x_chunk = np.array(x_chunk).astype(np.float32)[np.newaxis, :]
        if self.cached_feat is None:
            self.cached_feat = x_chunk
        else:
            self.cached_feat = np.concatenate([self.cached_feat, x_chunk], axis=1)
        self.remained_wav._samples = self.remained_wav.samples[160 * x_chunk.shape[1]:]

        # 识别的数据块大小
        decoding_chunk_size = 1
        context = 7
        subsampling = 4

        cached_feature_num = context - subsampling
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        stride = subsampling * decoding_chunk_size

        # 保证每帧数据长度都有效
        num_frames = self.cached_feat.shape[1]
        if num_frames < decoding_window and not is_end: return 0, ''
        if num_frames < context: return 0, ''

        # 如果识别结果，要使用最后一帧
        if is_end:
            left_frames = context
        else:
            left_frames = decoding_window

        score, text, end = None, None, None
        for cur in range(0, num_frames - left_frames + 1, stride):
            end = min(cur + decoding_window, num_frames)
            # 获取数据块
            x_chunk = self.cached_feat[:, cur:end, :]
            x_chunk_lens = np.array([x_chunk.shape[1]])
            # 执行识别
            output_chunk_probs, output_lens = self.predict_chunk(x_chunk=x_chunk, x_chunk_lens=x_chunk_lens)
            # 执行解码
            if self.configs.decoder == 'ctc_beam_search':
                # 集束搜索解码策略
                score, text = self.beam_search_decoder.decode_chunk(probs=output_chunk_probs, logits_lens=output_lens)
            else:
                # 贪心解码策略
                score, text, self.greedy_last_max_prob_list, self.greedy_last_max_index_list = \
                    greedy_decoder_chunk(probs_seq=output_chunk_probs[0], vocabulary=self._text_featurizer.vocab_list,
                                         last_max_index_list=self.greedy_last_max_index_list,
                                         last_max_prob_list=self.greedy_last_max_prob_list)
        # 更新特征缓存
        self.cached_feat = self.cached_feat[:, end - cached_feature_num:, :]

        # 加标点符号
        if use_pun and is_end and len(text) > 0:
            if self.pun_executor is not None:
                text = self.pun_executor(text)
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)

        return score, text

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.output_state_h = None
        self.output_state_c = None
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        if self.configs.decoder == 'ctc_beam_search':
            self.beam_search_decoder.reset_decoder()

    # 对文本进行反标准化
    def inverse_text_normalization(self, text):
        if self.configs.decoder == 'ctc_beam_search':
            logger.error("当解码器为ctc_beam_search时，因为包冲突，不能使用文本反标准化")
            return text
        if self.inv_normalizer is None:
            from itn.chinese.inverse_normalizer import InverseNormalizer
            self.inv_normalizer = InverseNormalizer()
        result_text = self.inv_normalizer.normalize(text)
        return result_text
