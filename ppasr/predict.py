import os
import platform
import sys

import cn2an
import numpy as np
import paddle.inference as paddle_infer

from ppasr import SUPPORT_MODEL
from ppasr.data_utils.audio import AudioSegment
from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer
from ppasr.decoders.ctc_greedy_decoder import greedy_decoder, greedy_decoder_chunk
from ppasr.utils.logger import setup_logger

logger = setup_logger(__name__)


class Predictor:
    def __init__(self,
                 model_dir='models/deepspeech2/infer/',
                 vocab_path='dataset/vocabulary.txt',
                 use_model='deepspeech2',
                 decoder='ctc_beam_search',
                 alpha=2.2,
                 beta=4.3,
                 lang_model_path='lm/zh_giga.no_cna_cmn.prune01244.klm',
                 use_pun=False,
                 feature_method='linear',
                 pun_model_dir='models/pun_models/',
                 beam_size=300,
                 cutoff_prob=0.99,
                 cutoff_top_n=40,
                 use_gpu=True,
                 gpu_mem=500,
                 num_threads=10):
        """
        语音识别预测工具
        :param model_dir: 导出的预测模型文件夹路径
        :param vocab_path: 数据集的词汇表文件路径
        :param use_model: 所使用的模型
        :param decoder: 结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)
        :param alpha: 集束搜索解码相关参数，LM系数
        :param beta: 集束搜索解码相关参数，WC系数
        :param lang_model_path: 集束搜索解码相关参数，语言模型文件路径
        :param use_pun: 是否使用加标点符号的模型
        :param feature_method: 所使用的预处理方法
        :param pun_model_dir: 给识别结果加标点符号的模型文件夹路径
        :param beam_size: 集束搜索解码相关参数，搜索的大小，范围建议:[5, 500]
        :param cutoff_prob: 集束搜索解码相关参数，剪枝的概率
        :param cutoff_top_n: 集束搜索解码相关参数，剪枝的最大值
        :param use_gpu: 是否使用GPU预测
        :param gpu_mem: 预先分配的GPU显存大小
        :param num_threads: 只用CPU预测的线程数量
        """
        self.decoder = decoder
        self.use_model = use_model
        self.alpha = alpha
        self.beta = beta
        self.lang_model_path = lang_model_path
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.use_gpu = use_gpu
        self.use_pun_model = use_pun
        self.lac = None
        self._text_featurizer = TextFeaturizer(vocab_filepath=vocab_path)
        self._audio_featurizer = AudioFeaturizer(feature_method=feature_method)
        # 流式解码参数
        self.output_state_h = None
        self.output_state_c = None
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        assert self.use_model in SUPPORT_MODEL, f'没有该模型：{self.use_model}'
        # 模型参数
        if self.use_model == 'deepspeech2':
            self.hidden_size = 1024
        elif self.use_model == 'deepspeech2_big':
            self.hidden_size = 2048

        # 集束搜索方法的处理
        if decoder == "ctc_beam_search":
            if platform.system() != 'Windows':
                try:
                    from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
                    self.beam_search_decoder = BeamSearchDecoder(beam_alpha=self.alpha,
                                                                 beam_beta=self.beta,
                                                                 beam_size=self.beam_size,
                                                                 cutoff_prob=self.cutoff_prob,
                                                                 cutoff_top_n=self.cutoff_top_n,
                                                                 vocab_list=self._text_featurizer.vocab_list,
                                                                 num_processes=1)
                except ModuleNotFoundError:
                    logger.warning('==================================================================')
                    logger.warning('缺少 paddlespeech-ctcdecoders 库，请根据文档安装。')
                    logger.warning('【注意】已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                    logger.warning('==================================================================\n')
                    self.decoder = 'ctc_greedy'
            else:
                logger.warning('==================================================================')
                logger.warning('【注意】Windows不支持ctc_beam_search，已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.decoder = 'ctc_greedy'

        # 创建 config
        model_path = os.path.join(model_dir, 'model.pdmodel')
        params_path = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise Exception("模型文件不存在，请检查%s和%s是否存在！" % (model_path, params_path))
        self.config = paddle_infer.Config(model_path, params_path)

        if self.use_gpu:
            self.config.enable_use_gpu(gpu_mem, 0)
        else:
            self.config.disable_gpu()
            self.config.set_cpu_math_library_num_threads(num_threads)
        # enable memory optim
        self.config.enable_memory_optim()
        self.config.disable_glog_info()

        # 根据 config 创建 predictor
        self.predictor = paddle_infer.create_predictor(self.config)

        logger.info(f'已加载模型：{model_dir}')

        # 获取输入层
        self.audio_data_handle = self.predictor.get_input_handle('audio')
        self.audio_len_handle = self.predictor.get_input_handle('audio_len')
        # 流式模型需要输入RNN的状态
        if 'no_stream' not in self.use_model:
            self.init_state_h_box_handle = self.predictor.get_input_handle('init_state_h_box')
            self.init_state_c_box_handle = self.predictor.get_input_handle('init_state_c_box')

        # 获取输出的名称
        self.output_names = self.predictor.get_output_names()

        # 加标点符号
        if self.use_pun_model:
            from ppasr.utils.text_utils import PunctuationExecutor
            self.pun_executor = PunctuationExecutor(model_dir=pun_model_dir,
                                                    use_gpu=use_gpu,
                                                    gpu_mem=gpu_mem,
                                                    num_threads=num_threads)

        # 预热
        warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(134240,))
        self.predict(audio_ndarray=warmup_audio, to_an=False)

    # 解码模型输出结果
    def decode(self, output_data, to_an):
        """
        解码模型输出结果
        :param output_data: 模型输出结果
        :param to_an: 是否转为阿拉伯数字
        :return:
        """
        # 执行解码
        if self.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            result = self.beam_search_decoder.decode_beam_search_offline(probs_split=output_data)
        else:
            # 贪心解码策略
            result = greedy_decoder(probs_seq=output_data, vocabulary=self._text_featurizer.vocab_list)

        score, text = result[0], result[1]
        # 加标点符号
        if self.use_pun_model and len(text) > 0:
            text = self.pun_executor(text)
        # 是否转为阿拉伯数字
        if to_an:
            text = self.cn2an(text)
        return score, text

    # 预测音频
    def predict(self,
                audio_path=None,
                audio_bytes=None,
                audio_ndarray=None,
                to_an=False):
        """
        预测函数，只预测完整的一句话。
        :param audio_path: 需要预测音频的路径
        :param audio_bytes: 需要预测的音频wave读取的字节流
        :param audio_ndarray: 需要预测的音频未预处理的numpy值
        :param to_an: 是否转为阿拉伯数字
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
        audio_data = np.array(audio_feature).astype('float32')[np.newaxis, :]
        audio_len = np.array([audio_data.shape[2]]).astype('int64')

        # 设置输入
        self.audio_data_handle.reshape([audio_data.shape[0], audio_data.shape[1], audio_data.shape[2]])
        self.audio_len_handle.reshape([audio_data.shape[0]])
        self.audio_data_handle.copy_from_cpu(audio_data)
        self.audio_len_handle.copy_from_cpu(audio_len)

        # 对流式模型RNN层的initial_states全零初始化
        if 'no_stream' not in self.use_model:
            init_state_h_box = np.zeros(shape=(5, audio_data.shape[0], self.hidden_size)).astype('float32')
            self.init_state_h_box_handle.reshape(init_state_h_box.shape)
            self.init_state_h_box_handle.copy_from_cpu(init_state_h_box)
            self.init_state_c_box_handle.reshape(init_state_h_box.shape)
            self.init_state_c_box_handle.copy_from_cpu(init_state_h_box)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_data = output_handle.copy_to_cpu()[0]
        # 解码
        score, text = self.decode(output_data=output_data, to_an=to_an)
        return score, text

    def predict_chunk(self, x_chunk, x_chunk_lens):
        # 设置输入
        self.audio_data_handle.reshape([x_chunk.shape[0], x_chunk.shape[1], x_chunk.shape[2]])
        self.audio_len_handle.reshape([x_chunk.shape[0]])
        self.audio_data_handle.copy_from_cpu(x_chunk.astype('float32'))
        self.audio_len_handle.copy_from_cpu(x_chunk_lens.astype('int64'))

        if self.output_state_h is None or self.output_state_c is None:
            # 对RNN层的initial_states全零初始化
            self.output_state_h = np.zeros(shape=(5, x_chunk.shape[0], self.hidden_size)).astype('float32')
            self.output_state_c = np.zeros(shape=(5, x_chunk.shape[0], self.hidden_size)).astype('float32')
        self.init_state_h_box_handle.reshape(self.output_state_h.shape)
        self.init_state_h_box_handle.copy_from_cpu(self.output_state_h)
        self.init_state_c_box_handle.reshape(self.output_state_c.shape)
        self.init_state_c_box_handle.copy_from_cpu(self.output_state_c)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_chunk_probs = output_handle.copy_to_cpu()
        output_lens_handle = self.predictor.get_output_handle(self.output_names[1])
        output_lens = output_lens_handle.copy_to_cpu()
        output_state_h_handle = self.predictor.get_output_handle(self.output_names[2])
        self.output_state_h = output_state_h_handle.copy_to_cpu()
        output_state_c_handle = self.predictor.get_output_handle(self.output_names[3])
        self.output_state_c = output_state_c_handle.copy_to_cpu()
        return output_chunk_probs, output_lens

    # 预测音频
    def predict_stream(self,
                       audio_bytes=None,
                       audio_ndarray=None,
                       is_end=False,
                       to_an=False):
        """
        预测函数，流式预测，通过一直输入音频数据，实现实现实时识别。
        :param audio_bytes: 需要预测的音频wave读取的字节流
        :param audio_ndarray: 需要预测的音频未预处理的numpy值
        :param is_end: 是否结束语音识别
        :param to_an: 是否转为阿拉伯数字
        :return: 识别得分, 识别结果
        """
        assert 'no_stream' not in self.use_model, f'当前模型不是流式模型，当前模型为：{self.use_model}'
        assert audio_bytes is not None or audio_ndarray is not None, \
            'audio_bytes和audio_ndarray至少有一个不为None！'
        # 加载音频文件
        if audio_ndarray is not None:
            audio_data = AudioSegment.from_ndarray(audio_ndarray)
        else:
            audio_data = AudioSegment.from_wave_bytes(audio_bytes)

        if self.remained_wav is None:
            self.remained_wav = audio_data
        else:
            self.remained_wav = AudioSegment(np.concatenate([self.remained_wav.samples, audio_data.samples]), audio_data.sample_rate)

        # 预处理语音块
        x_chunk = self._audio_featurizer.featurize(self.remained_wav)
        x_chunk = np.array(x_chunk).astype('float32')[np.newaxis, :]
        if self.cached_feat is None:
            self.cached_feat = x_chunk
        else:
            self.cached_feat = np.concatenate([self.cached_feat, x_chunk], axis=2)
        self.remained_wav._samples = self.remained_wav.samples[160 * x_chunk.shape[2]:]

        # 识别的数据块大小
        decoding_chunk_size = 1
        context = 7
        subsampling = 4

        cached_feature_num = context - subsampling
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        stride = subsampling * decoding_chunk_size

        # 保证每帧数据长度都有效
        num_frames = self.cached_feat.shape[2]
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
            x_chunk = self.cached_feat[:, :, cur:end]
            x_chunk_lens = np.array([x_chunk.shape[2]])
            # 执行识别
            output_chunk_probs, output_lens = self.predict_chunk(x_chunk=x_chunk, x_chunk_lens=x_chunk_lens)
            # 执行解码
            if self.decoder == 'ctc_beam_search':
                # 集束搜索解码策略
                score, text = self.beam_search_decoder.decode_chunk(probs=output_chunk_probs, logits_lens=output_lens)
            else:
                # 贪心解码策略
                score, text, self.greedy_last_max_prob_list, self.greedy_last_max_index_list =\
                    greedy_decoder_chunk(probs_seq=output_chunk_probs[0], vocabulary=self._text_featurizer.vocab_list,
                                         last_max_index_list=self.greedy_last_max_index_list,
                                         last_max_prob_list=self.greedy_last_max_prob_list)
        # 更新特征缓存
        self.cached_feat = self.cached_feat[:, :, end - cached_feature_num:]

        # 加标点符号
        if self.use_pun_model and len(text) > 0:
            text = self.pun_executor(text)
        # 是否转为阿拉伯数字
        if to_an:
            text = self.cn2an(text)

        return score, text

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.output_state_h = None
        self.output_state_c = None
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        if self.decoder == 'ctc_beam_search':
            self.beam_search_decoder.reset_decoder()

    # 是否转为阿拉伯数字
    def cn2an(self, text):
        # 获取分词模型
        if self.lac is None:
            from LAC import LAC
            self.lac = LAC(mode='lac', use_cuda=self.use_gpu)
        lac_result = self.lac.run(text)
        result_text = ''
        for t, r in zip(lac_result[0], lac_result[1]):
            if r == 'm' or r == 'TIME':
                t = cn2an.transform(t, "cn2an")
            result_text += t
        return result_text
