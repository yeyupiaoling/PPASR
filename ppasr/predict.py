import os
from io import BufferedReader

import numpy as np
import yaml

from ppasr import SUPPORT_MODEL
from ppasr.data_utils.audio import AudioSegment
from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer
from ppasr.decoders.ctc_greedy_decoder import greedy_decoder, greedy_decoder_chunk
from ppasr.infer_utils.inference_predictor import InferencePredictor
from ppasr.utils.logger import setup_logger
from ppasr.utils.utils import dict_to_object, print_arguments, download_model

logger = setup_logger(__name__)


class PPASRPredictor:
    def __init__(self,
                 configs=None,
                 model_tag='conformer_streaming_fbank_wenetspeech',
                 model_path='models/conformer_streaming_fbank/infer/',
                 use_pun=False,
                 pun_model_dir='models/pun_models/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置文件路径或者是yaml读取到的配置参数
        :param model_tag: 如果configs为None，则使用项目提供的模型预测
        :param model_path: 导出的预测模型文件夹路径
        :param use_pun: 是否使用加标点符号的模型
        :param pun_model_dir: 给识别结果加标点符号的模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if configs:
            if isinstance(configs, str):
                # 读取配置文件
                with open(configs, 'r', encoding='utf-8') as f:
                    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
                print_arguments(configs=configs)
        else:
            # 使用下载的模型
            cache_dir = os.path.expanduser("~/.cache/ppasr")
            model_url_dict = {
                'conformer_streaming_fbank_wenetspeech': 'https://ppasr.yeyupiaoling.cn/models/conformer_streaming_fbank_wenetspeech.zip'}
            model_url = model_url_dict[model_tag]
            _ = download_model(model_url, cache_dir)
            model_path = os.path.join(cache_dir, model_tag, 'models', model_tag[:model_tag.rfind('_')], 'infer')
            pun_model_dir = os.path.join(cache_dir, model_tag, 'models/pun_models/')
            configs = os.path.join(cache_dir, model_tag, 'configs',
                                   os.listdir(os.path.join(cache_dir, model_tag, 'configs'))[0])
            # 读取配置文件
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            configs['dataset_conf']['dataset_vocab'] = os.path.join(cache_dir, model_tag,
                                                                    configs['dataset_conf']['dataset_vocab'])
            print_arguments(configs=configs)

        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'没有该模型：{self.configs.use_model}'
        self.running = False
        self.inv_normalizer = None
        self.pun_predictor = None
        self.vad_predictor = None
        self._text_featurizer = TextFeaturizer(vocab_filepath=self.configs.dataset_conf.dataset_vocab)
        self._audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        # 流式解码参数
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        self.__init_decoder()
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 加标点符号
        if use_pun:
            from ppasr.infer_utils.pun_predictor import PunctuationPredictor
            self.pun_predictor = PunctuationPredictor(model_dir=pun_model_dir, use_gpu=use_gpu)
        # 获取预测器
        self.predictor = InferencePredictor(configs=self.configs,
                                            use_model=self.configs.use_model,
                                            streaming=self.configs.streaming,
                                            model_dir=model_path,
                                            use_gpu=use_gpu)
        # 预热
        warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(134240,))
        self.predict(audio_data=warmup_audio, is_itn=False)

    # 初始化解码器
    def __init_decoder(self):
        # 集束搜索方法的处理
        if self.configs.decoder == "ctc_beam_search":
            try:
                from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
                self.beam_search_decoder = BeamSearchDecoder(vocab_list=self._text_featurizer.vocab_list,
                                                             **self.configs.ctc_beam_search_decoder_conf)
            except ModuleNotFoundError:
                logger.warning('==================================================================')
                logger.warning('缺少 paddlespeech_ctcdecoders 库，请执行以下命令安装。')
                logger.warning('python -m pip install paddlespeech_ctcdecoders -U -i https://ppasr.yeyupiaoling.cn/pypi/simple/')
                logger.warning('【注意】现在已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.configs.decoder = 'ctc_greedy'

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
            if self.pun_predictor is not None:
                text = self.pun_predictor(text)
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)
        return score, text

    @staticmethod
    def _load_audio(audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        return audio_segment

    # 预测音频
    def predict(self,
                audio_data,
                use_pun=False,
                is_itn=False,
                sample_rate=16000):
        """ 预测函数，只预测完整的一句话
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        audio_segment = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        audio_feature = self._audio_featurizer.featurize(audio_segment)
        input_data = np.array(audio_feature).astype(np.float32)[np.newaxis, :]
        audio_len = np.array([input_data.shape[1]]).astype(np.int64)

        # 运行predictor
        output_data = self.predictor.predict(input_data, audio_len)[0]

        # 解码
        score, text = self.decode(output_data=output_data, use_pun=use_pun, is_itn=is_itn)
        result = {'text': text, 'score': score}
        return result

    # 长语音预测
    def predict_long(self,
                     audio_data,
                     use_pun=False,
                     is_itn=False,
                     sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        if self.vad_predictor is None:
            from ppasr.infer_utils.vad_predictor import VADPredictor
            self.vad_predictor = VADPredictor()
        # 加载音频文件，并进行预处理
        audio_segment = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        # 重采样，方便进行语音活动检测
        if audio_segment.sample_rate != self.configs.preprocess_conf.sample_rate:
            audio_segment.resample(self.configs.preprocess_conf.sample_rate)
        # 获取语音活动区域
        speech_timestamps = self.vad_predictor.get_speech_timestamps(audio_segment.samples, audio_segment.sample_rate)
        texts, scores = '', []
        for t in speech_timestamps:
            audio_ndarray = audio_segment.samples[t['start']: t['end']]
            # 执行识别
            result = self.predict(audio_data=audio_ndarray, use_pun=False, is_itn=is_itn)
            score, text = result['score'], result['text']
            if text != '':
                texts = texts + text if use_pun else texts + '，' + text
            scores.append(score)
            logger.info(f'长语音识别片段结果：{text}')
        if texts[0] == '，': texts = texts[1:]
        # 加标点符号
        if use_pun and len(texts) > 0:
            if self.pun_predictor is not None:
                texts = self.pun_predictor(texts)
            else:
                logger.warning('标点符号模型没有初始化！')
        result = {'text': texts, 'score': round(sum(scores) / len(scores), 2)}
        return result

    # 预测音频
    def predict_stream(self,
                       audio_data,
                       is_end=False,
                       use_pun=False,
                       is_itn=False,
                       channels=1,
                       samp_width=2,
                       sample_rate=16000):
        """
        预测函数，流式预测，通过一直输入音频数据，实现实时识别。
        :param audio_data: 需要预测的音频wave读取的字节流或者未预处理的numpy值
        :param is_end: 是否结束语音识别
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param channels: 如果传入的是pcm字节流数据，需要指定通道数
        :param samp_width: 如果传入的是pcm字节流数据，需要指定音频宽度
        :param sample_rate: 如果传入的是numpy或者pcm字节流数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        if not self.configs.streaming:
            raise Exception(f"不支持改该模型流式识别，当前模型：{self.configs.use_model}")
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, np.ndarray):
            audio_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_data = AudioSegment.from_pcm_bytes(audio_data, channels=channels,
                                                     samp_width=samp_width, sample_rate=sample_rate)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
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
        decoding_chunk_size = 16
        context = 7
        subsampling = 4

        cached_feature_num = context - subsampling
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        stride = subsampling * decoding_chunk_size

        # 保证每帧数据长度都有效
        num_frames = self.cached_feat.shape[1]
        if num_frames < decoding_window and not is_end: return None
        if num_frames < context: return None

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

            # 执行识别
            if self.configs.use_model == 'deepspeech2':
                output_chunk_probs, output_lens = self.predictor.predict_chunk_deepspeech(x_chunk=x_chunk)
            elif 'former' in self.configs.use_model:
                num_decoding_left_chunks = -1
                required_cache_size = decoding_chunk_size * num_decoding_left_chunks
                output_chunk_probs = self.predictor.predict_chunk_conformer(x_chunk=x_chunk,
                                                                            required_cache_size=required_cache_size)
                output_lens = np.array([output_chunk_probs.shape[1]])
            else:
                raise Exception(f'当前模型不支持该方法，当前模型为：{self.configs.use_model}')
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
            if self.pun_predictor is not None:
                text = self.pun_predictor(text)
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)

        result = {'text': text, 'score': score}
        return result

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.predictor.reset_stream()
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        if self.configs.decoder == 'ctc_beam_search':
            self.beam_search_decoder.reset_decoder()

    # 对文本进行反标准化
    def inverse_text_normalization(self, text):
        if self.inv_normalizer is None:
            # 需要安装WeTextProcessing>=0.1.0
            from itn.chinese.inverse_normalizer import InverseNormalizer
            self.inv_normalizer = InverseNormalizer()
        result_text = self.inv_normalizer.normalize(text)
        return result_text
