import json
import os
from io import BufferedReader

import numpy as np
import paddle
import yaml
from loguru import logger
from yeaudio.audio import AudioSegment
from yeaudio.streaming_vad import StreamingVAD, VADParams, VADState

from ppasr.data_utils.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.tokenizer import PPASRTokenizer
from ppasr.decoders.attention_rescoring import attention_rescoring
from ppasr.decoders.ctc_greedy_search import ctc_greedy_search
from ppasr.decoders.ctc_prefix_beam_search import ctc_prefix_beam_search
from ppasr.infer_utils.inference_predictor import InferencePredictor
from ppasr.utils.utils import dict_to_object, print_arguments


class PPASRPredictor:
    def __init__(self,
                 model_dir='models/ConformerModel_fbank/inference_model/',
                 decoder="ctc_greedy",
                 decoder_configs=None,
                 use_pun=False,
                 pun_model_dir='models/pun_models/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param model_dir: 导出的预测模型文件夹路径
        :param decoder: 解码器，支持ctc_greedy、ctc_beam_search
        :param decoder_configs: 解码器配置参数文件路径，支持yaml格式
        :param use_pun: 是否使用加标点符号的模型
        :param pun_model_dir: 给识别结果加标点符号的模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        model_path = os.path.join(model_dir, 'inference')
        model_info_path = os.path.join(model_dir, 'inference.json')
        with open(model_info_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
            print_arguments(configs=configs, title="模型参数配置")
        self.model_info = dict_to_object(configs)
        if self.model_info.model_name == "DeepSpeech2Model":
            assert decoder != "attention_rescoring", f'DeepSpeech2Model不支持使用{decoder}解码器！'
        self.decoder = decoder
        # 读取解码器配置文件
        if isinstance(decoder_configs, str) and os.path.exists(decoder_configs):
            with open(decoder_configs, 'r', encoding='utf-8') as f:
                decoder_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=decoder_configs, title='解码器参数配置')
        self.decoder_configs = decoder_configs if decoder_configs is not None else {}
        self.running = False
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self.pun_predictor = None
        vocab_model_dir = os.path.join(model_dir, 'vocab_model/')
        assert os.path.exists(vocab_model_dir), f'词表模型文件夹[{vocab_model_dir}]不存在，请检查该文件是否存在！'
        self._tokenizer = PPASRTokenizer(vocab_model_dir=vocab_model_dir)
        self._audio_featurizer = AudioFeaturizer(**self.model_info.preprocess_conf)
        # 流式解码参数
        self.remained_wav = None
        self.cached_feat = None
        self.last_chunk_text = ""
        self.last_chunk_time = 0
        self.reset_state_time = 30
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 加标点符号
        if use_pun:
            from ppasr.infer_utils.pun_predictor import PunctuationPredictor
            self.pun_predictor = PunctuationPredictor(model_dir=pun_model_dir, use_gpu=use_gpu)
        # 获取预测器
        self.predictor = InferencePredictor(model_name=self.model_info.model_name,
                                            streaming=self.model_info.streaming,
                                            model_path=model_path,
                                            use_gpu=self.use_gpu)
        # 加载流式VAD模型
        if self.model_info.streaming:
            self.streaming_vad = StreamingVAD(sample_rate=self.model_info.sample_rate,
                                              params=VADParams(stop_secs=0.2))
        # 预热
        warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(134240,))
        self.predict(audio_data=warmup_audio, is_itn=False)
        if self.model_info.streaming:
            self.predict_stream(audio_data=warmup_audio[:16000], is_itn=False)
        self.reset_stream()
        logger.info("预测器已准备完成！")

    # 解码模型输出结果
    def decode(self, encoder_outs, ctc_probs, ctc_lens, use_pun, is_itn):
        """解码模型输出结果

        :param encoder_outs: 编码器输出
        :param ctc_probs: 模型输出的CTC概率
        :param ctc_lens: 模型输出的CTC长度
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :return:
        """
        # 执行解码
        if self.decoder == "ctc_greedy_search":
            result = ctc_greedy_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens, blank_id=self._tokenizer.blank_id)
        elif self.decoder == "ctc_prefix_beam_search":
            decoder_args = self.decoder_configs.get('ctc_prefix_beam_search_args', {})
            result, _ = ctc_prefix_beam_search(ctc_probs=ctc_probs, ctc_lens=ctc_lens,
                                               blank_id=self._tokenizer.blank_id, **decoder_args)
        elif self.decoder == "attention_rescoring":
            decoder_args = self.decoder_configs.get('attention_rescoring_args', {})
            result = attention_rescoring(model=self.predictor.model,
                                         ctc_probs=ctc_probs,
                                         ctc_lens=ctc_lens,
                                         blank_id=self._tokenizer.blank_id,
                                         encoder_outs=encoder_outs,
                                         encoder_lens=ctc_lens,
                                         **decoder_args)
        else:
            raise ValueError(f"不支持该解码器：{self.decoder}")
        text = self._tokenizer.ids2text(result[0])

        # 加标点符号
        if use_pun and len(text) > 0:
            if self.pun_predictor is not None:
                text = self.pun_predictor(text)
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)
        return text

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
    def _infer(self,
               audio_data,
               use_pun=False,
               is_itn=False,
               sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 音频数据
        :type audio_data: np.ndarray
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 提取音频特征
        audio_feature = self._audio_featurizer.featurize(waveform=audio_data, sample_rate=sample_rate)
        audio_feature = audio_feature.unsqueeze(0)
        audio_len = np.array([audio_feature.shape[1]], dtype=np.int64)

        # 运行predictor
        encoder_outs, ctc_probs, ctc_lens = self.predictor.predict(audio_feature, audio_len)

        # 解码
        text = self.decode(encoder_outs=encoder_outs, ctc_probs=ctc_probs, ctc_lens=ctc_lens,
                           use_pun=use_pun, is_itn=is_itn)
        return text

    # 语音预测
    def predict(self,
                audio_data,
                use_pun=False,
                is_itn=False,
                sample_rate=16000,
                allow_use_vad=True):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param use_pun: 是否使用加标点符号的模型
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :param allow_use_vad: 当音频长度大于30秒，是否允许使用语音活动检测分割音频进行识别
        :return: 识别的文本结果和解码的得分数
        """
        if isinstance(audio_data, np.ndarray):
            assert isinstance(sample_rate, int), '当传入的是numpy数据时，需要指定采样率'
        # 加载音频文件，并进行预处理
        audio_segment = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        # 重采样，方便进行语音活动检测
        if audio_segment.sample_rate != self.model_info.sample_rate:
            audio_segment.resample(self.model_info.sample_rate)
        if audio_segment.duration <= 30 or not allow_use_vad:
            text = self._infer(audio_data=audio_segment.samples, use_pun=use_pun, is_itn=is_itn,
                               sample_rate=audio_segment.sample_rate)
            result = {'text': text,
                      'sentences': [{'text': text, 'start': 0, 'end': audio_segment.duration}]}
            return result
        elif allow_use_vad and audio_segment.duration > 30:
            last_audio_ndarray = None
            # 获取语音活动区域
            speech_timestamps = audio_segment.vad()
            texts, sentences = '', []
            for t in speech_timestamps:
                audio_ndarray = audio_segment.samples[t['start']: t['end']]
                # 如果语音片段小于0.5秒，则跳过推理，下次合并使用
                if (t['end'] - t['start']) * audio_segment.sample_rate < 0.5 and last_audio_ndarray is None:
                    continue
                if last_audio_ndarray is not None:
                    audio_ndarray = np.concatenate((last_audio_ndarray, audio_ndarray))
                    last_audio_ndarray = None
                # 执行识别
                text = self._infer(audio_data=audio_ndarray, use_pun=False, is_itn=is_itn,
                                   sample_rate=audio_segment.sample_rate)
                if text != '':
                    texts = texts + text if use_pun else texts + '，' + text
                sentences.append({'text': text,
                                  'start': round(t['start'] / audio_segment.sample_rate, 3),
                                  'end': round(t['end'] / audio_segment.sample_rate, 3)})
                logger.info(f'长语音识别片段结果：{text}')
            if texts[0] == '，': texts = texts[1:]
            # 加标点符号
            if use_pun and len(texts) > 0:
                if self.pun_predictor is not None:
                    texts = self.pun_predictor(texts)
                else:
                    logger.warning('标点符号模型没有初始化！')
            result = {'text': texts, 'sentences': sentences}
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
        assert self.model_info.streaming, f'不支持改该模型流式识别，当前模型：{self.model_info.model_name}'
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
        x_chunk = self._audio_featurizer.featurize(self.remained_wav, sample_rate=self.remained_wav.sample_rate)
        x_chunk = np.array(x_chunk).astype(np.float32)[np.newaxis, :]
        if self.cached_feat is None:
            self.cached_feat = x_chunk
        else:
            self.cached_feat = np.concatenate([self.cached_feat, x_chunk], axis=1)
        self.remained_wav._samples = self.remained_wav.samples[160 * x_chunk.shape[1]:]

        # 实时检测VAD
        state = VADState.SPEAKING
        if self.streaming_vad is not None:
            for ith_frame in range(0, len(audio_data.samples), self.streaming_vad.vad_frames):
                buffer = audio_data.samples[ith_frame:ith_frame + self.streaming_vad.vad_frames]
                state = self.streaming_vad(buffer)

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
            if self.model_info.model_name == 'DeepSpeech2Model':
                output_chunk_probs, output_lens = self.predictor.predict_chunk_deepspeech(x_chunk=x_chunk)
            elif 'ConformerModel' in self.model_info.model_name:
                num_decoding_left_chunks = -1
                required_cache_size = decoding_chunk_size * num_decoding_left_chunks
                output_chunk_probs = self.predictor.predict_chunk_conformer(x_chunk=x_chunk,
                                                                            required_cache_size=required_cache_size)
                output_lens = paddle.to_tensor([output_chunk_probs.size(1)], dtype=paddle.int32,
                                               place=output_chunk_probs.place)
            else:
                raise Exception(f'当前模型不支持该方法，当前模型为：{self.model_info.model_name}')
            # 执行解码
            chunk_result = ctc_greedy_search(ctc_probs=output_chunk_probs, ctc_lens=output_lens,
                                             blank_id=self._tokenizer.blank_id)[0]
            chunk_text = self._tokenizer.ids2text(chunk_result)
            self.last_chunk_text = self.last_chunk_text + chunk_text
        # 更新特征缓存
        self.cached_feat = self.cached_feat[:, end - cached_feature_num:, :]
        # 如果是静音并且推理音频时间足够长，重置流式识别状态，以免显存不足
        if state == VADState.QUIET and self.last_chunk_time > self.reset_state_time:
            logger.info('检测到静音，重置流式识别状态！')
            self.predictor.reset_stream()
            self.last_chunk_time = 0
        # 加标点符号
        if use_pun and is_end and len(self.last_chunk_text) > 0:
            if self.pun_predictor is not None:
                self.last_chunk_text = self.pun_predictor(self.last_chunk_text)
            else:
                logger.warning('标点符号模型没有初始化！')
        # 是否对文本进行逆标准化
        if is_itn and is_end and len(self.last_chunk_text) > 0:
            self.last_chunk_text = self.inverse_text_normalization(self.last_chunk_text)

        result = {'text': self.last_chunk_text}
        return result

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.predictor.reset_stream()
        self.last_chunk_time = 0
        self.remained_wav = None
        self.cached_feat = None
        self.last_chunk_text = ''

    # 对文本进行反标准化
    def inverse_text_normalization(self, text):
        if self.inv_normalizer is None:
            # 需要安装WeTextProcessing>=1.0.4.1
            from itn.chinese.inverse_normalizer import InverseNormalizer
            user_dir = os.path.expanduser('~')
            cache_dir = os.path.join(user_dir, '.cache/itn_v1.0.4.1')
            exists = os.path.exists(os.path.join(cache_dir, 'zh_itn_tagger.fst')) and \
                     os.path.exists(os.path.join(cache_dir, 'zh_itn_verbalizer.fst'))
            self.inv_normalizer = InverseNormalizer(cache_dir=cache_dir, enable_0_to_9=False,
                                                    overwrite_cache=not exists)
        result_text = self.inv_normalizer.normalize(text)
        return result_text
