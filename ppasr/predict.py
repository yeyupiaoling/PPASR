import os
import sys

from LAC import LAC
import cn2an
import numpy as np
import paddle.inference as paddle_infer

from ppasr.data_utils.audio_process import AudioProcess
from ppasr.decoders.ctc_greedy_decoder import greedy_decoder


class Predictor:
    def __init__(self, model_dir, vocab_path, decoder='ctc_greedy', alpha=1.2, beta=0.35, lang_model_path=None,
                 beam_size=10, cutoff_prob=1.0, cutoff_top_n=40, use_gpu=True, gpu_mem=500, num_threads=10):
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        self.lang_model_path = lang_model_path
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.use_gpu = use_gpu
        self.lac = None
        self.audio_process = AudioProcess(vocab_filepath=vocab_path)
        # 集束搜索方法的处理
        if decoder == "ctc_beam_search":
            try:
                from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
                self.beam_search_decoder = BeamSearchDecoder(alpha, beta, lang_model_path, self.audio_process.vocab_list)
            except ModuleNotFoundError:
                raise Exception('缺少ctc_decoders库，请在decoders目录中安装ctc_decoders库，如果是Windows系统，请使用ctc_greed。')

        # 创建 config
        model_path = os.path.join(model_dir, 'model.pdmodel')
        params_path = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise Exception("模型文件不存在，请检查%s和%s是否存在！" % (model_path, params_path))
        self.config = paddle_infer.Config(model_path, params_path)
        self.config.enable_use_gpu(1000, 0)
        self.config.enable_memory_optim()

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

        # 获取输入层
        self.audio_data_handle = self.predictor.get_input_handle('audio')
        self.audio_len_handle = self.predictor.get_input_handle('audio_len')

        # 获取输出的名称
        self.output_names = self.predictor.get_output_names()

        # 预热
        warmup_audio_path = 'dataset/test.wav'
        if os.path.exists(warmup_audio_path):
            self.predict(warmup_audio_path, to_an=True)
        else:
            print('预热文件不存在，忽略预热！', file=sys.stderr)

    # 预测图片
    def predict(self, audio_path, to_an=False):
        # 加载音频文件，并进行预处理
        audio_feature = self.audio_process.process_utterance(audio_path)
        audio_data = np.array(audio_feature).astype('float32')[np.newaxis, :]
        audio_len = np.array([audio_data.shape[2]]).astype('int64')

        # 设置输入
        self.audio_data_handle.reshape([audio_data.shape[0], audio_data.shape[1], audio_data.shape[2]])
        self.audio_len_handle.reshape([audio_data.shape[0]])
        self.audio_data_handle.copy_from_cpu(audio_data)
        self.audio_len_handle.copy_from_cpu(audio_len)

        # 运行predictor
        self.predictor.run()

        # 获取输出
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_data = output_handle.copy_to_cpu()[0]

        # 执行解码
        if self.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            result = self.beam_search_decoder.decode_beam_search(probs_split=output_data,
                                                                 beam_alpha=self.alpha,
                                                                 beam_beta=self.beta,
                                                                 beam_size=self.beam_size,
                                                                 cutoff_prob=self.cutoff_prob,
                                                                 cutoff_top_n=self.cutoff_top_n,
                                                                 vocab_list=self.audio_process.vocab_list)
        else:
            # 贪心解码策略
            result = greedy_decoder(probs_seq=output_data, vocabulary=self.audio_process.vocab_list)

        score, text = result[0], result[1]
        # 是否转为阿拉伯数字
        if to_an:
            text = self.cn2an(text)
        return score, text

    # 是否转为阿拉伯数字
    def cn2an(self, text):
        # 获取分词模型
        if self.lac is None:
            self.lac = LAC(mode='lac', use_cuda=self.use_gpu)
        lac_result = self.lac.run(text)
        result_text = ''
        for t, r in zip(lac_result[0], lac_result[1]):
            if r == 'm' or r == 'TIME':
                t = cn2an.transform(t, "cn2an")
            result_text += t
        return result_text
