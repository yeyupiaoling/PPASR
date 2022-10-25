import os

import numpy as np
import paddle
import paddle.inference as paddle_infer

from ppasr.model_utils.conformer.model import ConformerModelOnline, ConformerModelOffline
from ppasr.model_utils.deepspeech2.model import DeepSpeech2ModelOnline, DeepSpeech2ModelOffline
from ppasr.utils.logger import setup_logger

logger = setup_logger(__name__)


class PythonPredictor:
    def __init__(self,
                 configs,
                 use_model,
                 input_dim,
                 vocab_size,
                 model_dir='models/deepspeech2_online_fbank/infer/',
                 use_gpu=True):
        """
        语音识别预测工具
        :param use_model: 使用模型的名称
        :param model_dir: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        self.running = False
        self.configs = configs
        self.use_gpu = use_gpu
        self.use_model = use_model
        # 流式解码参数
        self.output_state_h = None
        self.output_state_c = None
        # 创建 predictor
        if self.configs.use_model == 'conformer_online':
            self.predictor = ConformerModelOnline(configs=self.configs,
                                                  input_dim=input_dim,
                                                  vocab_size=vocab_size,
                                                  **self.configs.model_conf)
        elif self.configs.use_model == 'conformer_offline':
            self.predictor = ConformerModelOffline(configs=self.configs,
                                                   input_dim=input_dim,
                                                   vocab_size=vocab_size,
                                                   **self.configs.model_conf)
        elif self.configs.use_model == 'deepspeech2_online':
            self.predictor = DeepSpeech2ModelOnline(configs=self.configs,
                                                    input_dim=input_dim,
                                                    vocab_size=vocab_size)
        elif self.configs.use_model == 'deepspeech2_offline':
            self.predictor = DeepSpeech2ModelOffline(configs=self.configs,
                                                     input_dim=input_dim,
                                                     vocab_size=vocab_size)
        else:
            raise Exception('没有该模型：{}'.format(self.configs.use_model))
        if os.path.isdir(model_dir):
            model_dir = os.path.join(model_dir, 'model.pdparams')
        assert os.path.exists(model_dir), f"{model_dir} 模型不存在！"
        model_state_dict = paddle.load(model_dir)
        self.predictor.set_state_dict(model_state_dict)
        logger.info(f'成功加载模型：{model_dir}')
        self.predictor.eval()

    # 预测音频
    def predict(self, speech, speech_lengths):
        """
        预测函数，只预测完整的一句话。
        :param speech: 经过处理的音频数据
        :param speech_lengths: 音频长度
        :return: 识别的文本结果和解码的得分数
        """
        audio_data = paddle.to_tensor(speech, dtype=paddle.float32)
        audio_len = paddle.to_tensor(speech_lengths)
        if self.use_gpu:
            audio_data = audio_data.cuda()
            audio_len = audio_len.cuda()
        # 运行predictor
        output_data = self.predictor.get_encoder_out(audio_data, audio_len).numpy()
        return output_data

    def predict_chunk(self, x_chunk, x_chunk_lens):
        # 设置输入
        x_chunk = paddle.to_tensor(x_chunk, dtype=paddle.float32)
        x_chunk_lens = paddle.to_tensor(x_chunk_lens)

        if self.use_gpu:
            x_chunk = x_chunk.cuda()
            x_chunk_lens = x_chunk_lens.cuda()

        # 运行predictor
        output_chunk_probs, output_lens, self.output_state_h, self.output_state_c = \
            self.predictor.get_encoder_out_chunk(x_chunk, x_chunk_lens, self.output_state_h, self.output_state_c)

        output_chunk_probs = output_chunk_probs.numpy()
        output_lens = output_lens.numpy()
        return output_chunk_probs, output_lens

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.output_state_h = None
        self.output_state_c = None
