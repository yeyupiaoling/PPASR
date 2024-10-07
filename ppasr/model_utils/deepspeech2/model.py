import importlib
from typing import Dict

import paddle
from paddle import nn

from ppasr.data_utils.normalizer import FeatureNormalizer
from ppasr.model_utils.deepspeech2.encoder import CRNNEncoder
from ppasr.model_utils.loss.ctc import CTCLoss
from ppasr.model_utils.utils.cmvn import GlobalCMVN
from ppasr.utils.utils import DictObject

__all__ = ["DeepSpeech2Model"]


class DeepSpeech2Model(nn.Layer):
    """The DeepSpeech2 network structure.

    :param input_size: feature size for audio.
    :type input_size: int
    :param vocab_size: Dictionary size for tokenized transcription.
    :type vocab_size: int
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 mean_istd_path: str,
                 streaming: bool = True,
                 encoder_conf: DictObject = None,
                 decoder_conf: DictObject = None):
        super().__init__()
        self.input_size = input_size
        self.streaming = streaming
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=mean_istd_path)
        global_cmvn = GlobalCMVN(paddle.to_tensor(feature_normalizer.mean, dtype=paddle.float32),
                                 paddle.to_tensor(feature_normalizer.istd, dtype=paddle.float32))
        # 创建编码器
        mod = importlib.import_module(__name__)
        self.encoder: CRNNEncoder = getattr(mod, encoder_conf.encoder_name)
        self.encoder = self.encoder(input_size=input_size,
                                    vocab_size=vocab_size,
                                    global_cmvn=global_cmvn,
                                    rnn_direction='forward' if streaming else 'bidirect',
                                    **encoder_conf.encoder_args if encoder_conf.encoder_args is not None else {})
        self.decoder = CTCLoss(odim=vocab_size,
                               encoder_output_size=self.encoder.output_size,
                               dopout_rate=0.1)

    def forward(self, speech, speech_lengths, text, text_lengths):
        """Compute Model loss

        Args:
            speech (Tensor): [B, T, D]
            speech_lengths (Tensor): [B]
            text (Tensor): [B, U]
            text_lengths (Tensor): [B]

        Returns:
            loss (Tensor): [1]
        """
        eouts, eouts_len, final_state_h_box, final_state_c_box = self.encoder(speech, speech_lengths, None, None)
        loss = self.decoder(eouts, eouts_len, text, text_lengths)
        return {'loss': loss}

    def get_encoder_out(self, speech, speech_lengths):
        encoder_outs, encoder_lens, _, _ = self.encoder(speech, speech_lengths, None, None)
        ctc_probs = self.decoder.softmax(encoder_outs)
        return encoder_outs, ctc_probs, encoder_lens

    def get_encoder_out_chunk(self, speech, speech_lengths, init_state_h_box=None, init_state_c_box=None):
        eouts, eouts_len, final_chunk_state_h_box, final_chunk_state_c_box = self.encoder(speech, speech_lengths,
                                                                                          init_state_h_box,
                                                                                          init_state_c_box)
        ctc_probs = self.decoder.softmax(eouts)
        return ctc_probs, eouts_len, final_chunk_state_h_box, final_chunk_state_c_box

    def export(self):
        if self.streaming:
            static_model = paddle.jit.to_static(
                self.get_encoder_out_chunk,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, None, self.input_size],
                                            dtype=paddle.float32),  # [B, chunk_size, feat_dim]
                    paddle.static.InputSpec(shape=[None], dtype=paddle.int64),  # audio_length, [B]
                    paddle.static.InputSpec(shape=[None, None, None], dtype=paddle.float32),
                    paddle.static.InputSpec(shape=[None, None, None], dtype=paddle.float32)
                ])
        else:
            static_model = paddle.jit.to_static(
                self.get_encoder_out,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, None, self.input_size], dtype=paddle.float32),  # [B, T, D]
                    paddle.static.InputSpec(shape=[None], dtype=paddle.int64),  # audio_length, [B]
                ])
        return static_model
