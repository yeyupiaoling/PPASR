import paddle
from paddle import nn

from ppasr.data_utils.normalizer import FeatureNormalizer
from ppasr.model_utils.deepspeech2.encoder import CRNNEncoder
from ppasr.model_utils.loss.ctc import CTCLoss
from ppasr.model_utils.utils.cmvn import GlobalCMVN


class DeepSpeech2Model(nn.Layer):
    """The DeepSpeech2 network structure.

    :param input_dim: feature size for audio.
    :type input_dim: int
    :param vocab_size: Dictionary size for tokenized transcription.
    :type vocab_size: int
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """

    def __init__(self,
                 configs,
                 input_dim: int,
                 vocab_size: int,
                 rnn_direction='forward'):
        super().__init__()
        feature_normalizer = FeatureNormalizer(mean_istd_filepath=configs.dataset_conf.mean_istd_path)
        global_cmvn = GlobalCMVN(paddle.to_tensor(feature_normalizer.mean, dtype=paddle.float32),
                                 paddle.to_tensor(feature_normalizer.istd, dtype=paddle.float32))
        self.encoder = CRNNEncoder(input_dim=input_dim,
                                   vocab_size=vocab_size,
                                   global_cmvn=global_cmvn,
                                   rnn_direction=rnn_direction,
                                   **configs.encoder_conf)
        self.decoder = CTCLoss(vocab_size, self.encoder.output_size, **configs.decoder_conf)

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
        eouts, _, _, _ = self.encoder(speech, speech_lengths, None, None)
        ctc_probs = self.decoder.softmax(eouts)
        return ctc_probs

    def get_encoder_out_chunk(self, speech, speech_lengths, init_state_h_box=None, init_state_c_box=None):
        eouts, eouts_len, final_chunk_state_h_box, final_chunk_state_c_box = self.encoder(speech, speech_lengths,
                                                                                          init_state_h_box,
                                                                                          init_state_c_box)
        ctc_probs = self.decoder.softmax(eouts)
        return ctc_probs, eouts_len, final_chunk_state_h_box, final_chunk_state_c_box

    def export(self):
        if self.encoder.rnn_direction == "forward":
            static_model = paddle.jit.to_static(
                self.get_encoder_out_chunk,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, None, self.encoder.input_dim],
                                            dtype=paddle.float32),  # [B, chunk_size, feat_dim]
                    paddle.static.InputSpec(shape=[None], dtype=paddle.int64),  # audio_length, [B]
                    paddle.static.InputSpec(shape=[None, None, None], dtype=paddle.float32),
                    paddle.static.InputSpec(shape=[None, None, None], dtype=paddle.float32)
                ])
        elif self.encoder.rnn_direction == "bidirect":
            static_model = paddle.jit.to_static(
                self.get_encoder_out,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, None, self.encoder.input_dim], dtype=paddle.float32),  # [B, T, D]
                    paddle.static.InputSpec(shape=[None], dtype=paddle.int64),  # audio_length, [B]
                ])
        else:
            raise Exception(f"模型错误，没有：{self.encoder.rnn_direction}")
        return static_model


def DeepSpeech2ModelOnline(configs,
                           input_dim: int,
                           vocab_size: int):
    model = DeepSpeech2Model(configs=configs,
                             input_dim=input_dim,
                             vocab_size=vocab_size,
                             rnn_direction='forward')
    return model


def DeepSpeech2ModelOffline(configs,
                            input_dim: int,
                            vocab_size: int):
    model = DeepSpeech2Model(configs=configs,
                             input_dim=input_dim,
                             vocab_size=vocab_size,
                             rnn_direction='bidirect')
    return model
