import importlib

from loguru import logger

from ppasr.model_utils.conformer.model import ConformerModel
from ppasr.model_utils.deepspeech2.model import DeepSpeech2Model

__all__ = ['build_model']


def build_model(input_size, vocab_size, mean_istd_path, eos_id, encoder_conf, decoder_conf, model_conf):
    use_model = model_conf.get('model', 'ConformerModel')
    model_args = model_conf.get('model_args', {})
    model_args.input_size = input_size
    model_args.vocab_size = vocab_size
    model_args.mean_istd_path = mean_istd_path
    model_args.vocab_size = vocab_size
    if use_model != 'DeepSpeech2Model':
        model_args.eos_id = eos_id
    mod = importlib.import_module(__name__)
    model = getattr(mod, use_model)(encoder_conf=encoder_conf, decoder_conf=decoder_conf, **model_args)
    logger.info(f'成功创建模型：{use_model}，参数为：{model_args}')
    return model
