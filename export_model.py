import argparse
import functools

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformer.yml',    '配置文件')
add_arg("use_gpu",          bool,  True,                       '是否使用GPU评估模型')
add_arg("save_quant",       bool,  False,                      '是否保存量化模型')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('resume_model',     str,   'models/conformer_streaming_fbank/best_model/', '准备导出的模型路径')
args = parser.parse_args()
print_arguments(args=args)


# 获取训练器
trainer = PPASRTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 导出预测模型
trainer.export(save_model_path=args.save_model,
               resume_model=args.resume_model,
               save_quant=args.save_quant)
