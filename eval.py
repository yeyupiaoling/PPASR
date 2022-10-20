import argparse
import functools
import time

import yaml

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformer_offline_zh.yml',     "配置文件")
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('resume_model',     str,   'models/{}_{}/best_model/',  "模型的路径")
args = parser.parse_args()


# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)

# 获取训练器
trainer = PPASRTrainer(configs=configs, use_gpu=args.use_gpu)

# 开始评估
start = time.time()
loss, error_result = trainer.evaluate(resume_model=args.resume_model.format(configs['use_model'],
                                                                            configs['preprocess_conf']['feature_method']))
end = time.time()
print('评估消耗时间：{}s，{}：{:.5f}'.format(int(end - start), configs['metrics_type'], error_result))
