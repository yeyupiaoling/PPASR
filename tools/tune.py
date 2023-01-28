"""查找最优的集束搜索方法的alpha参数和beta参数"""

import argparse
import functools

import numpy as np
import yaml

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformer.yml',  "配置文件")
add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
add_arg('resume_model',     str,   'models/conformer_streaming_fbank/best_model/',  "模型的路径")
add_arg('num_alphas',       int,    30,    "用于调优的alpha候选项")
add_arg('num_betas',        int,    20,    "用于调优的beta候选项")
add_arg('alpha_from',       float,  1.0,   "alpha调优开始大小")
add_arg('alpha_to',         float,  3.2,   "alpha调优结速大小")
add_arg('beta_from',        float,  0.1,   "beta调优开始大小")
add_arg('beta_to',          float,  4.5,   "beta调优结速大小")
args = parser.parse_args()


def tune():
    # 逐步调整alphas参数和betas参数
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    # 读取配置文件
    with open(args.configs, 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    print_arguments(args, configs)

    # 创建用于搜索的alphas参数和betas参数
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(round(alpha, 2), round(beta, 2)) for alpha in cand_alphas for beta in cand_betas]

    print('开始使用识别结果解码...')
    print('解码alpha和beta的排列：%s' % params_grid)
    # 搜索alphas参数和betas参数
    best_alpha, best_beta, best_result = 0, 0, 1
    for i, (alpha, beta) in enumerate(params_grid):
        # 获取训练器
        configs['decoder'] = 'ctc_beam_search'
        configs['ctc_beam_search_decoder_conf']['alpha'] = alpha
        configs['ctc_beam_search_decoder_conf']['beta'] = beta
        trainer = PPASRTrainer(configs=configs, use_gpu=args.use_gpu)
        _, error_result = trainer.evaluate(resume_model=args.resume_model.format(configs['use_model'],
                                                                                 configs['preprocess_conf'][ 'feature_method']))
        if error_result < best_result:
            best_alpha = alpha
            best_beta = beta
            best_result = error_result
        print('当alpha为：%f, beta为：%f，%s：%f' % (alpha, beta, configs['metrics_type'], error_result))
        print('【目前最优】当alpha为：%f, beta为：%f，%s最低，为：%f' % (best_alpha, best_beta, configs['metrics_type'], best_result))
    print('【最后结果】当alpha为：%f, beta为：%f，%s最低，为：%f' % (best_alpha, best_beta, configs['metrics_type'], best_result))


if __name__ == '__main__':
    tune()
