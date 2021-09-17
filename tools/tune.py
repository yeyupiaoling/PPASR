"""查找最优的集束搜索方法的alpha参数和beta参数"""
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import argparse
import functools
import paddle.fluid as fluid
from tqdm import tqdm
from decoders.beam_search_decoder import BeamSearchDecoder
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from utils.error_rate import char_errors, word_errors
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_batches',      int,    -1,    "用于评估的数据数量，当为-1时使用全部数据")
add_arg('batch_size',       int,    64,    "评估是每一批数据的大小")
add_arg('beam_size',        int,    10,    "定向搜索的大小，范围:[5, 500]")
add_arg('num_proc_bsearch', int,    8,     "定向搜索方法使用CPU数量")
add_arg('num_conv_layers',  int,    2,     "卷积层数量")
add_arg('num_rnn_layers',   int,    3,     "循环神经网络的数量")
add_arg('rnn_layer_size',   int,    1024,  "循环神经网络的大小")
add_arg('num_alphas',       int,    45,    "用于调优的alpha候选项")
add_arg('num_betas',        int,    8,     "用于调优的beta候选项")
add_arg('alpha_from',       float,  1.0,   "alpha调优开始大小")
add_arg('alpha_to',         float,  3.2,   "alpha调优结速大小")
add_arg('beta_from',        float,  0.1,   "beta调优开始大小")
add_arg('beta_to',          float,  0.45,  "beta调优结速大小")
add_arg('alpha',            float,  1.2,   "定向搜索的LM系数")
add_arg('beta',             float,  0.35,  "定向搜索的WC系数")
add_arg('cutoff_prob',      float,  1.0,   "剪枝的概率")
add_arg('cutoff_top_n',     int,    40,    "剪枝的最大值")
add_arg('use_gpu',          bool,   True,  "是否使用GPU训练")
add_arg('tune_manifest',    str,    './dataset/manifest.test',     "需要评估的测试数据列表")
add_arg('mean_std_path',    str,    './dataset/mean_std.npz',      "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',       str,    './dataset/zh_vocab.txt',      "数据集的词汇表文件路径")
add_arg('lang_model_path',  str,    './lm/zh_giga.no_cna_cmn.prune01244.klm',   "语言模型文件路径")
add_arg('model_path',       str,    './models/param/50.pdparams',               "训练保存的模型文件夹路径")
add_arg('error_rate_type',  str,    'cer',    "评估所使用的错误率方法，有字错率(cer)、词错率(wer)", choices=['wer', 'cer'])
args = parser.parse_args()


def tune():
    # 逐步调整alphas参数和betas参数
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    # 是否使用GPU
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()

    # 获取数据生成器
    data_generator = DataGenerator(vocab_filepath=args.vocab_path,
                                   mean_std_filepath=args.mean_std_path,
                                   keep_transcription_text=True,
                                   place=place,
                                   is_training=False)
    # 获取评估数据
    batch_reader = data_generator.batch_reader_creator(manifest_path=args.tune_manifest,
                                                       batch_size=args.batch_size,
                                                       shuffle_method=None)
    # 获取DeepSpeech2模型，并设置为预测
    ds2_model = DeepSpeech2Model(vocab_size=data_generator.vocab_size,
                                 num_conv_layers=args.num_conv_layers,
                                 num_rnn_layers=args.num_rnn_layers,
                                 rnn_layer_size=args.rnn_layer_size,
                                 use_gru=args.use_gru,
                                 place=place,
                                 pretrained_model=args.model_path,
                                 share_rnn_weights=args.share_rnn_weights,
                                 is_infer=True)

    # 初始化集束搜索方法
    beam_search_decoder = BeamSearchDecoder(args.alpha, args.beta, args.lang_model_path, data_generator.vocab_list)

    # 获取评估函数，有字错率和词错率
    errors_func = char_errors if args.error_rate_type == 'cer' else word_errors
    # 创建用于搜索的alphas参数和betas参数
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(alpha, beta) for alpha in cand_alphas for beta in cand_betas]

    err_sum = [0.0 for _ in range(len(params_grid))]
    err_ave = [0.0 for _ in range(len(params_grid))]
    num_ins, len_refs, cur_batch = 0, 0, 0
    # 多批增量调优参数
    ds2_model.logger.info("start tuning ...")
    for infer_data in batch_reader():
        if (args.num_batches >= 0) and (cur_batch >= args.num_batches):
            break
        # 执行预测
        probs_split = ds2_model.infer_batch_data(infer_data=infer_data)
        target_transcripts = infer_data[1]

        num_ins += len(target_transcripts)
        # 搜索alphas参数和betas参数
        for index, (alpha, beta) in enumerate(tqdm(params_grid)):
            result_transcripts = beam_search_decoder.decode_batch_beam_search(probs_split=probs_split,
                                                                              beam_alpha=alpha,
                                                                              beam_beta=beta,
                                                                              beam_size=args.beam_size,
                                                                              cutoff_prob=args.cutoff_prob,
                                                                              cutoff_top_n=args.cutoff_top_n,
                                                                              vocab_list=data_generator.vocab_list,
                                                                              num_processes=args.num_proc_bsearch)
            for target, result in zip(target_transcripts, result_transcripts):
                errors, len_ref = errors_func(target, result)
                err_sum[index] += errors
                if args.alpha_from == alpha and args.beta_from == beta:
                    len_refs += len_ref

            err_ave[index] = err_sum[index] / len_refs

        # 输出每一个batch的计算结果
        err_ave_min = min(err_ave)
        min_index = err_ave.index(err_ave_min)
        print("\nBatch %d [%d/?], current opt (alpha, beta) = (%s, %s), "
              " min [%s] = %f" % (cur_batch, num_ins, "%.3f" % params_grid[min_index][0],
                                  "%.3f" % params_grid[min_index][1], args.error_rate_type, err_ave_min))
        cur_batch += 1

    # 输出字错率和词错率以及(alpha, beta)
    print("\nFinal %s:\n" % args.error_rate_type)
    for index in range(len(params_grid)):
        print("(alpha, beta) = (%s, %s), [%s] = %f"
              % ("%.3f" % params_grid[index][0], "%.3f" % params_grid[index][1], args.error_rate_type, err_ave[index]))

    err_ave_min = min(err_ave)
    min_index = err_ave.index(err_ave_min)
    print("\n一共使用了 %d 批数据推理, 最优的参数为 (alpha, beta) = (%s, %s)"
          % (cur_batch, "%.3f" % params_grid[min_index][0], "%.3f" % params_grid[min_index][1]))

    ds2_model.logger.info("finish tuning")


def main():
    print_arguments(args)
    tune()


if __name__ == '__main__':
    main()
