import argparse
import functools
import os
import re
from datetime import datetime

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from visualdl import LogWriter

from data.utility import add_arguments, print_arguments
from utils.data import PPASRDataset, collate_fn
from utils.decoder import GreedyDecoder
from utils.model import PPASR

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,  32,                       '训练的批量大小')
add_arg('num_workers',      int,  8,                        '读取数据的线程数量')
add_arg('num_epoch',        int,  200,                      '训练的轮数')
add_arg('learning_rate',    int,  1e-3,                     '初始学习率的大小')
add_arg('data_mean',        int,  -3.146301,                '数据集的均值')
add_arg('data_std',         int,  52.998405,                '数据集的标准值')
add_arg('min_duration',     int,  0,                        '过滤最短的音频长度')
add_arg('max_duration',     int,  20,                       '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('train_manifest',   str,  'dataset/manifest.train', '训练数据的数据列表路径')
add_arg('test_manifest',    str,  'dataset/manifest.test',  '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,  'dataset/zh_vocab.json',  '数据字典的路径')
add_arg('save_model',       str,  'models/',                '模型保存的路径')
add_arg('pretrained_model', str,  None,                     '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
def evaluate(model, test_loader, greedy_decoder):
    cer = []
    for batch_id, (inputs, labels, _, _) in enumerate(test_loader()):
        # 执行识别
        outs = model(inputs)
        outs = paddle.nn.functional.softmax(outs, 1)
        outs = paddle.transpose(outs, perm=[0, 2, 1])
        # 解码获取识别结果
        out_strings, out_offsets = greedy_decoder.decode(outs)
        labels = greedy_decoder.convert_to_strings(labels)
        for out_string, label in zip(*(out_strings, labels)):
            # 计算字错率
            c = greedy_decoder.cer(out_string[0], label[0]) / float(len(label[0]))
            cer.append(c)
    cer = float(np.mean(cer))
    return cer


# 保存模型
def save_model(args, epoch, model, optimizer):
    model_path = os.path.join(args.save_model, 'epoch_%d' % epoch)
    if epoch == args.num_epoch - 1:
        model_path = os.path.join(args.save_model, 'step_final')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))


def train(args):
    if dist.get_rank() == 0:
        # 日志记录器
        writer = LogWriter(logdir='log')
    # 设置支持多卡训练
    dist.init_parallel_env()
    # 获取训练数据
    train_dataset = PPASRDataset(args.train_manifest, args.dataset_vocab,
                                 mean=args.data_mean,
                                 std=args.data_std,
                                 min_duration=args.min_duration,
                                 max_duration=args.max_duration)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              num_workers=args.num_workers,
                              use_shared_memory=False)
    train_loader_shuffle = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      use_shared_memory=False)
    # 获取测试数据
    test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab, mean=args.data_mean, std=args.data_std)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=collate_fn,
                             num_workers=args.num_workers,
                             use_shared_memory=False)
    # 获取解码器，用于评估
    greedy_decoder = GreedyDecoder(train_dataset.vocabulary)
    # 获取模型，同时数据均值和标准值到模型中，方便以后推理使用
    model = PPASR(train_dataset.vocabulary, data_mean=paddle.to_tensor(args.data_mean), data_std=paddle.to_tensor(args.data_std))
    if dist.get_rank() == 0:
        print('input_size的第三个参数是变长的，这里为了能查看输出的大小变化，指定了一个值！')
        paddle.summary(model, input_size=(args.batch_size, 128, 500))
    # 设置支持多卡训练
    model = paddle.DataParallel(model)
    # 设置优化方法
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    # 分段学习率
    boundaries = [10, 20, 50, 100]
    lr = [0.1 ** l * args.learning_rate for l in range(len(boundaries) + 1)]
    # 获取预训练的epoch数
    last_epoch = int(re.findall(r'\d+', args.pretrained_model)[-1]) if args.pretrained_model is not None else -1
    scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=lr, last_epoch=last_epoch, verbose=True)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=scheduler, grad_clip=clip)
    # 获取损失函数
    ctc_loss = paddle.nn.CTCLoss()
    # 加载预训练模型
    if args.pretrained_model is not None:
        model.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'optimizer.pdopt')))
    train_step = 0
    test_step = 0
    # 开始训练
    for epoch in range(last_epoch, args.num_epoch):
        # 第一个epoch不打乱数据
        if epoch == 1:
            train_loader = train_loader_shuffle
        for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(train_loader()):
            out, out_lens = model(inputs, input_lens)
            out = paddle.transpose(out, perm=[2, 0, 1])
            # 计算损失
            loss = ctc_loss(out, labels, out_lens, label_lens)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and dist.get_rank() == 0:
                print('[%s] Train epoch %d, batch %d, loss: %f' % (datetime.now(), epoch, batch_id, loss))
                writer.add_scalar('Train loss', loss, train_step)
                train_step += 1
            # 固定步数也要保存一次模型
            if batch_id % 2000 == 0 and batch_id != 0 and dist.get_rank() == 0:
                # 保存模型
                save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
        # 多卡训练只使用一个进程执行评估和保存模型
        if dist.get_rank() == 0:
            # 执行评估
            model.eval()
            cer = evaluate(model, test_loader, greedy_decoder)
            print('[%s] Test epoch %d, cer: %f' % (datetime.now(), epoch, cer))
            writer.add_scalar('Test cer', cer, test_step)
            test_step += 1
            model.train()
            # 记录学习率
            writer.add_scalar('Learning rate', scheduler.last_lr, epoch)
            # 保存模型
            save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    dist.spawn(train, args=(args,))
