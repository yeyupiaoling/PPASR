import argparse
import functools
import os
import re
import shutil
import time
from datetime import datetime

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from tqdm import tqdm
from visualdl import LogWriter

from data.utility import add_arguments, print_arguments
from data_utils.reader import PPASRDataset, collate_fn
from decoders.ctc_greedy_decoder import greedy_decoder_batch
from model_utils.deepspeech2 import DeepSpeech2Model
from utils.metrics import cer
from utils.utils import labels_to_string
from paddle.static import InputSpec

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpus',             str,   '0',                        '训练使用的GPU序号，使用英文逗号,隔开，如：0,1')
add_arg('batch_size',       int,   16,                         '训练的批量大小')
add_arg('num_workers',      int,   8,                          '读取数据的线程数量')
add_arg('num_epoch',        int,   50,                         '训练的轮数')
add_arg('learning_rate',    float, 1e-3,                       '初始学习率的大小')
add_arg('num_conv_layers',  int,   2,                          '卷积层数量')
add_arg('num_rnn_layers',   int,   3,                          '循环神经网络的数量')
add_arg('rnn_layer_size',   int,   1024,                       '循环神经网络的大小')
add_arg('min_duration',     int,   0,                          '过滤最短的音频长度')
add_arg('max_duration',     int,   27,                         '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('train_manifest',   str,   'dataset/manifest.train',   '训练数据的数据列表路径')
add_arg('test_manifest',    str,   'dataset/manifest.test',    '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,   'dataset/vocabulary.json',  '数据字典的路径')
add_arg('mean_std_path',    str,   'dataset/mean_std.npz',     '数据集的均值和标准值的npy文件路径')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('resume',           str,    None,                      '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                      '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def evaluate(model, test_loader, vocabulary):
    c = []
    for inputs, labels, input_lens, _ in tqdm(test_loader()):
        # 执行识别
        outs, _ = model(inputs, input_lens)
        outs = paddle.nn.functional.softmax(outs, 2)
        # 解码获取识别结果
        out_strings = greedy_decoder_batch(outs.numpy(), vocabulary)
        labels_str = labels_to_string(labels.numpy(), vocabulary)
        for out_string, label in zip(*(out_strings, labels_str)):
            # 计算字错率
            c.append(cer(out_string, label) / float(len(label)))
    c = float(sum(c) / len(c))
    return c


# 保存模型
def save_model(args, epoch, model, optimizer, feature_dim):
    model_path = os.path.join(args.save_model, 'epoch_%d' % epoch)
    if epoch == args.num_epoch - 1:
        model_path = os.path.join(args.save_model, 'step_final')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    # 删除旧的模型
    old_model_path = os.path.join(args.save_model, 'epoch_%d' % (epoch - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)
    # 保存预测模型
    infer_model_path = os.path.join(args.save_model, 'infer')
    if not os.path.exists(infer_model_path):
        os.makedirs(infer_model_path)
    paddle.jit.save(layer=model,
                    path=os.path.join(infer_model_path, 'model'),
                    input_spec=[InputSpec(shape=(-1, feature_dim, -1), dtype=paddle.float32),
                                InputSpec(shape=(-1, ), dtype=paddle.int64)])


def train(args):
    if dist.get_rank() == 0:
        shutil.rmtree('log', ignore_errors=True)
        # 日志记录器
        writer = LogWriter(logdir='log')

    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        dist.init_parallel_env()

    # 获取训练数据
    train_dataset = PPASRDataset(args.train_manifest, args.dataset_vocab,
                                 mean_std_filepath=args.mean_std_path,
                                 min_duration=args.min_duration,
                                 max_duration=args.max_duration)
    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        train_batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=collate_fn,
                              batch_sampler=train_batch_sampler,
                              num_workers=args.num_workers)
    # 获取测试数据
    test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab, mean_std_filepath=args.mean_std_path)
    test_batch_sampler = paddle.io.BatchSampler(test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=collate_fn,
                             batch_sampler=test_batch_sampler,
                             num_workers=args.num_workers)

    # 获取模型
    model = DeepSpeech2Model(feat_size=train_dataset.feature_dim,
                             dict_size=len(train_dataset.vocabulary),
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_size=args.rnn_layer_size)
    if dist.get_rank() == 0:
        print(f"{model}")
        print('[{}] input_size的第三个参数是变长的，这里为了能查看输出的大小变化，指定了一个值！'.format(datetime.now()))
        paddle.summary(model, input_size=[(None, train_dataset.feature_dim, 970), (None,)], dtypes=['float32', 'int64'])

    # 设置支持多卡训练
    if len(args.gpus.split(',')) > 1:
        model = paddle.DataParallel(model)

    # 设置优化方法
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=400.0)
    # 获取预训练的epoch数
    last_epoch = int(re.findall(r'\d+', args.resume)[-1]) if args.resume is not None else 0
    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.learning_rate, gamma=0.83, last_epoch=last_epoch - 1, verbose=True)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=scheduler,
                                      weight_decay=paddle.regularizer.L2Decay(0.0001),
                                      grad_clip=clip)

    # 获取损失函数
    ctc_loss = paddle.nn.CTCLoss(reduction='mean')

    # 加载预训练模型
    if args.pretrained_model is not None:
        model_dict = model.state_dict()
        model_state_dict = paddle.load(os.path.join(args.pretrained_model, 'model.pdparams'))
        # 特征层
        for name, weight in model_dict.items():
            if name in model_state_dict.keys():
                if weight.shape != list(model_state_dict[name].shape):
                    print('{} not used, shape {} unmatched with {} in model.'.
                            format(name, list(model_state_dict[name].shape), weight.shape))
                    model_state_dict.pop(name, None)
            else:
                print('Lack weight: {}'.format(name))
        model.set_dict(model_state_dict)
        print('[{}] 成功加载预训练模型'.format(datetime.now()))

    # 加载预训练模型
    if args.resume is not None:
        model.set_state_dict(paddle.load(os.path.join(args.resume, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.resume, 'optimizer.pdopt')))
        print('[{}] 成功恢复模型参数和优化方法参数'.format(datetime.now()))

    train_step = 0
    test_step = 0
    # 开始训练
    for epoch in range(last_epoch, args.num_epoch):
        for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(train_loader()):
            start = time.time()
            out, out_lens = model(inputs, input_lens)
            out = paddle.transpose(out, perm=[1, 0, 2])

            # 计算损失
            loss = ctc_loss(out, labels, out_lens, label_lens, norm_by_times=True)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and dist.get_rank() == 0:
                print('[{}] Train epoch: {}, batch: {}/{}, loss: {:.5f}, learning rate: {}, train time: {:.3f}s'.format(
                    datetime.now(), epoch, batch_id, len(train_loader), loss.numpy()[0], scheduler.get_lr(), (time.time() - start)))
                writer.add_scalar('Train loss', loss, train_step)
                train_step += 1

            # 固定步数也要保存一次模型
            if batch_id % 2000 == 0 and batch_id != 0 and dist.get_rank() == 0:
                # 保存模型
                save_model(args=args, epoch=epoch, model=model, optimizer=optimizer, feature_dim=train_dataset.feature_dim)

        # 多卡训练只使用一个进程执行评估和保存模型
        if dist.get_rank() == 0:
            # 执行评估
            model.eval()
            c = evaluate(model, test_loader, test_dataset.vocabulary)
            print('\n', '='*70)
            print('[{}] Test epoch: {}, cer: {}'.format(datetime.now(), epoch, c))
            print('='*70)
            writer.add_scalar('Test cer', c, test_step)
            test_step += 1
            model.train()

            # 记录学习率
            writer.add_scalar('Learning rate', scheduler.last_lr, epoch)

            # 保存模型
            save_model(args=args, epoch=epoch, model=model, optimizer=optimizer, feature_dim=train_dataset.feature_dim)
        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    if len(args.gpus.split(',')) > 1:
        dist.spawn(train, args=(args,), gpus=args.gpus, nprocs=len(args.gpus.split(',')))
    else:
        train(args)
