import argparse
import functools
import io
import os
import re
import shutil
import time
from datetime import datetime, timedelta

import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader
from visualdl import LogWriter

from data_utils.reader import PPASRDataset
from data_utils.collate_fn import collate_fn
from data_utils.sampler import SortagradBatchSampler, SortagradDistributedBatchSampler
from decoders.ctc_greedy_decoder import greedy_decoder_batch
from model_utils.deepspeech2.model import DeepSpeech2Model
from model_utils.deepspeech2_light.model import DeepSpeech2LightModel
from utils.metrics import cer
from utils.utils import add_arguments, print_arguments
from utils.utils import labels_to_string, fuzzy_delete

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,   16,                         '训练的批量大小')
add_arg('num_workers',      int,   8,                          '读取数据的线程数量')
add_arg('num_epoch',        int,   50,                         '训练的轮数')
add_arg('learning_rate',    int,   5e-4,                       '初始学习率的大小')
add_arg('min_duration',     int,   0,                          '过滤最短的音频长度')
add_arg('max_duration',     int,   20,                         '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('use_model',        str,   'deepspeech2',              '所使用的模型')
add_arg('train_manifest',   str,   'dataset/manifest.train',   '训练数据的数据列表路径')
add_arg('test_manifest',    str,   'dataset/manifest.test',    '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,   'dataset/vocabulary.txt',   '数据字典的路径')
add_arg('mean_std_path',    str,   'dataset/mean_std.npz',     '数据集的均值和标准值的npy文件路径')
add_arg('augment_conf_path',str,   'conf/augmentation.json',   '数据增强的配置文件，为json格式')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                      '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    None,                      '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
@paddle.no_grad()
def evaluate(model, test_loader, vocabulary, ctc_loss):
    c = []
    l = []
    for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(test_loader()):
        # 执行识别
        outs, out_lens = model(inputs, input_lens)
        out = paddle.transpose(outs, perm=[1, 0, 2])

        # 计算损失
        loss = ctc_loss(out, labels, out_lens, label_lens, norm_by_times=True)
        loss = loss.mean().numpy()[0]
        l.append(loss)
        outs = paddle.nn.functional.softmax(outs, 2)
        # 解码获取识别结果
        out_strings = greedy_decoder_batch(outs.numpy(), vocabulary)
        labels_str = labels_to_string(labels.numpy(), vocabulary)
        for out_string, label in zip(*(out_strings, labels_str)):
            # 计算字错率
            c.append(cer(out_string, label) / float(len(label)))
        if batch_id % 100 == 0:
            print('[{}] Test batch: [{}/{}], loss: {:.5f}, cer: {:.5f}'.format(datetime.now(), batch_id, len(test_loader),
                                                                               loss, float(sum(c) / len(c))))
    c = float(sum(c) / len(c))
    l = float(sum(l) / len(l))
    return c, l


# 保存模型
def save_model(args, epoch, model, optimizer):
    model_path = os.path.join(args.save_model, args.use_model, 'epoch_%d' % epoch)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    paddle.save(model.state_dict(), os.path.join(model_path, 'model.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    # 删除旧的模型
    old_model_path = os.path.join(args.save_model, args.use_model, 'epoch_%d' % (epoch - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)


def train(args):
    # 获取有多少张显卡训练
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if local_rank == 0:
        fuzzy_delete('log', 'vdlrecords')
        # 日志记录器
        writer = LogWriter(logdir='log')
    if nranks > 1:
        # 初始化Fleet环境
        fleet.init(is_collective=True)

    # 获取训练数据
    augmentation_config = io.open(args.augment_conf_path, mode='r', encoding='utf8').read() if args.augment_conf_path is not None else '{}'
    train_dataset = PPASRDataset(args.train_manifest, args.dataset_vocab,
                                 mean_std_filepath=args.mean_std_path,
                                 min_duration=args.min_duration,
                                 max_duration=args.max_duration,
                                 augmentation_config=augmentation_config)
    # 设置支持多卡训练
    if nranks > 1:
        train_batch_sampler = SortagradDistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_batch_sampler = SortagradBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset,
                              collate_fn=collate_fn,
                              batch_sampler=train_batch_sampler,
                              num_workers=args.num_workers)
    # 获取测试数据
    test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab, mean_std_filepath=args.mean_std_path)
    test_batch_sampler = SortagradBatchSampler(test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset,
                             collate_fn=collate_fn,
                             batch_sampler=test_batch_sampler,
                             num_workers=args.num_workers)

    # 获取模型
    if args.use_model == 'deepspeech2':
        model = DeepSpeech2Model(feat_size=train_dataset.feature_dim, vocab_size=train_dataset.vocab_size)
    elif args.use_model == 'deepspeech2_light':
        model = DeepSpeech2LightModel(vocab_size=train_dataset.vocab_size)
    else:
        raise Exception('没有该模型：%s' % args.use_model)

    if local_rank == 0:
        print(f"{model}")
        print('[{}] input_size的第三个参数是变长的，这里为了能查看输出的大小变化，指定了一个值！'.format(datetime.now()))
        paddle.summary(model, input_size=[(None, train_dataset.feature_dim, 970), (None,)], dtypes=[paddle.float32, paddle.int64])

    # 设置优化方法
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=400.0)
    # 获取预训练的epoch数
    last_epoch = int(re.findall(r'\d+', args.resume_model)[-1]) if args.resume_model is not None else 0
    scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=args.learning_rate, gamma=0.83, last_epoch=last_epoch - 1)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=scheduler,
                                      weight_decay=paddle.regularizer.L2Decay(1e-5),
                                      grad_clip=grad_clip)

    # 设置支持多卡训练
    if nranks > 1:
        optimizer = fleet.distributed_optimizer(optimizer)
        model = fleet.distributed_model(model)

    print('[{}] 训练数据：{}'.format(datetime.now(), len(train_dataset)))

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

    # 加载恢复模型
    if args.resume_model is not None:
        assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(args.resume_model, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.resume_model, 'optimizer.pdopt')))
        print('[{}] 成功恢复模型参数和优化方法参数'.format(datetime.now()))

    # 获取损失函数
    ctc_loss = paddle.nn.CTCLoss(reduction='none')

    train_step = 0
    test_step = 0
    sum_batch = len(train_loader) * args.num_epoch
    # 开始训练
    for epoch in range(last_epoch, args.num_epoch):
        epoch += 1
        start_epoch = time.time()
        for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(train_loader()):
            start = time.time()
            out, out_lens = model(inputs, input_lens)
            out = paddle.transpose(out, perm=[1, 0, 2])

            # 计算损失
            loss = ctc_loss(out, labels, out_lens, label_lens, norm_by_times=True)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0 and local_rank == 0:
                eta_sec = ((time.time() - start) * 1000) * (sum_batch - (epoch - 1) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print('[{}] Train epoch: [{}/{}], batch: [{}/{}], loss: {:.5f}, learning rate: {:>.8f}, eta: {}'.format(
                    datetime.now(), epoch, args.num_epoch, batch_id, len(train_loader), loss.numpy()[0], scheduler.get_lr(), eta_str))
                writer.add_scalar('Train loss', loss, train_step)
                train_step += 1

            # 固定步数也要保存一次模型
            if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)

        # 多卡训练只使用一个进程执行评估和保存模型
        if local_rank == 0:
            # 执行评估
            model.eval()
            print('\n', '='*70)
            c, l = evaluate(model, test_loader, test_dataset.vocab_list, ctc_loss)
            print('[{}] Test epoch: {}, time/epoch: {}, loss: {:.5f}, cer: {:.5f}'.format(
                datetime.now(), epoch, str(timedelta(seconds=(time.time() - start_epoch))), l, c))
            print('='*70, '\n')
            writer.add_scalar('Test cer', c, test_step)
            test_step += 1
            model.train()

            # 记录学习率
            writer.add_scalar('Learning rate', scheduler.last_lr, epoch)

            # 保存模型
            save_model(args=args, epoch=epoch, model=model, optimizer=optimizer)
        scheduler.step()


if __name__ == '__main__':
    print_arguments(args)
    train(args)
