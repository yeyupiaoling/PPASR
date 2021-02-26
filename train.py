import argparse
import functools
import os

import paddle
from paddle.io import DataLoader

from data.utility import add_arguments, print_arguments
from utils.data import PPASRDataset, collate_fn
from utils.model import PPASR

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size', int, 64, "Minibatch size.")
add_arg('num_workers', int, 0, "Minibatch size.")
add_arg('num_epoch', int, 200, "# of training epochs.")
add_arg('learning_rate', int, 1e-3, "# of training epochs.")
add_arg('train_manifest', str, "dataset/manifest.train", "# of training epochs.")
add_arg('dataset_vocab', str, "dataset/zh_vocab.json", "# of training epochs.")
add_arg('save_model', str, 'models', "Save model parameters and optimizer parameters path")
add_arg('pretrained_model', str, None,
        "If None, the training starts from scratch, otherwise, it resumes from the pre-trained model.")
args = parser.parse_args()


def train(args):
    # 获取训练数据
    train_dataset = PPASRDataset(args.train_manifest, args.dataset_vocab)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              num_workers=args.num_workers)
    train_loader_shuffle = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn,
                                      num_workers=args.num_workers,
                                      shuffle=True)
    model = PPASR(train_dataset.vocabulary)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.learning_rate)
    ctc_loss = paddle.nn.CTCLoss()
    # 加载预训练模型
    if args.pretrained_model is not None:
        model.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'model.pdparams')))
        optimizer.set_state_dict(paddle.load(os.path.join(args.pretrained_model, 'optimizer.pdopt')))
    # 开始训练
    for epoch in range(args.num_epoch):
        # 第一个epoch不打乱数据
        if epoch == 1:
            train_loader = train_loader_shuffle
        for batch_id, (inputs, labels, input_lens, label_lens) in enumerate(train_loader()):
            out = model(inputs)
            out = paddle.transpose(out, perm=[2, 0, 1])
            out_lens = paddle.to_tensor(input_lens / 2 + 1, dtype='int64')
            loss = ctc_loss(out, labels, out_lens, label_lens)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id % 100 == 0:
                print('epoch %d, batch %d, loss: %f' % (epoch, batch_id, loss))
        # 保存模型
        if not os.path.exists(os.path.join(args.save_model, 'epoch_%d' % epoch)):
            os.makedirs(os.path.join(args.save_model, 'epoch_%d' % epoch))
        paddle.save(model.state_dict(), os.path.join(args.save_model, 'epoch_%d' % epoch, 'model.pdparams'))
        paddle.save(optimizer.state_dict(), os.path.join(args.save_model, 'epoch_%d' % epoch, 'optimizer.pdopt'))


if __name__ == '__main__':
    print_arguments(args)
    train(args)
