import json
import os
import shutil

import paddle
from loguru import logger
from paddle import __version__


def load_pretrained(model, pretrained_model):
    """加载预训练模型

    :param model: 使用的模型
    :type model: paddle.nn.Module
    :param pretrained_model: 预训练模型路径
    :type pretrained_model: str
    """
    # 加载预训练模型
    if pretrained_model is None: return model
    if os.path.isdir(pretrained_model):
        pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
    model_state_dict = paddle.load(pretrained_model)
    model_dict = model.state_dict()
    # 过滤不存在的参数
    for name, weight in model_dict.items():
        if name in model_state_dict.keys():
            if list(weight.shape) != list(model_state_dict[name].shape):
                logger.warning(f'{name} not used, shape {list(model_state_dict[name].shape)} '
                               f'unmatched with {list(weight.shape)} in model.')
                model_state_dict.pop(name, None)
    # 加载权重
    missing_keys, unexpected_keys = model.set_state_dict(model_state_dict)
    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}. '
                       .format(', '.join('"{}"'.format(k) for k in missing_keys)))
    logger.info('成功加载预训练模型：{}'.format(pretrained_model))
    return model


def load_checkpoint(configs, model, optimizer, amp_scaler, scheduler,
                    step_epoch, save_model_path, resume_model):
    """加载模型

    :param configs: 配置信息
    :type configs: dict | Any
    :param model: 使用的模型
    :type model: paddle.nn.Module
    :param optimizer: 使用的优化方法
    :type optimizer: paddle.optim.Optimizer
    :param amp_scaler: 使用的自动混合精度
    :type amp_scaler: paddle.GradScaler
    :param scheduler: 使用的学习率调整策略
    :type scheduler: paddle.optim.lr_scheduler.LRScheduler
    :param step_epoch: 每个epoch的step数量
    :type step_epoch: int
    :param save_model_path: 模型保存路径
    :type save_model_path: str
    :param resume_model: 恢复训练的模型路径
    :type resume_model: str
    """
    last_epoch1 = 0
    error_rate1 = 1.0

    def load_model(model_path):
        assert os.path.exists(os.path.join(model_path, 'model.pdparams')), "模型参数文件不存在！"
        assert os.path.exists(os.path.join(model_path, 'optimizer.pdopt')), "优化方法参数文件不存在！"
        state_dict = paddle.load(os.path.join(model_path, 'model.pdparams'))
        model.set_state_dict(state_dict)
        optimizer.set_state_dict(paddle.load(os.path.join(model_path, 'optimizer.pdopt')))
        # 自动混合精度参数
        if amp_scaler is not None and os.path.exists(os.path.join(model_path, 'scaler.pdparams')):
            amp_scaler.set_state_dict(paddle.load(os.path.join(model_path, 'scaler.pdparams')))
        with open(os.path.join(model_path, 'model.state'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            error_rate = 1.0
            last_epoch = json_data['last_epoch']
            if 'cer' in json_data.keys():
                error_rate = abs(json_data['cer'])
            if 'wer' in json_data.keys():
                error_rate = abs(json_data['wer'])
            if 'mer' in json_data.keys():
                error_rate = abs(json_data['mer'])
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(model_path))
        optimizer.step()
        [scheduler.step() for _ in range(last_epoch * step_epoch)]
        return last_epoch, error_rate

    # 获取最后一个保存的模型
    save_feature_method = configs.preprocess_conf.feature_method
    last_model_dir = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'last_model')
    if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pdparams'))
                                    and os.path.exists(os.path.join(last_model_dir, 'optimizer.pdopt'))):
        if resume_model is not None:
            last_epoch1, error_rate1 = load_model(resume_model)
        else:
            try:
                # 自动获取最新保存的模型
                last_epoch1, error_rate1 = load_model(last_model_dir)
            except Exception as e:
                logger.warning(f'尝试自动恢复最新模型失败，错误信息：{e}')
    return model, optimizer, amp_scaler, scheduler, last_epoch1, error_rate1


# 保存模型
def save_checkpoint(configs, model, optimizer, amp_scaler, save_model_path, epoch_id,
                    error_rate=1.0, metrics_type=None, best_model=False):
    """保存模型

    :param configs: 配置信息
    :type configs: dict | Any
    :param model: 使用的模型
    :type model: paddle.nn.Module
    :param optimizer: 使用的优化方法
    :type optimizer: paddle.optim.Optimizer
    :param amp_scaler: 使用的自动混合精度
    :type amp_scaler: paddle.GradScaler
    :param save_model_path: 模型保存路径
    :type save_model_path: str
    :param epoch_id: 当前epoch
    :type epoch_id: int
    :param error_rate: 当前的错误率
    :type error_rate: float
    :param metrics_type: 当前使用的评估指标
    :type metrics_type: str
    :param best_model: 是否为最佳模型
    :type best_model: bool
    """
    state_dict = model.state_dict()
    # 保存模型的路径
    save_feature_method = configs.preprocess_conf.feature_method
    if best_model:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'best_model')
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}', 'epoch_{}'.format(epoch_id))
    os.makedirs(model_path, exist_ok=True)
    # 保存模型参数
    paddle.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pdopt'))
    paddle.save(state_dict, os.path.join(model_path, 'model.pdparams'))
    # 自动混合精度参数
    if amp_scaler is not None:
        paddle.save(amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pdparams'))
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        data = {"last_epoch": epoch_id, "version": __version__, "model": configs.model_conf.model,
                "feature_method": save_feature_method}
        if metrics_type is not None:
            data[metrics_type] = error_rate
        f.write(json.dumps(data, indent=4, ensure_ascii=False))
    if not best_model:
        last_model_path = os.path.join(save_model_path,
                                       f'{configs.model_conf.model}_{save_feature_method}', 'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)
        # 删除旧的模型
        old_model_path = os.path.join(save_model_path,
                                      f'{configs.model_conf.model}_{save_feature_method}',
                                      'epoch_{}'.format(epoch_id - 3))
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)
    logger.info('已保存模型：{}'.format(model_path))
