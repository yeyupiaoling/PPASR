# 训练模型

 - 训练流程，首先是准备数据集，具体看[数据准备](./dataset.md)部分，重点是执行`create_data.py`程序，执行完成之后检查是否在`dataset`目录下生成了`manifest.test`、`manifest.train`、`mean_istd.json`、`vocab_model/`这四个文件，并确定里面已经包含数据。然后才能往下执行开始训练。

 - 执行训练脚本，开始训练语音识别模型，详细参数请查看`configs`下的配置文件。每训练一轮和每10000个batch都会保存一次模型，模型保存在`models/<use_model>_<feature_method>/epoch_*/`目录下，默认会使用数据增强训练，如何不想使用数据增强，只需要将参数`data_augment_configs`设置为`None`即可。关于数据增强，请查看[数据增强](./augment.md)部分。如果没有关闭测试，在每一轮训练结果之后，都会执行一次测试计算模型在测试集的准确率，注意为了加快训练速度，训练只能用贪心解码。如果模型文件夹下包含`last_model`文件夹，在训练的时候会自动加载里面的模型，这是为了方便中断训练的之后继续训练，无需手动指定，如果手动指定了`resume_model`参数，则以`resume_model`指定的路径优先加载。如果不是原来的数据集或者模型结构，需要删除`last_model`这个文件夹。
```shell
# 单机单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 单机多卡训练
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch --devices=0,1 train.py
```

多机多卡的启动方式：
```shell
# 第一台服务器（主）
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch --devices=0,1 --nnodes 2 train.py
```

输出如下：
```
Copy the following command to other nodes to run.
--------------------------------------------------------------------------------
python -m paddle.distributed.launch --master 192.168.10.7:38945 --devices=0,1 --nnodes 2 train.py
--------------------------------------------------------------------------------
```

其他机器执行，上面输出的命令：
```shell
# 第二台服务器
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch --master 192.168.10.7:38945 --devices=0,1 --nnodes 2 train.py
```

训练输出结果如下：
```
2024-09-21 16:22:01.956 | INFO     | ppasr.utils.utils:print_arguments:13 - ----------- 额外配置参数 -----------
2024-09-21 16:22:01.956 | INFO     | ppasr.utils.utils:print_arguments:15 - configs: configs/conformer.yml
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - data_augment_configs: configs/augmentation.yml
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - local_rank: 0
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - metrics_type: cer
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - pretrained_model: None
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - resume_model: None
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - save_model_path: models/
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:15 - use_gpu: True
2024-09-21 16:22:01.957 | INFO     | ppasr.utils.utils:print_arguments:16 - ------------------------------------------------
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 配置文件参数 -----------
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:23 - dataset_conf:
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:26 - 	batch_sampler:
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		batch_size: 6
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		drop_last: True
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		shuffle: True
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		sortagrad: True
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:26 - 	dataLoader:
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_workers: 4
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:26 - 	dataset:
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		manifest_type: txt
2024-09-21 16:22:01.989 | INFO     | ppasr.utils.utils:print_arguments:28 - 		max_duration: 20
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		min_duration: 0.5
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		sample_rate: 16000
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		target_dB: -20
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		use_dB_normalization: True
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:30 - 	mean_istd_path: dataset/mean_istd.json
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:30 - 	test_manifest: dataset/manifest.test
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:30 - 	train_manifest: dataset/manifest.train
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:23 - decoder_conf:
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:26 - 	decoder_args:
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_heads: 4
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		dropout_rate: 0.1
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		linear_units: 1024
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_blocks: 3
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		positional_dropout_rate: 0.1
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		r_num_blocks: 3
2024-09-21 16:22:01.990 | INFO     | ppasr.utils.utils:print_arguments:28 - 		self_attention_dropout_rate: 0.1
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		src_attention_dropout_rate: 0.1
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:30 - 	decoder_name: BiTransformerDecoder
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:23 - encoder_conf:
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:26 - 	encoder_args:
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		activation_type: swish
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_dropout_rate: 0.1
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_heads: 4
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		cnn_module_kernel: 15
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		dropout_rate: 0.1
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		input_layer: conv2d
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		linear_units: 2048
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		normalize_before: True
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_blocks: 12
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		output_size: 256
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		pos_enc_layer_type: rel_pos
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		positional_dropout_rate: 0.1
2024-09-21 16:22:01.992 | INFO     | ppasr.utils.utils:print_arguments:28 - 		use_cnn_module: True
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:30 - 	encoder_name: ConformerEncoder
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:23 - model_conf:
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:30 - 	model: ConformerModel
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:26 - 	model_args:
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		ctc_weight: 0.3
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		length_normalized_loss: False
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		lsm_weight: 0.1
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		reverse_weight: 0.3
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		streaming: True
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:23 - optimizer_conf:
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:30 - 	optimizer: Adam
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:26 - 	optimizer_args:
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		lr: 0.001
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:28 - 		weight_decay: 1e-06
2024-09-21 16:22:01.993 | INFO     | ppasr.utils.utils:print_arguments:30 - 	scheduler: WarmupLR
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:26 - 	scheduler_args:
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:28 - 		min_lr: 1e-05
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:28 - 		warmup_steps: 25000
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:23 - preprocess_conf:
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	feature_method: fbank
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:26 - 	method_args:
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_mel_bins: 80
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:23 - tokenizer_conf:
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	build_vocab_size: None
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	model_type: char
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	non_linguistic_symbols: []
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	remove_non_linguistic_symbols: False
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	vocab_model_dir: dataset/vocab_model/
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:23 - train_conf:
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	accum_grad: 4
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	enable_amp: False
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	grad_clip: 5.0
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	log_interval: 100
2024-09-21 16:22:01.994 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_epoch: 200
2024-09-21 16:22:01.996 | INFO     | ppasr.utils.utils:print_arguments:30 - 	use_compile: False
2024-09-21 16:22:01.996 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 数据增强配置 -----------
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:23 - noise:
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_snr_dB: 50
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	min_snr_dB: 10
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	noise_dir: dataset/noise
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.5
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:23 - resample:
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	new_sample_rate: [8000, 16000, 24000]
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.0
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:23 - reverb:
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.2
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	reverb_dir: dataset/reverb
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:23 - shift:
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_shift_ms: 5
2024-09-21 16:22:01.998 | INFO     | ppasr.utils.utils:print_arguments:30 - 	min_shift_ms: -5
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.5
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:23 - spec_aug:
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	freq_mask_ratio: 0.15
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_time_warp: 5
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	n_freq_masks: 2
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	n_time_masks: 2
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.5
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	time_mask_ratio: 0.05
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:23 - spec_sub_aug:
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_time: 30
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	num_time_sub: 3
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.5
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:23 - speed:
2024-09-21 16:22:01.999 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.5
2024-09-21 16:22:02.002 | INFO     | ppasr.utils.utils:print_arguments:23 - volume:
2024-09-21 16:22:02.002 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_gain_dBFS: 15
2024-09-21 16:22:02.002 | INFO     | ppasr.utils.utils:print_arguments:30 - 	min_gain_dBFS: -15
2024-09-21 16:22:02.002 | INFO     | ppasr.utils.utils:print_arguments:30 - 	prob: 0.5
2024-09-21 16:22:02.002 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 16:22:02.002 | WARNING  | ppasr.trainer:__init__:101 - Windows系统不支持多线程读取数据，已自动关闭！
2024-09-21 16:22:02.142 | INFO     | yeaudio.augmentation:__init__:135 - 噪声增强的噪声音频文件数量: 0
2024-09-21 16:22:02.142 | INFO     | yeaudio.augmentation:__init__:170 - 混响增强音频文件数量: 0
2024-09-21 16:22:02.454 | INFO     | ppasr.model_utils:build_model:23 - 成功创建模型：ConformerModel，参数为：{'streaming': True, 'ctc_weight': 0.3, 'lsm_weight': 0.1, 'reverse_weight': 0.3, 'length_normalized_loss': False, 'input_size': 80, 'vocab_size': 8000, 'mean_istd_path': 'dataset/mean_istd.json', 'sos_id': 2, 'eos_id': 3}
2024-09-21 16:22:04.397 | INFO     | ppasr.optimizer:build_optimizer:16 - 成功创建优化方法：Adam，参数为：{'lr': 0.001, 'weight_decay': 1e-06}
2024-09-21 16:22:04.398 | INFO     | ppasr.optimizer:build_lr_scheduler:31 - 成功创建学习率衰减：WarmupLR，参数为：{'warmup_steps': 25000, 'min_lr': 1e-05}
2024-09-21 16:22:04.398 | INFO     | ppasr.trainer:train:475 - 训练数据：13382，词汇表大小：8000
2024-09-21 16:22:05.211 | INFO     | ppasr.trainer:__train_epoch:354 - Train epoch: [1/200], batch: [0/2230], loss: 32.43893, learning_rate: 0.00000008, reader_cost: 0.0436, batch_cost: 0.7689, ips: 7.3853 speech/sec, eta: 4 days, 4:39:00
```


 - 在训练过程中，程序会使用VisualDL记录训练结果，可以通过在根目录执行以下命令启动VisualDL。
```shell
visualdl --logdir=log --host=0.0.0.0
```

 - 然后再浏览器上访问`http://localhost:8040`可以查看结果显示，如下。

<div align="center">
  <img src="./images/visualdl.jpg" alt="VisualDL" width="800">
</div>

