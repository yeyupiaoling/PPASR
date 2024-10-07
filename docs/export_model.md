# 导出模型

训练保存的或者下载作者提供的模型都是模型参数，我们要将它导出为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测，详细参数请查看该程序。
```shell
python export_model.py --resume_model=models/ConformerModel_fbank/best_model/
```

输出结果：
```
2024-09-21 15:33:53.584 | INFO     | ppasr.utils.utils:print_arguments:13 - ----------- 额外配置参数 -----------
2024-09-21 15:33:53.585 | INFO     | ppasr.utils.utils:print_arguments:15 - configs: configs/conformer.yml
2024-09-21 15:33:53.585 | INFO     | ppasr.utils.utils:print_arguments:15 - resume_model: models/ConformerModel_fbank/best_model/
2024-09-21 15:33:53.585 | INFO     | ppasr.utils.utils:print_arguments:15 - save_model: models/
2024-09-21 15:33:53.585 | INFO     | ppasr.utils.utils:print_arguments:15 - save_quant: False
2024-09-21 15:33:53.585 | INFO     | ppasr.utils.utils:print_arguments:15 - use_gpu: True
2024-09-21 15:33:53.585 | INFO     | ppasr.utils.utils:print_arguments:16 - ------------------------------------------------
2024-09-21 15:33:53.607 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 配置文件参数 -----------
2024-09-21 15:33:53.607 | INFO     | ppasr.utils.utils:print_arguments:23 - dataset_conf:
2024-09-21 15:33:53.607 | INFO     | ppasr.utils.utils:print_arguments:26 - 	batch_sampler:
2024-09-21 15:33:53.607 | INFO     | ppasr.utils.utils:print_arguments:28 - 		batch_size: 6
2024-09-21 15:33:53.607 | INFO     | ppasr.utils.utils:print_arguments:28 - 		drop_last: True
2024-09-21 15:33:53.607 | INFO     | ppasr.utils.utils:print_arguments:28 - 		shuffle: True
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		sortagrad: True
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:26 - 	dataLoader:
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_workers: 4
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:26 - 	dataset:
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		manifest_type: txt
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		max_duration: 20
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		min_duration: 0.5
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		sample_rate: 16000
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		target_dB: -20
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		use_dB_normalization: True
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:30 - 	mean_istd_path: dataset/mean_istd.json
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:30 - 	test_manifest: dataset/manifest.test
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:30 - 	train_manifest: dataset/manifest.train
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:23 - decoder_conf:
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:26 - 	decoder_args:
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_heads: 4
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		dropout_rate: 0.1
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		linear_units: 1024
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_blocks: 3
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		positional_dropout_rate: 0.1
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		r_num_blocks: 3
2024-09-21 15:33:53.608 | INFO     | ppasr.utils.utils:print_arguments:28 - 		self_attention_dropout_rate: 0.1
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		src_attention_dropout_rate: 0.1
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:30 - 	decoder_name: BiTransformerDecoder
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:23 - encoder_conf:
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:26 - 	encoder_args:
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		activation_type: swish
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_dropout_rate: 0.1
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_heads: 4
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		cnn_module_kernel: 15
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		dropout_rate: 0.1
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		input_layer: conv2d
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		linear_units: 2048
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		normalize_before: True
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_blocks: 12
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		output_size: 256
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		pos_enc_layer_type: rel_pos
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		positional_dropout_rate: 0.1
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:28 - 		use_cnn_module: True
2024-09-21 15:33:53.610 | INFO     | ppasr.utils.utils:print_arguments:30 - 	encoder_name: ConformerEncoder
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:23 - model_conf:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	model: ConformerModel
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:26 - 	model_args:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		ctc_weight: 0.3
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		length_normalized_loss: False
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		lsm_weight: 0.1
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		reverse_weight: 0.3
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		streaming: True
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:23 - optimizer_conf:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	optimizer: Adam
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:26 - 	optimizer_args:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		lr: 0.001
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		weight_decay: 1e-06
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	scheduler: WarmupLR
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:26 - 	scheduler_args:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		min_lr: 1e-05
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		warmup_steps: 25000
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:23 - preprocess_conf:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	feature_method: fbank
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:26 - 	method_args:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_mel_bins: 80
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:23 - tokenizer_conf:
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	build_vocab_size: None
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	model_type: char
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	non_linguistic_symbols: []
2024-09-21 15:33:53.611 | INFO     | ppasr.utils.utils:print_arguments:30 - 	remove_non_linguistic_symbols: False
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:30 - 	vocab_model_dir: dataset/vocab_model/
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:23 - train_conf:
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:30 - 	accum_grad: 4
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:30 - 	enable_amp: False
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:30 - 	grad_clip: 5.0
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:30 - 	log_interval: 100
2024-09-21 15:33:53.612 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_epoch: 200
2024-09-21 15:33:53.613 | INFO     | ppasr.utils.utils:print_arguments:30 - 	use_compile: False
2024-09-21 15:33:53.613 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 15:33:53.979 | INFO     | ppasr.model_utils:build_model:23 - 成功创建模型：ConformerModel，参数为：{'streaming': True, 'ctc_weight': 0.3, 'lsm_weight': 0.1, 'reverse_weight': 0.3, 'length_normalized_loss': False, 'input_size': 80, 'vocab_size': 8000, 'mean_istd_path': 'dataset/mean_istd.json', 'sos_id': 2, 'eos_id': 3}
2024-09-21 15:33:54.401 | INFO     | ppasr.trainer:export:615 - 成功恢复模型参数和优化方法参数：models/ConformerModel_fbank/best_model/model.pth
2024-09-21 15:33:55.666 | INFO     | ppasr.trainer:export:627 - 预测模型已保存：models/ConformerModel_fbank/inference_model\inference.pth
```