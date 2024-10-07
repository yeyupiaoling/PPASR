# 评估

执行下面这个脚本对模型进行评估，通过字符错误率来评价模型的性能，详细参数请查看该程序。
```shell
python eval.py --resume_model=models/ConformerModel_fbank/best_model/
```

输出结果：
```
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:13 - ----------- 额外配置参数 -----------
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:15 - configs: configs/conformer.yml
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:15 - decoder: ctc_greedy_search
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:15 - decoder_configs: configs/decoder.yml
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:15 - metrics_type: cer
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:15 - resume_model: models/ConformerModel_fbank/best_model/
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:15 - use_gpu: True
2024-09-21 15:10:05.601 | INFO     | ppasr.utils.utils:print_arguments:16 - ------------------------------------------------
2024-09-21 15:10:05.629 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 配置文件参数 -----------
2024-09-21 15:10:05.629 | INFO     | ppasr.utils.utils:print_arguments:23 - dataset_conf:
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:26 - 	batch_sampler:
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		batch_size: 6
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		drop_last: True
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		shuffle: True
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		sortagrad: True
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:26 - 	dataLoader:
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_workers: 4
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:26 - 	dataset:
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		manifest_type: txt
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		max_duration: 20
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		min_duration: 0.5
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		sample_rate: 16000
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		target_dB: -20
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		use_dB_normalization: True
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:30 - 	mean_istd_path: dataset/mean_istd.json
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:30 - 	test_manifest: dataset/manifest.test
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:30 - 	train_manifest: dataset/manifest.train
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:23 - decoder_conf:
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:26 - 	decoder_args:
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_heads: 4
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		dropout_rate: 0.1
2024-09-21 15:10:05.630 | INFO     | ppasr.utils.utils:print_arguments:28 - 		linear_units: 1024
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_blocks: 3
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:28 - 		positional_dropout_rate: 0.1
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:28 - 		r_num_blocks: 3
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:28 - 		self_attention_dropout_rate: 0.1
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:28 - 		src_attention_dropout_rate: 0.1
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:30 - 	decoder_name: BiTransformerDecoder
2024-09-21 15:10:05.631 | INFO     | ppasr.utils.utils:print_arguments:23 - encoder_conf:
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:26 - 	encoder_args:
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		activation_type: swish
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_dropout_rate: 0.1
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		attention_heads: 4
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		cnn_module_kernel: 15
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		dropout_rate: 0.1
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		input_layer: conv2d
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		linear_units: 2048
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		normalize_before: True
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_blocks: 12
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		output_size: 256
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		pos_enc_layer_type: rel_pos
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		positional_dropout_rate: 0.1
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:28 - 		use_cnn_module: True
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:30 - 	encoder_name: ConformerEncoder
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:23 - model_conf:
2024-09-21 15:10:05.632 | INFO     | ppasr.utils.utils:print_arguments:30 - 	model: ConformerModel
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:26 - 	model_args:
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		ctc_weight: 0.3
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		length_normalized_loss: False
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		lsm_weight: 0.1
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		reverse_weight: 0.3
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		streaming: True
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:23 - optimizer_conf:
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:30 - 	optimizer: Adam
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:26 - 	optimizer_args:
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		lr: 0.001
2024-09-21 15:10:05.633 | INFO     | ppasr.utils.utils:print_arguments:28 - 		weight_decay: 1e-06
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:30 - 	scheduler: WarmupLR
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:26 - 	scheduler_args:
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:28 - 		min_lr: 1e-05
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:28 - 		warmup_steps: 25000
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:23 - preprocess_conf:
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:30 - 	feature_method: fbank
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:26 - 	method_args:
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_mel_bins: 80
2024-09-21 15:10:05.634 | INFO     | ppasr.utils.utils:print_arguments:23 - tokenizer_conf:
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	build_vocab_size: None
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	model_type: char
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	non_linguistic_symbols: []
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	remove_non_linguistic_symbols: False
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	vocab_model_dir: dataset/vocab_model/
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:23 - train_conf:
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	accum_grad: 4
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	enable_amp: False
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	grad_clip: 5.0
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	log_interval: 100
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	max_epoch: 200
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:30 - 	use_compile: False
2024-09-21 15:10:05.635 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 解码器参数配置 -----------
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:23 - attention_rescoring_args:
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:30 - 	beam_size: 5
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:30 - 	ctc_weight: 0.3
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:30 - 	reverse_weight: 1.0
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:23 - ctc_prefix_beam_search_args:
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:30 - 	beam_size: 5
2024-09-21 15:10:05.636 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 15:10:05.636 | WARNING  | ppasr.trainer:__init__:101 - Windows系统不支持多线程读取数据，已自动关闭！
2024-09-21 15:10:06.001 | INFO     | ppasr.model_utils:build_model:23 - 成功创建模型：ConformerModel，参数为：{'streaming': True, 'ctc_weight': 0.3, 'lsm_weight': 0.1, 'reverse_weight': 0.3, 'length_normalized_loss': False, 'input_size': 80, 'vocab_size': 8000, 'mean_istd_path': 'dataset/mean_istd.json', 'sos_id': 2, 'eos_id': 3}
2024-09-21 15:10:06.442 | INFO     | ppasr.trainer:evaluate:543 - 成功加载模型：models/ConformerModel_fbank/best_model/model.pth
2024-09-21 15:10:06.885 | INFO     | ppasr.trainer:evaluate:575 - 实际标签为：四百二十三米跨径的主桥合龙误差不足两毫米比小米粒还小
2024-09-21 15:10:06.885 | INFO     | ppasr.trainer:evaluate:576 - 预测结果为：四百二十三米跨径的主桥合龙误差不足两毫米比小米粒还小
2024-09-21 15:10:06.885 | INFO     | ppasr.trainer:evaluate:577 - 这条数据的cer：0.0，当前cer：0.0
2024-09-21 15:10:06.885 | INFO     | ppasr.trainer:evaluate:579 - ----------------------------------------------------------------------
···············
2024-09-21 15:10:07.689 | INFO     | ppasr.trainer:evaluate:579 - ----------------------------------------------------------------------
2024-09-21 15:10:07.689 | INFO     | ppasr.trainer:evaluate:575 - 实际标签为：电炉炼钢价格成本低于转炉炼钢而且合理经济规模小吨钢投资较低是当今世界钢铁工业发展潮流
2024-09-21 15:10:07.689 | INFO     | ppasr.trainer:evaluate:576 - 预测结果为：电炉炼钢价格成本低于转炉炼钢而且合理经济规模小吨钢投资较低是当今世界钢铁工业发展潮流
2024-09-21 15:10:07.689 | INFO     | ppasr.trainer:evaluate:577 - 这条数据的cer：0.0，当前cer：0.013426
2024-09-21 15:10:07.689 | INFO     | ppasr.trainer:evaluate:579 - ----------------------------------------------------------------------
100%|██████████| 5/5 [00:01<00:00,  4.02it/s]

```
