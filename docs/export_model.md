# 导出模型

训练保存的或者下载作者提供的模型都是模型参数，我们要将它导出为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测，详细参数请查看该程序。
```shell
python export_model.py --resume_model=models/conformer_online_fbank/best_model/
```

输出结果：
```
[2022-10-28 22:42:54.069936 INFO   ] utils:print_arguments:19 - ----------- 额外配置参数 -----------
[2022-10-28 22:47:05.984016 INFO   ] utils:print_arguments:21 - configs: configs/conformer_online_zh.yml
[2022-10-28 22:47:05.984016 INFO   ] utils:print_arguments:21 - resume_model: models/conformer_online_fbank/best_model/
[2022-10-28 22:47:05.984016 INFO   ] utils:print_arguments:21 - save_model: models/
[2022-10-28 22:47:05.984016 INFO   ] utils:print_arguments:21 - use_gpu: True
[2022-10-28 22:42:54.069936 INFO   ] utils:print_arguments:22 - ------------------------------------------------
[2022-10-28 22:42:54.069936 INFO   ] utils:print_arguments:23 - ----------- 配置文件参数 -----------
[2022-10-28 22:42:54.069936 INFO   ] utils:print_arguments:26 - ctc_beam_search_decoder_conf:
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	alpha: 2.2
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	beam_size: 300
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	beta: 4.3
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	cutoff_prob: 0.99
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	cutoff_top_n: 40
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	language_model_path: lm/zh_giga.no_cna_cmn.prune01244.klm
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	num_processes: 10
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:26 - dataset_conf:
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	batch_size: 32
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	dataset_vocab: dataset/vocabulary.txt
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	manifest_type: txt
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	max_duration: 20
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	mean_istd_path: dataset/mean_istd.json
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	min_duration: 0.5
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	noise_manifest_path: dataset/manifest.noise
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	num_workers: 4
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	test_manifest: dataset/manifest.test
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:28 - 	train_manifest: dataset/manifest.train
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:30 - decoder: ctc_beam_search
[2022-10-28 22:42:54.070936 INFO   ] utils:print_arguments:26 - decoder_conf:
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	attention_heads: 4
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	dropout_rate: 0.1
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	linear_units: 2048
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	num_blocks: 6
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	positional_dropout_rate: 0.1
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	self_attention_dropout_rate: 0.0
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	src_attention_dropout_rate: 0.0
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:26 - encoder_conf:
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	activation_type: swish
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	attention_dropout_rate: 0.0
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	attention_heads: 4
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	cnn_module_kernel: 15
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	dropout_rate: 0.1
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	input_layer: conv2d
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	linear_units: 2048
[2022-10-28 22:42:54.079936 INFO   ] utils:print_arguments:28 - 	normalize_before: True
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	num_blocks: 12
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	output_size: 256
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	pos_enc_layer_type: rel_pos
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	positional_dropout_rate: 0.1
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	use_cnn_module: True
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:30 - metrics_type: cer
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:26 - model_conf:
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	ctc_weight: 0.3
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	length_normalized_loss: False
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	lsm_weight: 0.1
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:28 - 	reverse_weight: 0.0
[2022-10-28 22:42:54.080939 INFO   ] utils:print_arguments:26 - optimizer_conf:
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	learning_rate: 0.001
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	warmup_steps: 25000
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	weight_decay: 1e-6
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:26 - preprocess_conf:
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	feature_method: fbank
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	n_mels: 80
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	n_mfcc: 40
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	sample_rate: 16000
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	target_dB: -20
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:28 - 	use_dB_normalization: True
[2022-10-28 22:42:54.081940 INFO   ] utils:print_arguments:26 - train_conf:
[2022-10-28 22:42:54.082940 INFO   ] utils:print_arguments:28 - 	accum_grad: 4
[2022-10-28 22:42:54.082940 INFO   ] utils:print_arguments:28 - 	grad_clip: 5.0
[2022-10-28 22:42:54.082940 INFO   ] utils:print_arguments:28 - 	log_interval: 100
[2022-10-28 22:42:54.082940 INFO   ] utils:print_arguments:28 - 	max_epoch: 100
[2022-10-28 22:42:54.082940 INFO   ] utils:print_arguments:30 - use_model: conformer_online
[2022-10-28 22:42:54.082940 INFO   ] utils:print_arguments:31 - ------------------------------------------------
[2021-09-18 10:23:47.022243] 成功恢复模型参数和优化方法参数：models/conformer_online_fbank/best_model/model.pdparams

预测模型已保存：models/conformer_online_fbank/infer
```