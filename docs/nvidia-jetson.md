# Nvidia Jetson部署

1. 这对Nvidia Jetson设备，如Nano、Nx、AGX等设备，打开下面链接下载所需的PaddlePaddle的Inference预测库。
```shell
https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#python
```

2. 安装scikit-learn依赖库。
```shell
git clone git://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
pip3 install cython
git checkout 0.24.2
pip3 install --verbose --no-build-isolation --editable .
```

3. 安装其他依赖库。
```shell
pip3 install -r requirements.txt
```

3. 执行预测，直接使用根目录下的预测代码。
```shell
python infer_path.py --wav_path=./dataset/test.wav
```

以Nvidia AGX为例，输出结果如下：
```
[2022-10-28 22:48:02.777229 INFO   ] utils:print_arguments:19 - ----------- 额外配置参数 -----------
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - configs: configs/conformer_online_zh.yml
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - is_itn: False
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - is_long_audio: False
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - model_path: models/{}_{}/infer/
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - pun_model_dir: models/pun_models/
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - real_time_demo: False
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - use_gpu: True
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - use_pun: False
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:21 - wav_path: dataset/test.wav
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:22 - ------------------------------------------------
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:23 - ----------- 配置文件参数 -----------
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:26 - ctc_beam_search_decoder_conf:
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	alpha: 2.2
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	beam_size: 300
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	beta: 4.3
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	cutoff_prob: 0.99
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	cutoff_top_n: 40
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	language_model_path: lm/zh_giga.no_cna_cmn.prune01244.klm
[2022-10-28 22:48:02.778229 INFO   ] utils:print_arguments:28 - 	num_processes: 10
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:26 - dataset_conf:
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	batch_size: 32
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	dataset_vocab: dataset/vocabulary.txt
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	manifest_type: txt
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	max_duration: 20
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	mean_istd_path: dataset/mean_istd.json
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	min_duration: 0.5
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	noise_manifest_path: dataset/manifest.noise
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	num_workers: 4
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	test_manifest: dataset/manifest.test
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	train_manifest: dataset/manifest.train
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:30 - decoder: ctc_beam_search
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:26 - decoder_conf:
[2022-10-28 22:48:02.779232 INFO   ] utils:print_arguments:28 - 	attention_heads: 4
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	dropout_rate: 0.1
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	linear_units: 2048
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	num_blocks: 6
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	positional_dropout_rate: 0.1
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	self_attention_dropout_rate: 0.0
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	src_attention_dropout_rate: 0.0
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:26 - encoder_conf:
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	activation_type: swish
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	attention_dropout_rate: 0.0
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	attention_heads: 4
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	cnn_module_kernel: 15
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	dropout_rate: 0.1
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	input_layer: conv2d
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	linear_units: 2048
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	normalize_before: True
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	num_blocks: 12
[2022-10-28 22:48:02.780200 INFO   ] utils:print_arguments:28 - 	output_size: 256
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	pos_enc_layer_type: rel_pos
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	positional_dropout_rate: 0.1
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	use_cnn_module: True
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:30 - metrics_type: cer
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:26 - model_conf:
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	ctc_weight: 0.3
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	length_normalized_loss: False
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	lsm_weight: 0.1
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	reverse_weight: 0.0
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:26 - optimizer_conf:
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	learning_rate: 0.001
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	warmup_steps: 25000
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	weight_decay: 1e-6
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:26 - preprocess_conf:
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	feature_method: fbank
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	n_mels: 80
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	n_mfcc: 40
[2022-10-28 22:48:02.781200 INFO   ] utils:print_arguments:28 - 	sample_rate: 16000
[2022-10-28 22:48:02.782199 INFO   ] utils:print_arguments:28 - 	target_dB: -20
[2022-10-28 22:48:02.782199 INFO   ] utils:print_arguments:28 - 	use_dB_normalization: True
[2022-10-28 22:48:02.782199 INFO   ] utils:print_arguments:26 - train_conf:
[2022-10-28 22:48:02.782199 INFO   ] utils:print_arguments:28 - 	accum_grad: 4
[2022-10-28 22:48:02.795200 INFO   ] utils:print_arguments:28 - 	grad_clip: 5.0
[2022-10-28 22:48:02.795200 INFO   ] utils:print_arguments:28 - 	log_interval: 100
[2022-10-28 22:48:02.795200 INFO   ] utils:print_arguments:28 - 	max_epoch: 100
[2022-10-28 22:48:02.795200 INFO   ] utils:print_arguments:30 - use_model: conformer_online
[2022-10-28 22:48:02.795200 INFO   ] utils:print_arguments:31 - ------------------------------------------------
消耗时间：416ms, 识别结果: 近几年不但我用书给女儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 97
```