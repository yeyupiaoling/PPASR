# 评估

执行下面这个脚本对模型进行评估，通过字符错误率来评价模型的性能，详细参数请查看该程序。
```shell
python eval.py --resume_model=models/deepspeech2/best_model
```

输出结果：
```
----------- 额外配置参数 -----------
batch_size: 32
configs: configs/config_zh.yml
resume_model: models/{}_{}/best_model/
------------------------------------------------
----------- 配置文件参数 -----------
ctc_beam_search_decoder: {'alpha': 2.2, 'beta': 4.3, 'beam_size': 300, 'num_processes': 10, 'cutoff_prob': 0.99, 'cutoff_top_n': 40, 'language_model_path': 'lm/zh_giga.no_cna_cmn.prune01244.klm'}
dataset: {'batch_size': 32, 'num_workers': 4, 'min_duration': 0.5, 'max_duration': 20, 'train_manifest': 'dataset/manifest.train', 'test_manifest': 'dataset/manifest.test', 'dataset_vocab': 'dataset/vocabulary.txt', 'mean_std_path': 'dataset/mean_std.json', 'noise_manifest_path': 'dataset/manifest.noise'}
decoder: ctc_beam_search
metrics_type: cer
num_epoch: 65
optimizer: {'learning_rate': '5e-5', 'gamma': 0.93, 'clip_norm': 3.0, 'weight_decay': '1e-6'}
preprocess: {'feature_method': 'fbank', 'n_mels': 80, 'n_mfcc': 40, 'sample_rate': 16000, 'use_dB_normalization': True, 'target_dB': -20}
use_model: deepspeech2
------------------------------------------------
W0918 10:33:58.960235 16295 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0918 10:33:58.963088 16295 device_context.cc:422] device: 0, cuDNN Version: 7.6.
100%|██████████████████████████████| 45/45 [00:09<00:00,  4.50it/s]
评估消耗时间：10s，字错率：0.095808
```
