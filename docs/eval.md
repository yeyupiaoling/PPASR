# 评估

执行下面这个脚本对模型进行评估，通过字符错误率来评价模型的性能，详细参数请查看该程序。
```shell
python eval.py --resume_model=models/deepspeech2/best_model
```

输出结果：
```
-----------  Configuration Arguments -----------
alpha: 2.2
batch_size: 32
beam_size: 300
beta: 4.3
cutoff_prob: 0.99
cutoff_top_n: 40
dataset_vocab: dataset/vocabulary.txt
decoder: ctc_beam_search
lang_model_path: lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: dataset/mean_std.npz
num_proc_bsearch: 10
num_workers: 8
resume_model: models/deepspeech2/best_model/
test_manifest: dataset/manifest.test
use_model: deepspeech2
------------------------------------------------
W0918 10:33:58.960235 16295 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0918 10:33:58.963088 16295 device_context.cc:422] device: 0, cuDNN Version: 7.6.
100%|██████████████████████████████| 45/45 [00:09<00:00,  4.50it/s]
评估消耗时间：10s，字错率：0.095808
```
