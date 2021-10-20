# 导出模型

训练保存的或者下载作者提供的模型都是模型参数，我们要将它导出为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测，详细参数请查看该程序。
```shell
python export_model.py --resume_model=models/deepspeech2/epoch_50/
```

输出结果：
```
-----------  Configuration Arguments -----------
dataset_vocab: dataset/vocabulary.txt
mean_std_path: dataset/mean_std.npz
resume_model: models/deepspeech2/epoch_50
save_model: models/deepspeech2/
use_model: deepspeech2
------------------------------------------------
[2021-09-18 10:23:47.022243] 成功恢复模型参数和优化方法参数：models/deepspeech2/epoch_50/model.pdparams

预测模型已保存：models/deepspeech2/infer
```