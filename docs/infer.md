# 本地预测

我们可以使用这个脚本使用模型进行预测，如果如何还没导出模型，需要执行[导出模型](./export_model.md)操作把模型参数导出为预测模型，通过传递音频文件的路径进行识别，通过参数`--audio_path`指定需要预测的音频路径。支持中文数字转阿拉伯数字，将参数`--is_itn`设置为True即可。默认情况下，如果音频大于30秒，会通过VAD分割音频，再对短音频进行识别，拼接结果，最终得到长语音识别结果。
```shell script
python infer_path.py --audio_path=./dataset/test.wav
```

输出结果：
```
2024-09-21 15:35:48.565 | INFO     | ppasr.utils.utils:print_arguments:13 - ----------- 额外配置参数 -----------
2024-09-21 15:35:48.565 | INFO     | ppasr.utils.utils:print_arguments:15 - allow_use_vad: True
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - decoder: ctc_greedy_search
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - decoder_configs: configs/decoder.yml
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - is_itn: False
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - model_dir: models/ConformerModel_fbank/inference_model/
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - pun_model_dir: models/pun_models/
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - real_time_demo: False
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - use_gpu: True
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - use_pun: False
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:15 - audio_path: dataset/test.wav
2024-09-21 15:35:48.566 | INFO     | ppasr.utils.utils:print_arguments:16 - ------------------------------------------------
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 模型参数配置 -----------
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:32 - model_name: ConformerModel
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:23 - preprocess_conf:
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:30 - 	feature_method: fbank
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:26 - 	method_args:
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:28 - 		num_mel_bins: 80
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:32 - sample_rate: 16000
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:32 - streaming: True
2024-09-21 15:35:48.567 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:19 - ----------- 解码器参数配置 -----------
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:23 - attention_rescoring_args:
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:30 - 	beam_size: 5
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:30 - 	ctc_weight: 0.3
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:30 - 	reverse_weight: 1.0
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:23 - ctc_prefix_beam_search_args:
2024-09-21 15:35:48.568 | INFO     | ppasr.utils.utils:print_arguments:30 - 	beam_size: 5
2024-09-21 15:35:48.570 | INFO     | ppasr.utils.utils:print_arguments:33 - ------------------------------------------------
2024-09-21 15:35:49.257 | INFO     | ppasr.infer_utils.inference_predictor:__init__:38 - 已加载模型：models/ConformerModel_fbank/inference_model/inference.pth
消耗时间：794ms, 识别结果: {'text': '近几年不但我用书给女儿压岁也全身亲朋不要给女儿压岁钱而改送压岁书', 'sentences': [{'text': '近几年不但我用书给女儿压岁也全身亲朋不要给女儿压岁钱而改送压岁书', 'start': 0, 'end': 8.39}]}
```

## 模拟实时识别
这里提供一个简单的实时识别例子，如果想完整使用实时识别，可以使用`infer_gui.py`中的录音实时识别功能。在`--real_time_demo`指定为True。目前流式识别只支持贪心解码器（ctc_greedy_search）。
```shell
python infer_path.py --wav_path=./dataset/test.wav --real_time_demo=True
```

输出结果：
```
······
【实时结果】：消耗时间：69ms, 识别结果: 
【实时结果】：消耗时间：37ms, 识别结果: 
【实时结果】：消耗时间：44ms, 识别结果: 近几年
【实时结果】：消耗时间：85ms, 识别结果: 近几年不但我用
【实时结果】：消耗时间：42ms, 识别结果: 近几年不但我用书给女儿
【实时结果】：消耗时间：45ms, 识别结果: 近几年不但我用书给女儿压岁
【实时结果】：消耗时间：45ms, 识别结果: 近几年不但我用书给女儿压岁也确实
【实时结果】：消耗时间：44ms, 识别结果: 近几年不但我用书给女儿压岁也确实亲朋
【实时结果】：消耗时间：51ms, 识别结果: 近几年不但我用书给女儿压岁也确实亲朋不要给女儿
【实时结果】：消耗时间：26ms, 识别结果: 近几年不但我用书给女儿压岁也确实亲朋不要给女儿压岁钱
【实时结果】：消耗时间：25ms, 识别结果: 近几年不但我用书给女儿压岁也确实亲朋不要给女儿压岁钱而
【实时结果】：消耗时间：23ms, 识别结果: 近几年不但我用书给女儿压岁也确实亲朋不要给女儿压岁钱而改送压
【实时结果】：消耗时间：23ms, 识别结果: 近几年不但我用书给女儿压岁也确实亲朋不要给女儿压岁钱而改送压岁书
```


## Web部署

在服务器执行下面命令通过创建一个Web服务，通过提供HTTP接口来实现语音识别。启动服务之后，如果在本地运行的话，在浏览器上访问`http://localhost:5000`，否则修改为对应的 IP地址。打开页面之后可以选择上传长音或者短语音音频文件，也可以在页面上直接录音，录音完成之后点击上传，播放功能只支持录音的音频。支持中文数字转阿拉伯数字，将参数`--is_itn`设置为True即可，默认为False。
```shell script
python infer_server.py
```

打开页面如下：
![录音测试页面](./images/infer_server.jpg)


## GUI界面部署
通过打开页面，在页面上选择长语音或者短语音进行识别，也支持录音识别实时识别，带播放音频功能。该程序可以在本地识别，也可以通过指定服务器调用服务器的API进行识别。
```shell script
python infer_gui.py
```

打开界面如下：
![GUI界面](./images/infer_gui.jpg)
