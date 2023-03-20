//兼容
window.URL = window.URL || window.webkitURL;
//获取计算机的设备：摄像头或者录音设备
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

var PPASRRecorder = function (stream, url, textResult) {
    var socket = new WebSocket(url);
    var sampleBits = 16; //输出采样数位 8, 16
    var sampleRate = 16000; //输出采样率
    var context = new AudioContext(); //创建一个音频环境对象
    var audioInput = context.createMediaStreamSource(stream);
    // 第二个和第三个参数指的是输入和输出都是单声道，1是单声道，2是双声道。
    var recorder = context.createScriptProcessor(4096 * 4, 1, 1);
    var audioData = {
        size: 0, //录音文件长度
        buffer: [], //录音缓存
        inputSampleRate: context.sampleRate, //输入采样率
        inputSampleBits: 16, //输入采样数位 8, 16
        outputSampleRate: sampleRate, //输出采样数位
        oututSampleBits: sampleBits, //输出采样率
        clear: function () {
            this.buffer = [];
            this.size = 0;
        },
        input: function (data) {
            this.buffer.push(new Float32Array(data));
            this.size += data.length;
        },
        compress: function () { //合并压缩
            //合并
            var data = new Float32Array(this.size);
            var offset = 0;
            for (var i = 0; i < this.buffer.length; i++) {
                data.set(this.buffer[i], offset);
                offset += this.buffer[i].length;
            }
            //压缩
            var compression = parseInt(this.inputSampleRate / this.outputSampleRate);
            var length = data.length / compression;
            var result = new Float32Array(length);
            var index = 0, j = 0;
            while (index < length) {
                result[index] = data[j];
                j += compression;
                index++;
            }
            return result;
        },
        encodePCM: function () {
            var sampleRate = Math.min(this.inputSampleRate, this.outputSampleRate);
            var sampleBits = Math.min(this.inputSampleBits, this.oututSampleBits);
            var bytes = this.compress();
            var dataLength = bytes.length * (sampleBits / 8);
            var buffer = new ArrayBuffer(dataLength);
            var data = new DataView(buffer);
            var offset = 0;
            for (var i = 0; i < bytes.length; i++, offset += 2) {
                var s = Math.max(-1, Math.min(1, bytes[i]));
                data.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            return new Blob([data]);
        }
    };

    this.start = function () {
        audioInput.connect(recorder);
        recorder.connect(context.destination);
    }

    this.stop = function () {
        console.log('关闭对讲以及WebSocket');
        recorder.disconnect();
        if (socket) {
            socket.close();
        }
    }

    this.getBlob = function () {
        return audioData.encodePCM();
    }
    this.clear = function () {
        audioData.clear();
    }

    // 发送语音数据
    var sendData = function () {
        var reader = new FileReader();
        reader.onload = e => {
            socket.send(e.target.result);
        };
        reader.readAsArrayBuffer(audioData.encodePCM());
        //每次发送完成则清理掉旧数据
        audioData.clear();
    };

    // 一直获取录音数据和发送数据
    recorder.onaudioprocess = function (e) {
        var inputBuffer = e.inputBuffer.getChannelData(0);
        audioData.input(inputBuffer);
        sendData();
    }

    // WebSocket客户端操作
    //连接成功建立的回调方法
    socket.onopen = () => {
        socket.binaryType = 'arraybuffer';
        this.start();
        textResult.innerText = ''
    };
    //接收到消息的回调方法
    socket.onmessage = function (MesssageEvent) {
        //返回结果
        let jsonStr = MesssageEvent.data;
        console.log(jsonStr)
        let data = JSON.parse(jsonStr)
        let code = data['code'];
        if (code === 0){
            textResult.innerText = data['result']
        }else {
            let msg = data['msg'];
            alert('报错，错误信息：' + msg)
        }
    }
    //连接关闭的回调方法
    socket.onerror = function (err) {
        console.info(err)
        textResult.innerText = err
    }
    //关闭websocket连接
    socket.onclose = function (msg) {
        console.info(msg);
    };
};

// WebSocket客户端
PPASRWebSocket = function useWebSocket(url, record, textResult) {
    ws = new WebSocket(url);
    //连接成功建立的回调方法
    ws.onopen = function () {
        ws.binaryType = 'arraybuffer';
        record.start();
        textResult.innerText = ''
    };
    //接收到消息的回调方法
    ws.onmessage = function (MesssageEvent) {
        //返回结果
        var jsonStr = MesssageEvent.data;
        console.log(jsonStr)
        textResult.innerText = JSON.parse(jsonStr)['result']
    }
    //连接关闭的回调方法
    ws.onerror = function (err) {
        console.info(err)
        textResult.innerText = err
    }
    //关闭websocket连接
    ws.onclose = function (msg) {
        console.info(msg);
    };
}

//抛出异常
PPASRRecorder.throwError = function (message) {
    alert(message);
    throw new function () {
        this.toString = function () {
            return message;
        }
    }
}
//是否支持录音
PPASRRecorder.canRecording = (navigator.getUserMedia != null);
//获取录音机
PPASRRecorder.get = function (callback, url, textarea) {
    if (callback) {
        if (navigator.getUserMedia) {
            navigator.getUserMedia(
                {audio: true} //只启用音频
                , function (stream) {
                    var record = new PPASRRecorder(stream, url, textarea);
                    callback(record);
                }
                , function (error) {
                    switch (error.code || error.name) {
                        case 'PERMISSION_DENIED':
                        case 'PermissionDeniedError':
                            PPASRRecorder.throwError('用户拒绝提供信息。');
                            break;
                        case 'NOT_SUPPORTED_ERROR':
                        case 'NotSupportedError':
                            PPASRRecorder.throwError('浏览器不支持硬件设备。');
                            break;
                        case 'MANDATORY_UNSATISFIED_ERROR':
                        case 'MandatoryUnsatisfiedError':
                            PPASRRecorder.throwError('无法发现指定的硬件设备。');
                            break;
                        default:
                            PPASRRecorder.throwError('无法打开麦克风。异常信息:' + (error.code || error.name));
                            break;
                    }
                });
        } else {
            window.alert('不是HTTPS协议或者localhost地址，不能使用录音功能！')
            PPASRRecorder.throwErr('当前浏览器不支持录音功能。');
            return;
        }
    }
};