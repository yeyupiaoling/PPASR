import json

import audioread
import numpy as np
from pydub import AudioSegment


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.5):
    """解析数据列表
    持续时间在[min_duration, max_duration]之外的实例将被过滤。

    :param manifest_path: 数据列表的路径
    :type manifest_path: str
    :param max_duration: 过滤的最长音频长度
    :type max_duration: float
    :param min_duration: 过滤的最短音频长度
    :type min_duration: float
    :return: 数据列表，JSON格式
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in open(manifest_path, 'r', encoding='utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if max_duration >= json_data["duration"] >= min_duration:
            manifest.append(json_data)
    return manifest


# 读取soundfile不支持的格式音频
def audioread_load(path, dtype='float32'):
    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        n = 0
        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n = n + len(frame)
            # tack on the current frame
            y.append(frame)
    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native


# 将音频流转换为numpy
def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def opus_to_wav(opus_path, save_wav_path, rate=16000):
    source_wav = AudioSegment.from_file(opus_path)
    target_audio = source_wav.set_frame_rate(rate)
    target_audio.export(save_wav_path, format="wav")

