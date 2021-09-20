import _thread
import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import paddle
import soundfile
import yaml
from tqdm import tqdm
from yacs.config import CfgNode
from parakeet.models.fastspeech2 import FastSpeech2, FastSpeech2Inference
from parakeet.models.parallel_wavegan import PWGGenerator, PWGInference
from parakeet.modules.normalizer import ZScore

from frontend import Frontend


def generate(args, fastspeech2_config, pwg_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for generate
    sentences = []
    with open(args.text, 'rt', encoding='utf-8') as f:
        for line in f:
            utt_id, sentence = line.strip().split()
            sentences.append((utt_id, sentence))

    with open(args.phones_dict, "r", encoding='utf-8') as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    with open(args.speaker_dict, 'rt', encoding='utf-8') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    num_speakers = len(spk_id)
    print("num_speakers:", num_speakers)

    odim = fastspeech2_config.n_mels
    model = FastSpeech2(idim=vocab_size,
                        odim=odim,
                        num_speakers=num_speakers,
                        **fastspeech2_config["model"])

    model.set_state_dict(paddle.load(args.fastspeech2_checkpoint)["main_params"])
    model.eval()

    vocoder = PWGGenerator(**pwg_config["generator_params"])
    vocoder.set_state_dict(paddle.load(args.pwg_params))
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    frontend = Frontend(args.phones_dict)
    print("frontend done!")

    stat = np.load(args.fastspeech2_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    fastspeech2_normalizer = ZScore(mu, std)

    stat = np.load(args.pwg_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    pwg_normalizer = ZScore(mu, std)

    fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)
    pwg_inference = PWGInference(pwg_normalizer, vocoder)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(os.path.dirname(args.annotation_path))
    annotation_path.mkdir(parents=True, exist_ok=True)
    start_num = 0
    if os.path.exists(args.annotation_path):
        with open(args.annotation_path, 'r', encoding='utf-8') as f_ann:
            start_num = len(f_ann.readlines())
    f_ann = open(args.annotation_path, 'a', encoding='utf-8')
    # 开始生成音频
    for i in tqdm(range(start_num, len(sentences))):
        utt_id, sentence = sentences[i]
        # 随机说话人
        spk_id = random.randint(0, num_speakers - 1)
        try:
            input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
        except:
            continue
        phone_ids = input_ids["phone_ids"]
        flags = 0
        for part_phone_ids in phone_ids:
            with paddle.no_grad():
                mel = fastspeech2_inference(part_phone_ids, spk_id=paddle.to_tensor(spk_id))
                temp_wav = pwg_inference(mel)
                if flags == 0:
                    wav = temp_wav
                    flags = 1
                else:
                    wav = paddle.concat([wav, temp_wav])
        save_audio_path = str(output_dir / (utt_id + ".wav"))
        soundfile.write(save_audio_path, wav.numpy(), samplerate=fastspeech2_config.fs)
        f_ann.write('%s\t%s\n' % (save_audio_path[6:].replace('\\', '/'), sentence.replace('。', '').replace('，', '')
                                  .replace('！', '').replace('？', '')))
        f_ann.flush()


def main():
    parser = argparse.ArgumentParser(description="Synthesize with fastspeech2 & parallel wavegan.")
    parser.add_argument("--fastspeech2-config",
                        type=str,
                        default='models/fastspeech2_nosil_aishell3_ckpt_0.4/default.yaml',
                        help="fastspeech2 config file to overwrite default config.")
    parser.add_argument("--fastspeech2-checkpoint",
                        type=str,
                        default='models/fastspeech2_nosil_aishell3_ckpt_0.4/snapshot_iter_96400.pdz',
                        help="fastspeech2 checkpoint to load.")
    parser.add_argument("--fastspeech2-stat",
                        type=str,
                        default='models/fastspeech2_nosil_aishell3_ckpt_0.4/speech_stats.npy',
                        help="mean and standard deviation used to normalize spectrogram when training fastspeech2.")
    parser.add_argument("--pwg-config",
                        type=str,
                        default='models/parallel_wavegan_baker_ckpt_0.4/pwg_default.yaml',
                        help="parallel wavegan config file to overwrite default config.")
    parser.add_argument("--pwg-params",
                        type=str,
                        default='models/parallel_wavegan_baker_ckpt_0.4/pwg_generator.pdparams',
                        help="parallel wavegan generator parameters to load.")
    parser.add_argument("--pwg-stat",
                        type=str,
                        default='models/parallel_wavegan_baker_ckpt_0.4/pwg_stats.npy',
                        help="mean and standard deviation used to normalize spectrogram when training parallel wavegan.")
    parser.add_argument("--phones-dict",
                        type=str,
                        default="models/fastspeech2_nosil_aishell3_ckpt_0.4/phone_id_map.txt",
                        help="phone vocabulary file.")
    parser.add_argument("--speaker-dict",
                        type=str,
                        default="models/fastspeech2_nosil_aishell3_ckpt_0.4/speaker_id_map.txt",
                        help="speaker id map file.")
    parser.add_argument("--text",
                        type=str,
                        default='corpus.txt',
                        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, default='../../dataset/audio/generate', help="output audio dir.")
    parser.add_argument("--annotation_path", type=str, default='../../dataset/annotation/generate.txt',
                        help="audio annotation path.")
    parser.add_argument("--device", type=str, default="gpu", help="device type to use.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    args = parser.parse_args()
    with open(args.fastspeech2_config) as f:
        fastspeech2_config = CfgNode(yaml.safe_load(f))
    with open(args.pwg_config) as f:
        pwg_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(fastspeech2_config)
    print(pwg_config)

    generate(args, fastspeech2_config, pwg_config)


if __name__ == "__main__":
    main()
