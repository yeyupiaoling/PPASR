import argparse
import os
import random
from pathlib import Path
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice
from modelscope import snapshot_download
from tqdm import tqdm


def generate(args):
    # construct dataset for generate
    sentences = []
    with open(args.text, 'rt', encoding='utf-8') as f:
        for line in f:
            utt_id, sentence = line.strip().split()
            sentences.append((utt_id, sentence))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(os.path.dirname(args.annotation_path))
    annotation_path.mkdir(parents=True, exist_ok=True)
    start_num = 0
    if os.path.exists(args.annotation_path):
        with open(args.annotation_path, 'r', encoding='utf-8') as f_ann:
            start_num = len(f_ann.readlines())
    f_ann = open(args.annotation_path, 'a', encoding='utf-8')

    snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
    cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
    print(f"支持说话人：{cosyvoice.list_avaliable_spks()}")
    # 开始生成音频
    for i in tqdm(range(start_num, len(sentences))):
        utt_id, sentence = sentences[i]
        save_audio_path = str(output_dir / (utt_id + ".wav"))
        # 执行合成语音
        speaker = random.choice(["中文男", "中文女"])
        for j in cosyvoice.inference_sft(sentence, speaker):
            save_audio_path = save_audio_path.replace('\\', '/')
            torchaudio.save(save_audio_path, j['tts_speech'], 22050)
        sentence = sentence.replace('。', '').replace('，', '').replace('！', '').replace('？', '')
        f_ann.write(f'{save_audio_path[6:]}\t{sentence}\n')
        f_ann.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",
                        type=str,
                        default='./corpus.txt',
                        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='../../dataset/audio/generate',
                        help="output audio dir.")
    parser.add_argument("--annotation_path",
                        type=str,
                        default='../../dataset/annotation/generate.txt',
                        help="audio annotation path.")
    parser.add_argument("--device",
                        type=str,
                        default="gpu",
                        help="device type to use.")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
