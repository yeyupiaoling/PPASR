import json

from tqdm import tqdm


def create_data():
    with open('../dataset/manifest.train', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fp = open('../dataset/corpus.txt', 'w', encoding='utf-8')
    for line in tqdm(lines):
        data = json.loads(line)
        text = data['text']
        text = ' '.join(text)
        fp.write(f'{text}\n')
    fp.close()


if __name__ == '__main__':
    create_data()
