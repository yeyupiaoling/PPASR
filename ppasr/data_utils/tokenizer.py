import os
import tempfile
from typing import List, Union

import sentencepiece as spm

from ppasr.data_utils.utils import read_manifest


class PPASRTokenizer(object):
    """文本分词器

    :param vocab_model_dir: 词汇表模型目录
    :type vocab_model_dir: str
    :param model_type: 训练词汇表模型的类型，可选值为"unigram", "word", "char"
    :type model_type: str
    :param build_vocab_size: 构建词汇表的大小，仅在使用unigram模型时有效
    :type build_vocab_size: int
    :param non_linguistic_symbols: 非语言符号列表
    :type non_linguistic_symbols: list
    :param remove_non_linguistic_symbols: 是否移除非语言符号
    :type remove_non_linguistic_symbols: bool
    :param is_build_vocab: 是否构建词汇表
    :type is_build_vocab: bool
    """
    def __init__(self,
                 vocab_model_dir: str,
                 model_type: str = "char",
                 build_vocab_size: int = None,
                 non_linguistic_symbols: List[str] = None,
                 remove_non_linguistic_symbols: bool = False,
                 is_build_vocab: bool = False):
        self.vocab_model_dir = vocab_model_dir
        self.build_vocab_size = build_vocab_size
        self.non_linguistic_symbols = non_linguistic_symbols
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        os.makedirs(self.vocab_model_dir, exist_ok=True)
        self.model_prefix = os.path.join(self.vocab_model_dir, "model")
        if not is_build_vocab:
            model_path = self.model_prefix + ".model"
            assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
            # 获取词汇表内容
            vocab_path = self.model_prefix + ".vocab"
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.token_list = [line.strip().split("\t")[0] for line in f.readlines()]
            assert len(self.token_list) == self.sp.vocab_size(), "词汇表大小不一致"
        else:
            self.smp_args = dict(model_type=model_type,
                                 model_prefix=self.model_prefix,
                                 pad_id=0,
                                 unk_id=1,
                                 eos_id=2,
                                 bos_id=-1,
                                 pad_piece="<blank>",
                                 unk_piece="<unk>",
                                 eos_piece="<eos>",
                                 input_sentence_size=1e8,
                                 character_coverage=0.9995,
                                 minloglevel=4)
            if self.build_vocab_size is not None:
                self.smp_args["vocab_size"] = self.build_vocab_size
            if model_type == "unigram":
                assert self.build_vocab_size is not None, "构建unigram模型需要指定词汇表大小"
            else:
                self.smp_args["use_all_vocab"] = True

    def build_vocab(self, manifest_paths: List[str]):
        """构建词汇表模型

        :param manifest_paths: 数据清单路径列表，格式跟项目数据列表一致
        :type manifest_paths: List[str]
        """
        fp = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding="utf-8")
        for manifest_path in manifest_paths:
            manifest_data = read_manifest(manifest_path)
            for line in manifest_data:
                text = line["text"]
                # 移除非语言符号
                if self.remove_non_linguistic_symbols:
                    for symbol in self.non_linguistic_symbols:
                        text = text.replace(symbol, "")
                fp.write(text + "\n")
        fp.close()
        spm.SentencePieceTrainer.Train(input=fp.name, **self.smp_args)
        os.unlink(fp.name)

    # 将文本转换为token列表
    def text2tokens(self, text: str) -> List[str]:
        return self.sp.EncodeAsPieces(text)

    # 将文本转换为id列表
    def text2ids(self, text: str) -> List[int]:
        return self.sp.EncodeAsIds(text)

    # 将id列表转换为文本
    def ids2text(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        return self.sp.DecodeIds(ids)

    @property
    def blank_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    # 获取词汇表大小
    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    # 获取词汇表列表
    @property
    def vocab_list(self) -> List[str]:
        return self.token_list
