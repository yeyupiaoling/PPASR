class TextFeaturizer(object):
    """文本特征器，用于处理或从文本中提取特征。支持字符级的令牌化和转换为令牌索引列表

    :param vocab_filepath: 令牌索引转换词汇表的文件路径
    :type vocab_filepath: str
    """

    def __init__(self, vocab_filepath):
        self.unk = "<unk>"
        self._vocab_dict, self._vocab_list = self._load_vocabulary_from_file(vocab_filepath)

    def featurize(self, text):
        """将文本字符串转换为字符级的令牌索引列表

        :param text: 文本
        :type text: str
        :return:字符级令牌索引列表
        :rtype: list
        """
        tokens = self._char_tokenize(text)
        token_indices = []
        for token in tokens:
            if token == ' ': token = '<space>'
            # 跳过词汇表不存在的字符
            if token not in self._vocab_list:
                token = self.unk
            token_indices.append(self._vocab_dict[token])
        return token_indices

    @property
    def vocab_size(self):
        """返回词汇表大小

        :return: Vocabulary size.
        :rtype: int
        """
        return len(self._vocab_list)

    @property
    def vocab_list(self):
        """返回词汇表的列表

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._vocab_list

    def _char_tokenize(self, text):
        """Character tokenizer."""
        return list(text.strip())

    def _load_vocabulary_from_file(self, vocab_filepath):
        """Load vocabulary from file."""
        vocab_lines = []
        with open(vocab_filepath, 'r', encoding='utf-8') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line.split('\t')[0].replace('\n', '') for line in vocab_lines]
        vocab_dict = dict([(token, id) for (id, token) in enumerate(vocab_list)])
        return vocab_dict, vocab_list
