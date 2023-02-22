import os

from ppasr.decoders.swig_wrapper import Scorer, CTCBeamSearchDecoder
from ppasr.decoders.swig_wrapper import ctc_beam_search_decoding_batch, ctc_beam_search_decoding
from ppasr.utils.utils import download


class BeamSearchDecoder:
    def __init__(self, alpha, beta, beam_size, cutoff_prob, cutoff_top_n, vocab_list, num_processes=10,
                 blank_id=0, language_model_path='lm/zh_giga.no_cna_cmn.prune01244.klm'):
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.vocab_list = vocab_list
        self.num_processes = num_processes
        self.blank_id = blank_id
        if not os.path.exists(language_model_path) and language_model_path == 'lm/zh_giga.no_cna_cmn.prune01244.klm':
            print('=' * 70)
            language_model_url = 'https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
            print("语言模型不存在，正在下载，下载地址： %s ..." % language_model_url)
            os.makedirs(os.path.dirname(language_model_path), exist_ok=True)
            download(url=language_model_url, download_target=language_model_path)
            print('=' * 70)
        print('=' * 70)
        print("初始化解码器...")
        assert os.path.exists(language_model_path), f'语言模型不存在：{language_model_path}'
        self._ext_scorer = Scorer(alpha, beta, language_model_path, vocab_list)
        lm_char_based = self._ext_scorer.is_character_based()
        lm_max_order = self._ext_scorer.get_max_order()
        lm_dict_size = self._ext_scorer.get_dict_size()
        print(f"language model: "
              f"model path = {language_model_path}, "
              f"is_character_based = {lm_char_based}, "
              f"max_order = {lm_max_order}, "
              f"dict_size = {lm_dict_size}")
        batch_size = 1
        self.beam_search_decoder = CTCBeamSearchDecoder(vocab_list, batch_size, beam_size, num_processes, cutoff_prob,
                                                        cutoff_top_n, self._ext_scorer, self.blank_id)
        print("初始化解码器完成!")
        print('=' * 70)

    # 单个数据解码
    def decode_beam_search_offline(self, probs_split):
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(self.alpha, self.beta)
        # beam search decode
        beam_search_result = ctc_beam_search_decoding(probs_seq=probs_split,
                                                      vocabulary=self.vocab_list,
                                                      beam_size=self.beam_size,
                                                      ext_scoring_func=self._ext_scorer,
                                                      cutoff_prob=self.cutoff_prob,
                                                      cutoff_top_n=self.cutoff_top_n,
                                                      blank_id=self.blank_id)
        return beam_search_result[0]

    # 一批数据解码
    def decode_batch_beam_search_offline(self, probs_split):
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(self.alpha, self.beta)
        # beam search decode
        self.num_processes = min(self.num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoding_batch(probs_split=probs_split,
                                                             vocabulary=self.vocab_list,
                                                             beam_size=self.beam_size,
                                                             num_processes=self.num_processes,
                                                             ext_scoring_func=self._ext_scorer,
                                                             cutoff_prob=self.cutoff_prob,
                                                             cutoff_top_n=self.cutoff_top_n,
                                                             blank_id=self.blank_id)
        results = [result[0][1] for result in beam_search_results]
        return results

    def decode_chunk(self, probs, logits_lens):
        """流式解码

        Args:
            probs (list(list(float))):一个batch模型输入的结构
            logits_lens (list(int)): 一个batch模型输出的长度
        """
        has_value = (logits_lens > 0).tolist()
        has_value = ["true" if has_value[i] is True else "false" for i in range(len(has_value))]
        probs_split = [probs[i, :l, :].tolist() if has_value[i] else probs[i].tolist()
                       for i, l in enumerate(logits_lens)]
        self.beam_search_decoder.next(probs_split, has_value)

        batch_beam_results = self.beam_search_decoder.decode()
        batch_beam_results = [[(res[0], res[1]) for res in beam_results] for beam_results in batch_beam_results]
        results_best = [result for result in batch_beam_results]
        return results_best[0][0]

    def reset_decoder(self):
        batch_size = 1
        self.beam_search_decoder.reset_state(batch_size, self.beam_size, self.num_processes,
                                             self.cutoff_prob, self.cutoff_top_n)
