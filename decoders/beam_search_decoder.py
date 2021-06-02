from ctc_decoders.swig_wrapper import Scorer
from ctc_decoders.swig_wrapper import ctc_beam_search_decoder_batch


class BeamSearchDecoder:
    def __init__(self, beam_alpha, beam_beta, language_model_path, vocab_list):
        """Initialize the external scorer.
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param language_model_path: Filepath for language model. If it is
                                    empty, the external scorer will be set to
                                    None, and the decoding method will be pure
                                    beam search without scorer.
        :type language_model_path: str|None
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        """
        if language_model_path != '':
            print("begin to initialize the external scorer for decoding")
            self._ext_scorer = Scorer(beam_alpha, beam_beta, language_model_path, vocab_list)
            lm_char_based = self._ext_scorer.is_character_based()
            lm_max_order = self._ext_scorer.get_max_order()
            lm_dict_size = self._ext_scorer.get_dict_size()
            print("language model: "
                  "is_character_based = %d," % lm_char_based +
                  " max_order = %d," % lm_max_order +
                  " dict_size = %d" % lm_dict_size)
            print("end initializing scorer")
        else:
            self._ext_scorer = None
            print("no language model provided, decoding by pure beam search without scorer.")

    def decode_batch_beam_search(self, probs_split, beam_alpha, beam_beta,
                                 beam_size, cutoff_prob, cutoff_top_n,
                                 vocab_list, num_processes):
        """Decode by beam search for a batch of probs matrix input.
        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                        characters with highest probs in vocabulary will be
                        used in beam search, default 40.
        :type cutoff_top_n: int
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of str
        """
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)
        # beam search decode
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoder_batch(probs_split=probs_split,
                                                            vocabulary=vocab_list,
                                                            beam_size=beam_size,
                                                            num_processes=num_processes,
                                                            ext_scoring_func=self._ext_scorer,
                                                            cutoff_prob=cutoff_prob,
                                                            cutoff_top_n=cutoff_top_n)

        results = [result[0][1] for result in beam_search_results]
        return results