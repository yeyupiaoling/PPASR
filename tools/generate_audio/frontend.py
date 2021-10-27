import re
from typing import Dict
from typing import List

import numpy as np
import paddle

from parakeet.frontend.zh_frontend import Frontend as cnFrontend


class Frontend():
    def __init__(self, phone_vocab_path=None, tone_vocab_path=None):
        self.frontend = cnFrontend()
        self.vocab_phones = {}
        self.vocab_tones = {}
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt', encoding='utf-8') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)
        if tone_vocab_path:
            with open(tone_vocab_path, 'rt', encoding='utf-8') as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            for tone, id in tone_id:
                self.vocab_tones[tone] = int(id)

    def _p2id(self, phonemes: List[str]) -> np.array:
        # replace unk phone with sp
        phonemes = [
            phn if phn in self.vocab_phones else "sp" for phn in phonemes
        ]
        phone_ids = [self.vocab_phones[item] for item in phonemes]
        return np.array(phone_ids, np.int64)

    def _t2id(self, tones: List[str]) -> np.array:
        # replace unk phone with sp
        tones = [tone if tone in self.vocab_tones else "0" for tone in tones]
        tone_ids = [self.vocab_tones[item] for item in tones]
        return np.array(tone_ids, np.int64)

    def _get_phone_tone(self, phonemes: List[str],
                        get_tone_ids: bool=False) -> List[List[str]]:
        phones = []
        tones = []
        if get_tone_ids and self.vocab_tones:
            for full_phone in phonemes:
                # split tone from finals
                match = re.match(r'^(\w+)([012345])$', full_phone)
                if match:
                    phone = match.group(1)
                    tone = match.group(2)
                    # if the merged erhua not in the vocab
                    # assume that the input is ['iaor3'] and 'iaor' not in self.vocab_phones, we split 'iaor' into ['iao','er']
                    # and the tones accordingly change from ['3'] to ['3','2'], while '2' is the tone of 'er2'
                    if len(phone) >= 2 and phone != "er" and phone[
                            -1] == 'r' and phone not in self.vocab_phones and phone[:
                                                                                    -1] in self.vocab_phones:
                        phones.append(phone[:-1])
                        phones.append("er")
                        tones.append(tone)
                        tones.append("2")
                    else:
                        phones.append(phone)
                        tones.append(tone)
                else:
                    phones.append(full_phone)
                    tones.append('0')
        else:
            for phone in phonemes:
                # if the merged erhua not in the vocab
                # assume that the input is ['iaor3'] and 'iaor' not in self.vocab_phones, change ['iaor3'] to ['iao3','er2']
                if len(phone) >= 3 and phone[:-1] != "er" and phone[
                        -2] == 'r' and phone not in self.vocab_phones and (
                            phone[:-2] + phone[-1]) in self.vocab_phones:
                    phones.append((phone[:-2] + phone[-1]))
                    phones.append("er2")
                else:
                    phones.append(phone)
        return phones, tones

    def get_input_ids(
            self,
            sentence: str,
            merge_sentences: bool=True,
            get_tone_ids: bool=False) -> Dict[str, List[paddle.Tensor]]:
        phonemes = self.frontend.get_phonemes(
            sentence, merge_sentences=merge_sentences)
        result = {}
        phones = []
        tones = []
        temp_phone_ids = []
        temp_tone_ids = []
        for part_phonemes in phonemes:
            phones, tones = self._get_phone_tone(
                part_phonemes, get_tone_ids=get_tone_ids)
            if tones:
                tone_ids = self._t2id(tones)
                tone_ids = paddle.to_tensor(tone_ids)
                temp_tone_ids.append(tone_ids)
            if phones:
                phone_ids = self._p2id(phones)
                phone_ids = paddle.to_tensor(phone_ids)
                temp_phone_ids.append(phone_ids)
        if temp_tone_ids:
            result["tone_ids"] = temp_tone_ids
        if temp_phone_ids:
            result["phone_ids"] = temp_phone_ids
        return result
