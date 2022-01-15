import Levenshtein as Lev


def cer(s1, s2):
    """
   通过计算两个字符串的距离，得出字错率

    Arguments:
        s1 (string): 比较的字符串
        s2 (string): 比较的字符串
    """
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
    return Lev.distance(s1, s2) / float(len(s2))


def wer(s1, s2):
    start = 34
    word_dict = {}
    s1 = s1.split(" ")
    s2 = s2.split(" ")
    for s in s1:
        if s not in word_dict.keys():
            word_dict[s] = start + len(word_dict)
    for s in s2:
        if s not in word_dict.keys():
            word_dict[s] = start + len(word_dict)
    s3 = ''.join([chr(word_dict[k]) for k in s1])
    s4 = ''.join([chr(word_dict[k]) for k in s2])
    return cer(s3, s4)
