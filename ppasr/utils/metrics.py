import Levenshtein as Lev


def cer(prediction, label):
    """
   通过计算两个字符串的距离，得出字错率

    Arguments:
        prediction (string): 比较的字符串
        label (string): 比较的字符串
    """
    prediction, label, = prediction.replace(" ", ""), label.replace(" ", "")
    return Lev.distance(prediction, label) / float(len(label))


def wer(prediction, label):
    start = 34
    word_dict = {}
    prediction = prediction.split(" ")
    label = label.split(" ")
    for s in prediction:
        if s not in word_dict.keys():
            word_dict[s] = start + len(word_dict)
    for s in label:
        if s not in word_dict.keys():
            word_dict[s] = start + len(word_dict)
    s3 = ''.join([chr(word_dict[k]) for k in prediction])
    s4 = ''.join([chr(word_dict[k]) for k in label])
    return cer(s3, s4)
