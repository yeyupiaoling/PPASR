import Levenshtein as Lev


def cer(s1, s2):
    """
   通过计算两个字符串的距离，得出字错率

    Arguments:
        s1 (string): 比较的字符串
        s2 (string): 比较的字符串
    """
    s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
    return Lev.distance(s1, s2)
