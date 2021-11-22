import re


def tokenize(sentence):
    """
    函数功能：将一个句子拆分成一个个单词列表
    1、先说sub是替换的意思。
    2、.是匹配任意字符（除换行符外）*是匹配前面的任意字符一个或多个
    3、？是非贪婪。
    4、组合起来的意思是将"<"和中间的任意字符">" 换为空字符串""
    由于有？是非贪婪。 所以是匹配"<"后面最近的一个">"
    :param sentence:
    :return:
    """
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result
if __name__ == '__main__':
    print(re.sub('|!|#|%|_','','hello world_ %'))
    print(tokenize('hello world_ %'))
    print(" a  ".split(" "))