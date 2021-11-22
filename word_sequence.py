"""
文本序列化
将文本转化成对应的张量才能进行处理
"""
import os
import pickle

from tqdm import tqdm  #进度条

from utils import tokenize


class WordSequence:
    UNK_TAG = '<UNK>'
    PAD_TAG = '<PAD'
    # unk用来标记词典中未出现过的字符串
    # pad用来对不到设置的规定长度句子进行数字填充
    UNK = 0
    PAD = 1

    def __init__(self):
        # self.dict用来对于词典中每种给一个对应的序号
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        # 统计每种单词的数量
        self.count = {}

    def fit(self, sentence):
        """
        统计词频
        :param sentence: 一个句子 ['今','天','我','们','很','开','心']
        :return:
        """
        for word in sentence:
            # 字典(Dictionary) get(key,default=None) 函数返回指定键的值，如果值不在字典中返回默认值
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=0, max_count=None, max_features=None):
        """
        根据条件构建词典
        :param min_count: 最小词频
        :param max_count: 最大词频
        :param max_features: 最大词语数
        :return:
        """
        if min_count is not None:
            # items()函数以列表返回可遍历的（键，值）元组数组
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # 排序
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features])
        for word in self.count:
            self.dict[word] = len(self.dict)

        # 把dict进行反转，就是键值和关键字进行反转
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为数字序列
        :param sentence: 句子
        :param max_len: 句子最大长度
        :return:
        """
        if len(sentence) > max_len:
            # 句子太长时进行截断
            sentence = sentence[:max_len]
        else:
            # 句子长度不够标准长度时，进行填充
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
        # 句子中的单词没有出现在词典中设置为数字0（self.UNK)
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把数字序列转化为字符
        :param indices: 数字序列
        :return:
        """
        return [self.inverse_dict.get(i, '<UNK>') for i in indices]

    def __len__(self):
        # 返回词典个数
        return len(self.dict)


def load_data_to_model():
    """
    将原始数据序列化到pkl模型
    :return:
    """
    ws = WordSequence()
    data_path = r"data/aclImdb"
    total_path = []
    for temp_path in [r"/train/pos", r"/test/neg"]:
        cur_path = data_path + temp_path
        # 添加积极和消极评论的所有文件
        total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]
    for file in tqdm(total_path, total=len(total_path)):
        # 读取评论内容
        content = open(file=file, encoding='utf-8').read()
        # 将评论分成一个个单词列表
        sentence = tokenize(content)  # tokenize有问题
        ws.fit(sentence)
    # 开始构建词典
    ws.build_vocab(min_count=5, max_count=15000)
    print(len(ws))
    # dump将数据通过特殊的形式转换为只有python语言认识的字符串，并写入文件
    pickle.dump(ws, open("./model/ws.pkl", "wb"))


if __name__ == '__main__':
   # load_data_to_model()
    sentences = [
        ["今天", "天气", "很", "好"],
        ["我们", "出去", "玩", "网球"]
    ]
    ws = WordSequence()
    for sentence in sentences:
        # 统计词频
        ws.fit(sentence)

    # 构建词典
    ws.build_vocab(min_count=0)
    print(ws.dict)

    # 将一句话转化成数字序列表示
    ret = ws.transform(["今天", "我们", "吃", "很", "好", "的"], 20)
    print(ret)
