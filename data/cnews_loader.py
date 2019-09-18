# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr

if sys.version_info[0] > 2:#查看当前的Python版本
    is_py3 = True
else:
    reload(sys)           #这两句是设置Python的默认编码为utf-8
    sys.setdefaultencoding("utf-8") 
    is_py3 = False


def native_word(word, encoding='utf-8'): #该函数的作用将  文本  在Python2环境下转化为utf-8编码。
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content): #在py2环境下将文件按utf-8格式解码
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'): #打开文件，返回了一个文件对象
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')  #在数据集中标签和句子是用制表符\t来分隔的
                if content:
                    contents.append(list(native_content(content)))#将每个字符转化为列表元素
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir) # _表示的是无关紧要可以抛弃的值，这里主要是为了接受处理过返回的训练集文本列表

    all_data = []
    for content in data_train:          # 列表的添加方法，append单纯添加元素在末尾，extend处理对象是可迭代对象，并且分别迭代出来添加到末尾。
        all_data.extend(content)        #此时列表中所有元素是，数据集文本中所有的字。

    counter = Counter(all_data)         #Counter作用是将统计可迭代对象中元素的个数，形式为元素为键，个数为值。
    count_pairs = counter.most_common(vocab_size - 1) #统计出现次数最多vocab_size - 1 元素，返回的是个可迭代对象，元素为元组，内容为（元素，个数）
    words, _ = list(zip(*count_pairs))                #只取出元组中元素
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)                   #在字列表前加上一<pad>字符
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n') #将列表元素按行写入，join的用法将可迭代 字符元素 按指定字符串连接。


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words)))) #将元组转换为字典
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):                           #此时content中表示的是 以id为元素的文本序列
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id]) #contents是一个嵌套列表。
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))           #相当于打乱了整个数据矩阵的行索引
    x_shuffle = x[indices]                        
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
