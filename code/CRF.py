# 基于CRF的命名实体识别(NER)模型
# 该NER任务使用BIO三位标注法

import json
import sklearn_crfsuite
from sklearn import metrics  # 导入评估指标模块
from itertools import chain  # 将多个可迭代对象连接成一个连续序列

# 一、数据处理（逐字拆分并给予标签）
def data_process(path):
    # 逐行读取JSON数据，存储到json_data列表
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))

    # json_data中每一条数据的格式为
    '''
    {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
     'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
    '''

    data = []
    # 遍历json_data中每组数据
    for i in range(len(json_data)):
        # 将标签全初始化为'O'
        label = ['O'] * len(json_data[i]['text'])

        # 遍历'label'中几组实体类型，如样例中'name'和'company'
        for n in json_data[i]['label']:

            # 遍历实体中几组文本，如样例中'name'下的'叶老桂'（有多组文本的情况，样例中只有一组）
            for key in json_data[i]['label'][n]:

                # 遍历文本中几组下标，如样例中[[9, 11]]（有时某个文本在该段中出现两次，则会有两组下标）
                for n_list in range(len(json_data[i]['label'][n][key])):

                    # 记录实体开始下标和结尾下标，这里每组位置是一个二元组 [start, end]
                    start = json_data[i]['label'][n][key][n_list][0]   # eg公司实体"浙商银行"，start=0, end=3
                    end = json_data[i]['label'][n][key][n_list][1]     # eg人名实体"叶老桂"，start=9, end=11

                    # 将开始位置标记为'B-' + n，如'B-' + 'name'即'B-name'
                    # 后续位置标记为'I-' + n
                    label[start] = 'B-' + n
                    label[start + 1: end + 1] = ['I-' + n] * (end - start)

        # 对字符串进行字符级分割
        # 英文文本如'bag'分割成'b'，'a'，'g'三位字符，数字文本如'125'分割成'1'，'2'，'5'三位字符
        texts = []
        for t in json_data[i]['text']:
            texts.append(t)

        # 最终数据结构为[字符列表, 标签列表]。
        data.append([texts, label])
    return data

# 处理完的json文件变成如下格式
'''
    [['浙', '商', '银', '行', '企', '业', '信', '贷', '部', '叶', '老', '桂', '博', '士', '则', '从', '另', '一', 
    '个', '角', '度', '对', '五', '道', '门', '槛', '进', '行', '了', '解', '读', '。', '叶', '老', '桂', '认', 
    '为', '，', '对', '目', '前', '国', '内', '商', '业', '银', '行', '而', '言', '，'], 
    ['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 
    'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
'''

# 二、特征工程（滑动窗口方法构建特征）
# 1.判断字符是否是英文
def is_english(c):
    if ord(c.lower()) >= 97 and ord(c.lower()) <= 122:  # ord()：获取字符的Unicode编码
        return True
    else:
        return False

# 2.构建特征函数
'''
    序列中的每一个字符处理成如下格式：
    {'bias': 1.0,
     'word': '商',
     'word.isdigit()': False,
     'word.is_english()': False,
     '-1:word': '浙',
     '-1:word.isdigit()': False,
     '-1:word.is_english()': False,
     '+1:word': '银',
     '+1:word.isdigit()': False,
     '+1:word.is_english()': False}
'''
def word2features(sent, i):
    # 本代码采用大小为3的滑动窗口构造特征，特征长度可以不同，当然可以增大窗口或增加其他特征
    word = sent[i][0]  # 从输入的句子序列sent中提取第i个位置的字符
    '''
        sent = [
        ('浙', 'B-company'), 
        ('商', 'I-company'), 
        ... ]
    '''
    features = {
        'bias': 1.0,                            # 偏置项（总是1.0，是模型自动学习的权重。）
        'word': word,                           # 当前字符
        'word.isdigit()': word.isdigit(),       # 字符是否数字
        'word.is_english()': is_english(word),  # 字符是否英文
    }

    # 添加上文特征
    if i > 0:
        word = sent[i - 1][0]
        features.update({
            '-1:word': word,
            '-1:word.isdigit()': word.isdigit(),
            '-1:word.is_english()': is_english(word),
        })
    else:
        features['BOS'] = True   # 如果是句首词，添加BOS(Beginning of Sentence)特征

    # 添加下文特征
    if i < len(sent) - 1:
        word = sent[i + 1][0]
        features.update({
            '+1:word': word,
            '+1:word.isdigit()': word.isdigit(),
            '+1:word.is_english()': is_english(word),
        })
    else:
        features['EOS'] = True   # 如果是句末词，添加EOS(End of Sentence)特征
    return features

# 将整个句子转换为特征列表
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# 从句子中提取命名实体标签序列
def sent2labels(sent):
    return [label for label in sent]

# 三、模型训练
train = data_process('train.json')
valid = data_process('dev.json')
# eg:[ ['浙', '商', '银', ...], ['B-company', 'I-company', 'I-company', ...] ]
print('训练集长度:', len(train))
print('验证集长度:', len(valid))

X_train = [sent2features(s[0]) for s in train]  # s[0]获取每个训练样本的字符列表
y_train = [sent2labels(s[1]) for s in train]    # s[1]获取每个训练样本的标签列表
X_dev = [sent2features(s[0]) for s in valid]    # sent2features函数将每个字符及其上下文转换为特征字典
y_dev = [sent2labels(s[1]) for s in valid]      # sent2labels函数直接返回标签列表
print(X_train[0][1], end="\n")                  # 打印训练集第一个样本的第二个字符的特征

crf_model = sklearn_crfsuite.CRF(
    algorithm = 'lbfgs',  # L-BFGS（拟牛顿法优化）
    c1 = 0.1,  # L1正则系数，用于特征选择，可能会使一些不重要的特征的权重变为0
    c2 = 0.1,  # L2正则系数，用于权重平滑，防止某些特征权重过大导致过拟合
    max_iterations = 100,  # 最大迭代次数
    all_possible_transitions = True  # 允许所有可能的标签转移,例如：允许"B-name"直接转移到"I-company"
)
try:
    crf_model.fit(X_train, y_train)
except:
    pass

# 从训练好的模型中获取所有可能的标签类别（这个列表将用于后续的评估和预测）
labels = list(crf_model.classes_)

# 四、评估指标
# 1.移除'O'标签并预测（因为大部分标签都是'O'，评估时关注实体标签更有意义）
labels.remove("O")
y_pred = crf_model.predict(X_dev)   # 对验证集的特征表示进行预测，返回预测的标签序列

# 2.将嵌套的标签列表展平为一维列表（评估指标函数需要一维的标签序列）
y_dev = list(chain.from_iterable(y_dev))      # y_dev的格式是列表的列表
y_pred = list(chain.from_iterable(y_pred))    # y_pred与y_dev的格式一样

# 3.计算F1分数
print('weighted F1 score:', metrics.f1_score(y_dev, y_pred,
                            average='weighted', labels=labels)) # 各类别的样本量加权平均

# 4.标签排序（先按标签类型排序，再按前缀（B/I）排序）
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

# 5.打印详细分数报告
print(metrics.classification_report(
    y_dev, y_pred, labels=sorted_labels, digits=3 # 数值保留3位小数
))

# 6.查看转移概率和发射概率
# print('CRF转移概率：', crf_model.transition_features_)
# print('CRF发射概率：', crf_model.state_features_)
print('CRF转移概率（前5个）：', list(crf_model.transition_features_.items())[:5])
print('CRF发射概率（前5个）：', list(crf_model.state_features_.items())[:5])