import re
import json
import copy
import sys
import numpy as np
import pandas as pd


EN_DICT = {
    '疾病和诊断': ['B-DIS', 'I-DIS'],
    '影像检查': ['B-SCR', 'I-SCR'],
    '实验室检验': ['B-LAB', 'I-LAB'],
    '手术': ['B-OPE', 'I-OPE'],
    '药物': ['B-MED', 'I-MED'],
    '解剖部位': ['B-POS', 'I-POS'],
    'Others': 'O'
}


def filter_chars(text):
    """过滤无用字符
    :param text: 文本
    """
    # 找出文本中所有非中，英和数字的字符
    add_chars = set(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', text))
    extra_chars = set(r"""!！￥$%*（）()-——【】:：“”";；'‘’，。？,.?、/+-""")
    add_chars = add_chars.difference(extra_chars)

    # 替换特殊字符组合
    text = re.sub('{IMG:.?.?.?}', '', text)
    text = re.sub(r'<!--IMG_\d+-->', '', text)
    text = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)  # 过滤网址
    text = re.sub('<a[^>]*>', '', text).replace("</a>", "")  # 过滤a标签
    text = re.sub('<P[^>]*>', '', text).replace("</P>", "")  # 过滤P标签
    text = re.sub('<strong[^>]*>', ',', text).replace("</strong>", "")  # 过滤strong标签
    text = re.sub('<br>', ',', text)  # 过滤br标签
    text = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text).replace("()", "")  # 过滤www开头的网址
    text = re.sub(r'\s', '', text)  # 过滤不可见字符
    text = re.sub('Ⅴ', 'V', text)

    # 清洗
    for c in add_chars:
        text = text.replace(c, '')
    return text


def get_pseudo_data_for_predict(mode):
    """将伪标签数据构造为待预测
    """
    with open('./data/pseudo/ncz_ner_unlabeled.txt', 'r', encoding='utf-8') as f_ncz, \
            open('./data/pseudo/ner_unlabeled.txt', 'r', encoding='utf-8') as f_unla:
        if mode == 'all':
            texts = [filter_chars(line.strip()) for line in f_ncz]
            texts += [filter_chars(line.strip()) for line in f_unla]
        elif mode =='ncz':
            texts = [filter_chars(line.strip()) for line in f_ncz]
        elif mode =='ccks':
            texts = [filter_chars(line.strip()) for line in f_unla]
        else:
            raise ValueError('Mode Error')


    tags = [['O'] * len(t) for t in texts]

    with open('./data/pseudo/sentences.txt', 'w', encoding='utf-8') as f_sen, \
            open('./data/pseudo/tags.txt', 'w', encoding='utf-8') as f_tag:
                #open('./is_pseudo.txt', 'w', encoding='utf-8') as f_is_pseudo:
        # 逐行写入
        for sentence, tag in zip(texts, tags):
            assert len(sentence) == len(tag)
            f_sen.write('{}\n'.format(' '.join(sentence)))
            f_tag.write('{}\n'.format(' '.join(tag)))
            # f_is_pseudo.write('{}\n'.format('1'))


def add_pseudo_data_to_train():
    """将伪标签数据追加到训练集
    """
    with open('./experiments/ex1/submit_pseudo.txt', 'r', encoding='utf-8') as f:
        data_src_li = [dict(eval(line.strip())) for idx, line in enumerate(f)]

    data_all_text = []
    data_all_tag = []

    for data in data_src_li:
        # 取文本和标注
        # 去掉原文本前后的回车换行符
        # 将原文本中间的回车换行符替换成UNK（符合源数据标注规则）
        # 将特殊字符替换为UNK
        data_ori = list(data['originalText'].strip().replace('\r\n', '✄').replace(' ', '✄'))
        data_text = copy.deepcopy(data_ori)
        data_entities = data['entities']

        for entity in data_entities:
            # 取当前实体类别
            en_type = entity['label_type']
            # 取当前实体标注
            en_tags = EN_DICT[en_type]  # ['B-XXX', 'I-XXX']
            start_ind = entity['start_pos']
            end_ind = entity['end_pos']
            # 替换实体
            data_text[start_ind] = en_tags[0]
            data_text[start_ind + 1:end_ind] = [en_tags[1] for _ in range(end_ind - start_ind - 1)]
        # 替换非实体
        for idx, item in enumerate(data_text):
            # 如果元素不是已标注的命名实体
            if len(item) != 5:
                data_text[idx] = EN_DICT['Others']
        # sanity check
        assert len(data_ori) == len(data_text), f'生成的标签与原文本长度不一致！'
        data_all_text.append(data_ori)
        data_all_tag.append(data_text)
    # sanity check
    assert len(data_all_text) == len(data_all_tag), '样本数不一致！'

    # 写入训练集
    with open('./data/train/sentences.txt', 'a', encoding='utf-8') as file_sentences, \
            open('./data/train/tags.txt', 'a', encoding='utf-8') as file_tags, \
                open('./data/train/is_pseudo.txt', 'a', encoding='utf-8') as file_is_pseudo:
        # 逐行对应写入
        for sentence, tag in zip(data_all_text, data_all_tag):
            file_sentences.write('{}\n'.format(' '.join(sentence)))
            file_tags.write('{}\n'.format(' '.join(tag)))
            file_is_pseudo.write('{}\n'.format('1'))

def ncz_process():
    path = './data/pseudo/ncz_unlabel_data.xlsx'
    data_df = pd.read_excel(path)
    data_df = data_df.drop(index=0)
    samples = data_df['住院病程记录'].tolist()
    examples = []
    for sample in samples:
       examples.extend([sample.replace('\r','').replace('\n','').replace('\t','').replace(' ','').strip()])
    
    with open('./data/pseudo/ncz_ner_unlabeled.txt','w',encoding='utf-8') as f_n:
        for example in examples:
            f_n.write('{}\n'.format(''.join(example)))

