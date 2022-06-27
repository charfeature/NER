import copy
from utils import EN_DICT
from metrics import f1_score, accuracy_score
from metrics import classification_report
import logging

def test(params,mode,verbose=True):

    with open(params.params_path / f'submit_{mode}.txt', 'r', encoding='utf-8') as f:
        data_src_li = [dict(eval(line.strip())) for line in f]

    data_all_text = []
    data_all_tag = []

    for data in data_src_li:
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
    pred_tags = data_all_tag

    true_tags = []
    with open(params.data_dir / f'{mode}/tags.txt', 'r', encoding='utf-8') as f_tag:
        for tag in f_tag:
            true_tags.append(tag.strip().split(' '))

    assert len(pred_tags) == len(true_tags), 'len(pred_tags) is not equal to len(true_tags)!'

    # logging loss, f1 and report
    metrics = {}
    p,r,f1 = f1_score(true_tags, pred_tags)
    accuracy = accuracy_score(true_tags, pred_tags)
    # metrics['loss'] = loss_avg()
    metrics['p'] = p
    metrics['r'] = r
    metrics['f1'] = f1
    metrics['accuracy'] = accuracy
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mode) + metrics_str)

    # f1 classification report
    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics