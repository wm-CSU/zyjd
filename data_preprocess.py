# encoding:utf-8

import os
import re
import numpy as np


def data_read(input_path):
    """
    读数据
    :param input_path:
    :return: data: [list]  [{'testid': int, 'features_content': str, 'labels_index': [int], 'labels_num': int}]
    """
    data = []
    for line in open(input_path, 'r', encoding='utf-8'):
        line_json = eval(line)
        mid = ''.join(line_json['features_content'])
        line_json['features_content'] = char_dropout(mid)
        data.append(line_json)

    return data


def char_dropout(_string):
    my_str = re.sub(u'[a-zA-Z×]', '', _string).replace(' ', '')
    # my_str
    return my_str


def length_stic(data):
    length = []
    for one in data:
        length.append(len(one['features_content']))
    max_len = max(length)
    min_len = min(length)
    average = np.mean(length)
    per_95 = np.percentile(length, 95)
    variance = np.std(length)
    return length, max_len, min_len, average, per_95, variance


if __name__ == '__main__':
    mydata = data_read(r'data/train_v2(训练集).json')
    length, max_len, min_len, average, per_95, variance = length_stic(mydata)
    print(sorted(length))
    print(max_len, min_len, average, per_95, variance)
    pass