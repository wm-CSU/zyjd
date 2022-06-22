# encoding:utf-8

import os, json, re
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, DataCollatorWithPadding


class MyDataset(Dataset):
    def __init__(self, data_file, vocab_file, max_len):
        self.data_file = data_file
        self.dataset = self.data_read(self.data_file)
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file)
        self.max_len = max_len

    def __getitem__(self, index):
        example = self.dataset[index]
        encoding = self.tokenizer.encode_plus(example['features_content'],
                                              max_length=self.max_len,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt',
                                              )
        self.item = {key: val.squeeze(-2) for key, val in encoding.items() if
                     key not in ['testid', 'features_content', 'labels_index', 'labels_num']}
        self.item['testid'] = example['testid']
        self.item['labels'] = torch.tensor(example['labels_index'])

        return self.item

    def __len__(self):
        return len(self.dataset)

    def data_read(self, input_path):
        """
        读数据
        :param input_path:
        :return: data: [list]  [{'testid': int, 'features_content': str, 'labels_index': [int], 'labels_num': int}]
        """
        data = []
        # count = 0
        for line in open(input_path, 'r', encoding='utf-8'):
            # if count > 2000:
            #     break
            line_json = json.loads(line)
            mid = ''.join(line_json['features_content'])
            line_json['features_content'] = re.sub(u'[a-zA-Z×]', '', mid).replace(' ', '')
            line_json['labels_index'] = self.labels_alignment(line_json['labels_index'], num_labels=148)

            data.append(line_json)
            # count += 1
        return data

    def labels_alignment(self, labels, num_labels):
        new_labels = [0] * num_labels
        for one in labels:
            new_labels[one] = 1

        return new_labels

    def dataset_split(self, dataset, split_ratio=0.7):
        train, valid = random_split(
            dataset,
            lengths=[int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))])
        print(len(train), 'train records loaded.', len(valid), 'valid records loaded.')

        return train, valid

# class MyDataset(Dataset):
#     def __init__(self, data_file, tokenizer, max_len):
#         self.data_file = data_file
#         self.dataset = self.data_read(self.data_file)
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#     def __getitem__(self, index):
#         example = self.dataset[index]
#         encoding = self.tokenizer.encode_plus(example['features_content'],
#                                               # add_special_tokenizerns=True,
#                                               max_length=self.max_len,
#                                               padding='max_length',
#                                               truncation=True,
#                                               return_tensors='pt',
#                                               )
#         self.item = {key: val.squeeze(-2) for key, val in encoding.items() if
#                      key not in ['testid', 'features_content', 'labels_index', 'labels_num']}
#         self.item['testid'] = example['testid']
#         self.item['labels'] = torch.tensor(example['labels_index'])
#
#         return self.item
#
#     def __len__(self):
#         return len(self.dataset)
#
#     @staticmethod
#     def data_read(input_path):
#         """
#         读数据
#         :param input_path:
#         :return: data: [list]  [{'testid': int, 'features_content': str, 'labels_index': [int], 'labels_num': int}]
#         """
#         data = []
#         for line in open(input_path, 'r', encoding='utf-8'):
#             line_json = json.loads(line)
#             line_json['features_content'] = ''.join(line_json['features_content'])
#             # if line_json['labels_num']>1:
#             for one_label in line_json['labels_index']:
#                 one = {
#                     'testid': line_json['testid'],
#                     'features_content': line_json['features_content'],
#                     # 'labels_index': line_json['labels_index'],
#                     'labels_index': one_label,
#                 }
#                 data.append(one)
#
#         return data


class TestDataset(Dataset):
    def __init__(self, tokenizer, test_file, max_len):
        self.tokenizer = tokenizer
        self.mydataset = self.testdata_read(test_file)
        self.max_len = max_len

    def __getitem__(self, index):
        """
        接收一个index，返回其对应的文本和类别
        :param index:
        :return:
        """
        self.example = self.mydataset[index]
        encoding = self.tokenizer.encode_plus(self.example['features_content'],
                                              max_length=self.max_len,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt',
                                              )
        self.item = {key: val.squeeze(-2) for key, val in encoding.items()}
        self.item['testid'] = self.example['testid']

        return self.item

    def __len__(self):
        return len(self.mydataset)

    @staticmethod
    def testdata_read(input_path):
        """
        读数据
        :param input_path:
        :return: data: [list]  [{'testid': int, 'features_content': str, 'labels_index': [int], 'labels_num': int}]
        """
        data = []
        for line in open(input_path, 'r', encoding='utf-8'):
            line_json = json.loads(line)
            # line_json['features_content'] = ''.join(line_json['features_content'])
            mid = ''.join(line_json['features_content'])
            line_json['features_content'] = re.sub(u'[a-zA-Z×]', '', mid).replace(' ', '')

            data.append(line_json)

        return data


def MyDataLoader(dataset, tokenizer, batch_size=8):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,  # 是否打乱
                        drop_last=True,  # 是否对无法整除的最后一个datasize进行丢弃
                        collate_fn=data_collator,
                        num_workers=0)

    return loader


def loader_import(tokenizer, input_path, batch_size, max_len=512, split_prob=0.7):
    _dataset = MyDataset(
        data_file=input_path,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    train_num = int(len(_dataset) * split_prob)
    eval_num = len(_dataset) - train_num
    train_dataset, eval_dataset = random_split(_dataset, [train_num, eval_num])
    train_loader = MyDataLoader(train_dataset, tokenizer=tokenizer, batch_size=batch_size)
    eval_loader = MyDataLoader(eval_dataset, tokenizer=tokenizer, batch_size=batch_size)

    print('dataload complete!')
    return train_dataset, train_loader, eval_dataset, eval_loader


def data_insert(file1, file2, out_file):
    # 临时函数，合并两阶段数据
    # data1 = TestDataset.testdata_read(file1)
    # data2 = TestDataset.testdata_read(file2)
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = f1.readlines()
    f1.close()
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = f2.readlines()
    f2.close()
    mydata = data1 + data2
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in mydata:
            f.write(line)
            # json.dump(line, f, indent=4, ensure_ascii=False)
    f.close()


if __name__ == '__main__':
    data_file = r'data/train_v1(训练集).json'
    data_file2 = r'data/train_v2(训练集).json'
    _out = r'data/train_read.json'
    # data_insert(data_file, data_file2, _out)
    tokenizer = AutoTokenizer.from_pretrained('chinese_roberta_wwm_ext/')
    max_len = 512
    mydataset = MyDataset(data_file, tokenizer, max_len=max_len)
    print(mydataset[2])
    pass
