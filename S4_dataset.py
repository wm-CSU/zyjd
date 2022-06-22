"""Data processor for sentence classification.

In data file, each line contains (19+5)*3=72 attributes (XX, XX_sentence, XX_keywords) and other attributes.
The data processor convert each sentence into (sentence, class number) pair,
each sample with 1 sentence and 1 label.

Usage:
    from data import Data
    # For BERT model
    # For training, load train and valid set
    data = Data('model/bert/vocab.txt')
    datasets = data.load_train_and_valid_files(train_file='data/batch_one.xlsx', train_sheet='Sheet1',
                                                train_txt='data/sent_multi_label/',)
    train_set, valid_set_train = datasets
    # For testing, load test set
    data = TestData('model/bert/vocab.txt')
    test_set = data.load_from_txt(os.path.join(test_txt, index + '.txt'))
"""
import re
import os
import json
import torch
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer
from tqdm import tqdm
from utils import read_annotation
from S3_sentence_division import Division


class Data:
    """Data processor for pretrained model for sentence classification.

    Attributes:
        model_type: 'bert'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
    """
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512):
        """Initialize data processor.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
        """
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len

    def data_read(self, filename: str = 'data/train.json'):
        """
        读数据
        :param input_path:
        :return: data: [list]  [{'testid': int, 'features_content': str, 'labels_index': [int], 'labels_num': int}]
        """
        # data = []
        sent_list, label_list = [], []
        for line in open(filename, 'r', encoding='utf-8').readlines():
            line_json = json.loads(line)
            mid = ''.join(line_json['features_content'])
            line_json['features_content'] = re.sub(u'[a-zA-Z×]', '', mid).replace(' ', '')
            line_json['labels_index'] = self.labels_alignment(line_json['labels_index'], num_labels=148)

            # data.append(line_json)

            sent_list.append(self.tokenizer.tokenize(line_json['features_content']))
            label_list.append(line_json['labels_index'])

        dataset = self._convert_sentence_to_bert_dataset(sent_list, label_list)

        return dataset

    def labels_alignment(self, labels, num_labels):
        new_labels = [0] * num_labels
        for one in labels:
            new_labels[one] = 1

        return new_labels

    def load_train_and_valid_files(self, train_file, split_ratio=0.7):
        """Load train files for task.

        Args:
            train_file: files for sentence classification.

        Returns:
            train_set, valid_set_train
            all are torch.utils.data.TensorDataset
        """
        print('Loading train records for train...')
        dataset = self.data_read(train_file)
        train_set, valid_set = random_split(
            dataset,
            lengths=[int(split_ratio * len(dataset)), len(dataset) - int(split_ratio * len(dataset))]
        )
        print(len(train_set), 'train records loaded.', len(valid_set), 'valid records loaded.')

        return train_set, valid_set

    def _convert_sentence_to_bert_dataset(
            self, sent_list, label_list=None):
        """Convert sentence-label to dataset for BERT model.

        Args:
            sent_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []

        Returns:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids)
        """
        all_input_ids, all_input_mask, all_segment_ids = [], [], []
        for i, _ in tqdm(enumerate(sent_list), ncols=80):
            tokens = ['[CLS]'] + sent_list[i] + ['[SEP]']
            # segment_ids = [0] * len(tokens)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
                # segment_ids = segment_ids[:self.max_seq_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_mask += [0] * (self.max_seq_len - len(input_ids))
            input_ids += [0] * (self.max_seq_len - len(input_ids))  # 补齐剩余位置
            segment_ids = [0] * self.max_seq_len

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # test
        return TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids)


class TestData:
    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512, ):
        """Initialize data processor.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
        """
        self.tokenizer = BertTokenizer(vocab_file)
        self.max_seq_len = max_seq_len
        self.datatool = Data(vocab_file, max_seq_len=max_seq_len)

    def load_from_txt(self, filename: str = 'data/test_v1.txt'):
        """Load train file and construct TensorDataset.

        Args:
            file_path: train file
            sheet_name: sheet name
            txt_path:
                If True, txt with 'sentence \t label'
                Otherwise, txt with paragraph
            train:
                If True, train file with 'sentence \t label' in txt_path
                Otherwise, test file without label

        Returns:
            BERT model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (input_ids, input_mask, segment_ids, label)
            Test:
                [torch.utils.data.TensorDataset]
                    each record: (input_ids, input_mask, segment_ids)
        """

        sent_list, test_id = [], []
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line_json = json.loads(line)
                mid = ''.join(line_json['features_content'])
                line_json['features_content'] = re.sub(u'[a-zA-Z×]', '', mid).replace(' ', '')

                sent_list.append(self.tokenizer.tokenize(line_json['features_content']))
                test_id.append(line_json['testid'])

            f.close()

        dataset = self.datatool._convert_sentence_to_bert_dataset(sent_list)

        return dataset, test_id


def test_data():
    """Test for data module."""
    # For BERT model
    # data = Data('model/bert-base-uncased/vocab.txt', max_seq_len=500)
    # train, _ = data.load_train_and_valid_files(
    #     train_file='data/train.json',
    # )
    data = TestData('model/chinese_roberta_wwm_ext/vocab.txt', max_seq_len=500)
    train, _ = data.load_from_txt()


if __name__ == '__main__':
    test_data()
