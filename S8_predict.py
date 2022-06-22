"""Predict.

Author: Min Wang; wangmin0918@csu.edu.cn
"""
import os
import json
import numpy as np
import time
from torch.utils.data import DataLoader
from S4_dataset import Data, TestData
from S7_evaluate import evaluate, subclass_confusion_matrix, compute_metrics, perf_measure


class Prediction:
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test_v1(赛题).txt',
                 ):
        self.dataset_tool = TestData(vocab_file, max_seq_len=max_seq_len)
        self.dataset, self.test_id = self.dataset_tool.load_from_txt(test_file)
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def evaluator(self, model, device,
                  to_file='data/predict.txt'):
        '''
        遍历测试集，逐条数据预测
        :param model:
        :param device:
        :param to_file:
        :return:
        '''
        predictions = evaluate(model, self.test_loader, device)
        self.output_write(to_file, preds=predictions)

        return

    def output_write(self, file, preds):
        with open(file, "w", encoding="utf-8") as f:
            # f.write(json.dumps(preds))
            for testid, line in zip(self.test_id, preds):
                result = [index for index, value in enumerate(line) if value == 1]
                f.write(str({"testid": testid, "labels_index": result}) + "\n")

        f.close()


class PredictionWithlabels:
    def __init__(self, vocab_file='',
                 max_seq_len: int = 512,
                 test_file='data/test.xlsx',
                 ):
        self.test_file = test_file
        self.dataset_tool = Data(vocab_file, max_seq_len=max_seq_len)
        self.dataset, self.test_id = self.dataset_tool.load_train_and_valid_files(
            train_file=test_file, split_ratio=1,
        )
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def evaluator(self, model, device, to_file='data/predict.txt',
                  metrics_save_file: str = 'result/result.txt',):
        '''
        遍历测试集，逐条数据预测
        :param model:
        :param device:
        :param to_file:
        :return:
        '''
        predictions, labels = self.evaluate_for_test(model, self.test_loader, device)

        self.output_write(to_file, preds=predictions)
        self.metrics_output(pred_to_file=to_file, target_list=labels, pred_list=predictions,
                            result_file=metrics_save_file,)

        return

    def evaluate_for_test(self, model, data_loader, device):
        import torch
        from torch import nn
        model.eval()
        answer_list, labels = [], []
        # for batch in tqdm(data_loader, desc='Evaluation:', ascii=True, ncols=80, leave=True, total=len(data_loader)):
        for _, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits, _ = model(*batch[:-1])

            labels.extend(batch[-1].detach().cpu().numpy().tolist())

            predictions = nn.Sigmoid()(logits)
            compute_pred = [[1 if one > 0.80 else 0 for one in row] for row in
                            predictions.detach().cpu().numpy().tolist()]
            answer_list.extend(compute_pred)  # multi-class

        return answer_list, labels

    def output_write(self, file, preds):
        with open(file, "w", encoding="utf-8") as f:
            # f.write(json.dumps(preds))
            for testid, line in enumerate(self.test_id, preds):
                result = [index for index, value in enumerate(line) if value == 1]
                f.write(str({"testid": testid, "labels_index": result}) + "\n")

        f.close()

        return

    def metrics_output(self, pred_to_file, target_list, pred_list,
                       result_file: str = 'result/result.txt',):
        result = compute_metrics(labels=target_list, preds=pred_list)

        with open(result_file, 'a+') as f:
            f.write('\n\n\n' + time.asctime() + '   PredictionWithlabels   ' + self.test_file +
                    '  ->  ' + pred_to_file + '\n')
            f.write(str([str(k) + ': ' + str(format(v, '.6f')) for k, v in result.items() if
                         k != 'subclass_confusion_matrix']) + '\n')

            f.write('subclass confusion_matrix and metrics: \n')
            for k, v in enumerate(result['subclass_confusion_matrix']):
                f.write(str(k) + ': ' + ''.join(np.array2string(v).splitlines()) + '    ')

        f.close()

        return
