# encoding:utf-8

import os

import numpy as np

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import warnings, json

warnings.filterwarnings("ignore")
import torch
from S1_1_Longformer_Model import SentenceClassifyModel
from transformers import AutoTokenizer, LongformerForSequenceClassification, LongformerConfig
from S0_MyDataset import TestDataset


def load_network_multi_gpu(network, save_path):
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    network.load_state_dict(new_state_dict)

    return network


def load_network_single_gpu(network, save_path):
    # 单GPU读取
    checkpoint = torch.load(save_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    # self.lr_scheduler.last_epoch = start_epoch
    return network


class Prediction(object):
    def __init__(self, input_path, save_path, tokenizer, max_len=512):
        self.input_path = input_path
        self.save_path = save_path
        self.datasets = TestDataset(tokenizer, test_file=input_path, max_len=max_len)

    def prediction(self,
                   model_network,
                   best_model_path,
                   prior):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_list = []
        for (net_name, network), (one_name, one) in zip(model_network.items(), best_model_path.items()):
            one_name = load_network_multi_gpu(network, one)
            model_list.append(one_name)
            one_name.eval()
            one_name.to(device)
            print('{} import success'.format(net_name))
            # exec(f'model_list.append(model_name)', {'model_list': model_list, 'model_name': one_name})
        print('model import success')

        pred_labels = []
        with torch.no_grad():
            for i, data in enumerate(self.datasets):
                input_ids = torch.Tensor([data['input_ids'].cpu().detach().numpy()]).to(torch.int64).long().to(device)
                # token_type_ids = torch.Tensor([data['token_type_ids'].cpu().detach().numpy()]).to(torch.int64).long().to(
                #     device)
                attention_mask = torch.Tensor([data['attention_mask'].cpu().detach().numpy()]).to(device)
                out_list = []
                for model in model_list[:-1]:
                    one_output = model(input_ids=input_ids, attention_mask=attention_mask,
                                       mode='test',
                                       )
                    out_list.append(one_output)
                _model = model_list[-1]
                one_output = _model(input_ids=input_ids, attention_mask=attention_mask, )
                out_list.append(one_output.logits)
                # mid = sorted(torch.nn.functional.softmax(output, dim=-1).squeeze(0).cpu().numpy())
                output = torch.mean(torch.stack(out_list), dim=0)
                pred = torch.nn.functional.softmax(output, dim=-1).squeeze(0).detach().cpu().numpy()
                # .tolist()
                new_pred = CAN(pred, prior)
                one_result = {
                    'testid': data['testid'],
                    'labels_index': [new_pred.index(i) for i in new_pred if i > 0.15]
                }
                pred_labels.append(one_result)
        print('predict success')
        output_write(self.save_path, pred_labels)

        return pred_labels

    def prediction_single(self,
                          model_network,
                          best_model_path):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = load_network_multi_gpu(list(model_network.values())[0], list(best_model_path.values())[0])
        model.eval()
        model.to(device)
        print('model import success')
        pred_labels = []
        with torch.no_grad():
            for i, data in enumerate(self.datasets):
                input_ids = torch.Tensor([data['input_ids'].cpu().detach().numpy()]).to(torch.int64).long().to(device)
                # token_type_ids = torch.Tensor([data['token_type_ids'].cpu().detach().numpy()]).to(torch.int64).long().to(
                #     device)
                attention_mask = torch.Tensor([data['attention_mask'].cpu().detach().numpy()]).to(device)
                one_output = model(input_ids=input_ids, attention_mask=attention_mask,
                                   # mode='test',
                                   )
                # pred = torch.nn.Sigmoid()(one_output).squeeze(0).detach().cpu().numpy().tolist()
                pred = torch.nn.functional.softmax(one_output, dim=-1).squeeze(0).detach().cpu().numpy().tolist()
                # pred = one_output.squeeze(0).detach().cpu().numpy().tolist()
                one_result = {
                    'testid': data['testid'],
                    'labels_index': [pred.index(score) for score in pred if score > 0.15],
                    # 'labels_index': [pred.index(score) for score in pred if score > -1]
                }
                pred_labels.append(one_result)
        print('predict success')
        output_write(self.save_path, pred_labels)

        return pred_labels


def output_write(file, preds):
    with open(file, "w", encoding="utf-8") as f:
        # f.write(json.dumps(preds))
        for line in preds:
            f.write(str(line) + "\n")

    f.close()


def CAN(y_pred, prior):
    # 评价每个预测结果的不确定性
    k = 3
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)  # top-k归一化
    y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)  # 计算不确定指标
    # 选择阈值，划分高、低置信度两部分
    threshold = 0.9
    y_pred_confident = y_pred[y_pred_uncertainty < threshold]
    # y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
    # 显示两部分各自的准确率  一般而言，高置信度集准确率会远高于低置信度的
    # 逐个修改低置信度样本，并重新评价准确率
    new_pred = []
    # new_pred.extend(y_pred_confident)
    right, alpha, iters = 0, 1, 1
    for sample, number in zip(y_pred, y_pred_uncertainty):
        if number < threshold:
            # for i, y in enumerate(y_pred_unconfident):
            Y = np.concatenate([y_pred_confident, sample[None]], axis=0)
            for j in range(iters):
                Y = Y ** alpha
                Y /= Y.mean(axis=0, keepdims=True)
                Y *= prior[None]
                Y /= Y.sum(axis=1, keepdims=True)
            new_pred.append(Y[-1])
        else:
            new_pred.append(sample)
    return new_pred


def Prior_Statistic(train_data, num_classes):
    # 无效函数
    # 从训练集统计先验分布
    prior = np.zeros(num_classes)
    for d in train_data:
        prior[d[1]] += 1.
    prior /= prior.sum()

    return prior


if __name__ == '__main__':
    # main函数仅作测试
    # y_pred = np.array([[0.9, 0, 0.1, 0],
    #                    [0, 0.8, 0.1, 0.1]])
    # y_true = np.array([0, 1])
    # CAN(y_pred, y_true, 4)
    test = {
        'A': 1,
        'B': 2
    }
    print(list(test.values())[0])
    A = [0.9, 0, 0.4, 0]
    print(A[None])
