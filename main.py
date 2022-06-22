"""Test model for multi label classification.

Author: wangmin0918@csu.edu.cn

"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import warnings

warnings.filterwarnings('ignore')  # “error”, “ignore”, “always”, “default”, “module” or “once”

import json
from types import SimpleNamespace

# import fire
import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from S4_dataset import Data
from S5_model import BertForClassification
from S6_train import Trainer
from S8_predict import Prediction
from utils import load_torch_model, get_path, get_label_cooccurance_matrix


def main(config_file='config/bert_config.json',
         need_train: bool = False,
         ReTrain: bool = False):
    """Main method for training.

    Args:
        config_file: in config dir
    """
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))
    get_path(os.path.join(config.model_path, config.experiment_name))
    get_path(config.log_path)

    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len)
    train_set, valid_set_train = data.load_train_and_valid_files(
        train_file=config.train_file, split_ratio=0.8,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('cuda is available!')
    else:
        device = torch.device('cpu')

    # with WeightedRandomSampler
    target = train_set[:][-1]
    class_sample_counts = target.sum(axis=0, keepdims=False, dtype=torch.int)
    class_sample_counts.add_(torch.ones(class_sample_counts.shape, dtype=torch.int))
    weights = 1. / class_sample_counts
    mid = np.array([(weights * t).sum(axis=0, dtype=torch.float) for t in target])
    samples_weights = np.where(mid == 0., 0.001, mid)
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    data_loader = {
        'train': DataLoader(
            train_set, sampler=sampler, batch_size=config.batch_size, shuffle=False),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
    }

    config.labels_co_mat = torch.from_numpy(get_label_cooccurance_matrix(train_set[:][-1])).to(device)

    # 2. Build model
    model = BertForClassification(config)
    model.to(device)

    if need_train:
        # 3. Train
        trainer = Trainer(model=model, data_loader=data_loader,
                          device=device, config=config)
        best_model_state_dict = trainer.train(ReTrain=ReTrain)
        # 4. Save model
        torch.save(best_model_state_dict,
                   os.path.join(config.model_path, 'model.bin'))
    # else:
    #     # 3. Valid
    #     trainer = Trainer(model=model, data_loader=data_loader,
    #                       device=device, config=config)
    #     trainer.model, _, _, _ = trainer.load_last_model(
    #         model=model,
    #         model_path=os.path.join(config.model_path, config.experiment_name, config.model_type + '-last_model.bin'),
    #         optimizer=trainer.optimizer,
    #         multi_gpu=False
    #     )
    #     train_result, valid_result = trainer.evaluate_train_valid()
    #     print(train_result, valid_result)
    #
    # # 5. evaluate
    # model = load_torch_model(
    #     model,
    #     model_path=os.path.join(config.model_path, config.experiment_name, config.model_type + '-best_model.bin'),
    #     multi_gpu=False
    # )
    # from S8_predict import PredictionWithlabels
    # pred_tool = PredictionWithlabels(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
    #                                  max_seq_len=config.max_seq_len,
    #                                  test_file=config.test_file,
    #                                  )
    pred_tool = Prediction(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                           max_seq_len=config.max_seq_len,
                           test_file=config.test_file,
                           )
    pred_tool.evaluator(model=model, device=device,
                        to_file=config.test_to_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/bert_config.json',
        help='model config file')

    parser.add_argument(
        '--local_rank', default=0,
        help='used for distributed parallel')
    args = parser.parse_args()
    main(args.config_file, need_train=True, ReTrain=False)
    # fire.Fire(main)
