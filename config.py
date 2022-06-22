class Config:
    def __init__(self):
        self.gpu_ids = '0'
        self.token_length = 768
        # 训练参数
        self.max_len = 1024
        self.epochs = 15
        self.batch_size = 16
        self.dropout_prob = 0.5
        self.train_batch_size = 32
        self.valid_batch_size = 64
        self.predict_batch_size = 1
        self.lr = 2e-4
        self.other_lr = 2e-4
        self.weight_decay = 0.01
        self.other_weight_decay = 0.01
        self.adam_epsilon = 1e-6
        self.warmup_proportion = 0
        self.dropout_prob = 0.5
        self.use_distant_triggers = True
        self.distant_embed_dims = 256
        self.mid_linear_dims = 128
        self.attack_train_mode = ''
        self.log_interval = 20
        self.val_interval = 1
        self.save_interval = 1
        self.val_and_save_interval = 1
        self.max_grad_norm = 20
        self.start_threshold = 0.5
        self.end_threshold = 0.5
        self.swa_start = 1
        self.ratio_of_validation = 0.1
        # CNN
        self.kernel_num = 10
        self.kernel_size = [2, 3, 4, 5, 6,
                            # 7, 8, 9, 10, 11,]
                            7, 8, 9, 10, 11,
                            15, 20, 30, 50, 80,]
        self.CNN_dropout_prob = 0.5
        # K折交叉验证
        self.K = 8
        # 目录
        self.raw_data_dir = 'data/raw_data/out.json'
        self.distant_triggers_path = 'data/raw_data/distant_triggers.json'
        self.problem_data_dir = 'data/problem_data/'
        self.train_data_dir = 'data/train_data.json'
        self.valid_data_dir = 'data/valid_data.json'
        self.test_data_dir = 'data/test_data.json'
        self.bert_dir = 'bert/chinese-roberta-wwm-ext/'
        self.output_dir = 'out/model/'
        self.scheme = 'base'
        self.best_trigger_extractor = ''

        # 类型选择
        self.bert_type = 'roberta-wwm-ext'
        self.task_type = 'train'
        self.model_type = 'trigger'
