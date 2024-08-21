from mmengine import Config
cfg = Config.fromfile('../configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py')
dataset_cfg = Config.fromfile('../configs/_base_/datasets/ZihaoDataset_pipeline.py')
cfg.merge_from_dict(dataset_cfg)

# 类别个数
NUM_CLASS = 6
cfg.crop_size = (512, 512)
cfg.model.data_preprocessor.size = cfg.crop_size

# 单卡训练时，需要把 SyncBN 改成 BN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head.num_classes = NUM_CLASS

# 训练 Batch Size
cfg.train_dataloader.batch_size = 4

# 结果保存目录
cfg.work_dir = '../work_dirs/Dataset-DeepLabV3plus'

# 模型保存与日志记录
cfg.train_cfg.max_iters = 20000 # 训练迭代次数
cfg.train_cfg.val_interval = 500 # 评估模型间隔
cfg.default_hooks.logger.interval = 100 # 日志记录间隔
cfg.default_hooks.checkpoint.interval = 2500 # 模型权重保存间隔
cfg.default_hooks.checkpoint.max_keep_ckpts = 1 # 最多保留几个模型权重
cfg.default_hooks.checkpoint.save_best = 'mIoU' # 保留指标最高的模型权重

# 随机数种子
cfg['randomness'] = dict(seed=0)

cfg.dump('../cn-configs/cnDataset_DeepLabV3plus.py')
