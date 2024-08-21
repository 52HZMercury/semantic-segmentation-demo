from mmengine import Config
cfg = Config.fromfile('../configs/fastscnn/fast_scnn_8xb4-160k_cityscapes-512x1024.py')
dataset_cfg = Config.fromfile('../configs/_base_/datasets/ZihaoDataset_pipeline.py')
cfg.merge_from_dict(dataset_cfg)
#修改config配置文件
# 类别个数
NUM_CLASS = 6
cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head[0].norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head[1].norm_cfg = cfg.norm_cfg

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.auxiliary_head[0]['num_classes'] = NUM_CLASS
cfg.model.auxiliary_head[1]['num_classes'] = NUM_CLASS

cfg.train_dataloader.batch_size = 4

cfg.test_dataloader = cfg.val_dataloader

# 结果保存目录
cfg.work_dir = '../work_dirs/Dataset-FastSCNN'

cfg.train_cfg.max_iters = 30000 # 训练迭代次数
cfg.train_cfg.val_interval = 500 # 评估模型间隔
cfg.default_hooks.logger.interval = 100 # 日志记录间隔
cfg.default_hooks.checkpoint.interval = 2500 # 模型权重保存间隔
cfg.default_hooks.checkpoint.max_keep_ckpts = 2 # 最多保留几个模型权重
cfg.default_hooks.checkpoint.save_best = 'mIoU' # 保留指标最高的模型权重

# 随机数种子
cfg['randomness'] = dict(seed=0)
#查看完整config配置文件
# print(cfg.pretty_text)
#保存最终的config配置文件
cfg.dump('cn-Configs/Dataset_FastSCNN.py')
