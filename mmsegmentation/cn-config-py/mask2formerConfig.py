# 安装环境
#!pip install 'mmdet>=3.1.0' -i https://pypi.tuna.tsinghua.edu.cn/simple

from mmengine import Config
cfg = Config.fromfile('.。/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py')
dataset_cfg = Config.fromfile('./configs/_base_/datasets/ZihaoDataset_pipeline.py')
cfg.merge_from_dict(dataset_cfg)
#修改config配置文件
# 类别个数
NUM_CLASS = 6
cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.crop_size = (512, 512)
cfg.model.data_preprocessor.size = cfg.crop_size

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = NUM_CLASS
cfg.model.decode_head.loss_cls.class_weight = [1.0] * NUM_CLASS + [0.1]

cfg.train_dataloader.batch_size = 2

cfg.test_dataloader = cfg.val_dataloader

# 结果保存目录
cfg.work_dir = '../work_dirs/Dataset-Mask2Former'

cfg.train_cfg.max_iters = 20000 # 训练迭代次数
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
cfg.dump('cn-Configs/Dataset_Mask2Former.py')
