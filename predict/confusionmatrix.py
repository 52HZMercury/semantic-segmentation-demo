import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmseg.apis import init_model, inference_model, show_result_pyplot

# 模型 config 配置文件
config_file = '../cn-Configs/Dataset_KNet.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'checkpoint/cn_KNet.pth'

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)
# 载入测试集图像，或新图像
img_path = 'Watermelon87_Semantic_Seg_Mask/img_dir/val/01bd15599c606aa801201794e1fa30.jpg'
# img_path = 'Watermelon87_Semantic_Seg_Mask/img_dir/val/la_wm_img01.jpg'
# img_path = 'data/watermelon_test1.jpg'

img_bgr = cv2.imread(img_path)

result = inference_model(model, img_bgr)
# result.keys()
# ['seg_logits', 'pred_sem_seg']
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# 获取测试集标注
label_path = '../mmsegmentation/Watermelon87_Semantic_Seg_Mask/ann_dir/val/01bd15599c606aa801201794e1fa30.png'

# label_path = 'Watermelon87_Semantic_Seg_Mask/ann_dir/val/la_wm_img01.png'
label = cv2.imread(label_path)
# 三个通道全部一样，只取一个通道作为标注即可。

label_mask = label[:,:,0]
np.unique(label_mask)
# array([0, 1, 2, 3, 4, 5], dtype=uint8)
plt.imshow(label_mask)
plt.show()

from mmseg.datasets import cnDataset
classes = cnDataset.METAINFO['classes']
# 对比测试集标注和语义分割预测结果
# 测试集标注
print(label_mask.shape)
# (1280, 1280)
# 语义分割预测结果
print(pred_mask.shape)
# (1280, 1280)
print(classes)
# ['background', 'red', 'green', 'white', 'seed-black', 'seed-white']
# 真实为 西瓜红瓤，预测为 西瓜红壤
TP = (label_mask == 1) & (pred_mask==1)
plt.imshow(TP)
plt.show()


# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix_model = confusion_matrix(label_mask.flatten(), pred_mask.flatten())
print(print(confusion_matrix_model))

import itertools


def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    """
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    """
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(len(classes))

    plt.title('Confusion Matrix', fontsize=30)
    plt.xlabel('Pred', fontsize=25, c='r')
    plt.ylabel('True', fontsize=25, c='r')
    plt.tick_params(labelsize=16)  # 设置类别文字大小
    plt.xticks(tick_marks, classes, rotation=90)  # 横轴文字旋转
    plt.yticks(tick_marks, classes)

    # 写数字
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=12)

    plt.tight_layout()

    plt.savefig('outputs/K1-混淆矩阵.pdf', dpi=300)  # 保存图像
    plt.show()


cnf_matrix_plotter(confusion_matrix_model, classes, cmap='Blues')
