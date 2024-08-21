import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2

# 模型 config 配置文件
config_file = '../mmsegmentation/cn-Configs/Dataset_UNet.py'

# 模型 checkpoint 权重文件
checkpoint_file = '../checkpoint/cn_UNet.pth'

# device = 'cpu'
device = 'cuda:0'

model = init_model(config_file, checkpoint_file, device=device)

# 载入测试集图像，或新图像
img_path = '../mmsegmentation/Watermelon87_Semantic_Seg_Mask/img_dir/val/01bd15599c606aa801201794e1fa30.jpg'
# img_path = 'Watermelon87_Semantic_Seg_Mask/img_dir/val/la_wm_img01.jpg'
# img_path = 'data/watermelon_test1.jpg'

img_bgr = cv2.imread(img_path)
plt.figure(figsize=(8, 8))
# plt.imshow(img_bgr[:,:,::-1])
plt.show()

# 语义分割预测
result = inference_model(model, img_bgr)
# result.keys()
# ['seg_logits', 'pred_sem_seg']
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
# pred_mask.shape
# (1280, 1280)
np.unique(pred_mask)
# array([0, 1, 2, 3, 4, 5])

# 语义分割预测结果-定性
plt.figure(figsize=(8, 8))
plt.imshow(pred_mask)
plt.savefig('../outputs/U1-0.jpg')
plt.show()


# 语义分割预测结果-定量
# result.seg_logits.data.shape

# 可视化语义分割预测结果-方法一
# 显示语义分割结果
plt.figure(figsize=(10, 8))
plt.imshow(img_bgr[:,:,::-1])
plt.imshow(pred_mask, alpha=0.55) # alpha 高亮区域透明度，越小越接近原图
plt.axis('off')
plt.savefig('../outputs/U1-1.jpg')
plt.show()

# 可视化语义分割预测结果-方法二（和原图并排显示）
plt.figure(figsize=(14, 8))

plt.subplot(1,2,1)
plt.imshow(img_bgr[:,:,::-1])
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_bgr[:,:,::-1])
plt.imshow(pred_mask, alpha=0.6) # alpha 高亮区域透明度，越小越接近原图
plt.axis('off')
plt.savefig('../outputs/U1-2.jpg')
plt.show()

# 可视化语义分割预测结果-方法三（按配色方案叠加在原图上显示）
# 各类别的配色方案（BGR）
palette = [
    ['background', [127,127,127]],
    ['red', [0,0,200]],
    ['green', [0,200,0]],
    ['white', [144,238,144]],
    ['seed-black', [30,30,30]],
    ['seed-white', [8,189,251]]
]

palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]

opacity = 0.3 # 透明度，越大越接近原图
# 将预测的整数ID，映射为对应类别的颜色
pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
for idx in palette_dict.keys():
    pred_mask_bgr[np.where(pred_mask==idx)] = palette_dict[idx]
pred_mask_bgr = pred_mask_bgr.astype('uint8')

# 将语义分割预测图和原图叠加显示
pred_viz = cv2.addWeighted(img_bgr, opacity, pred_mask_bgr, 1-opacity, 0)

cv2.imwrite('../outputs/U1-3.jpg', pred_viz)

plt.figure(figsize=(8, 8))
plt.imshow(pred_viz[:,:,::-1])
plt.show()

# 有点问题
# # 可视化语义分割预测结果-方法四
# # 按照mmseg/datasets/cnDataset.py里定义的类别颜色可视化
#
# from mmseg.apis import show_result_pyplot
# img_viz = show_result_pyplot(model, img_path, result, opacity=0.8, title='MMSeg', out_file='../outputs/U1-4.jpg')
#
# # opacity控制透明度，越小，越接近原图。
#
# # img_viz.shape
# # (1280, 1280, 3)
# plt.figure(figsize=(14, 8))
# plt.imshow(img_viz)
# plt.savefig('../outputs/U1-4.jpg')
# plt.show()

# 可视化语义分割预测结果-方法五（加图例）
from mmseg.datasets import cnDataset
import numpy as np
import mmcv
from PIL import Image

# 获取类别名和调色板
classes = cnDataset.METAINFO['classes']
palette = cnDataset.METAINFO['palette']
opacity = 0.15 # 透明度，越大越接近原图

# 将分割图按调色板染色
# seg_map = result[0].astype('uint8')
seg_map = pred_mask.astype('uint8')
seg_img = Image.fromarray(seg_map).convert('P')
seg_img.putpalette(np.array(palette, dtype=np.uint8))

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.figure(figsize=(14, 8))
img_plot = ((np.array(seg_img.convert('RGB')))*(1-opacity) + mmcv.imread(img_path)*opacity) / 255
im = plt.imshow(img_plot)


# 创建图例patch列表
patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(len(classes))]
# patches = [mpatches.Patch(color=np.array(color)/255., label=f' {classes[i]}') for i, color in enumerate(palette)]

plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

plt.savefig('../outputs/U1-5.jpg')
plt.show()


