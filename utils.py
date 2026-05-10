import os
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

import torchvision.transforms.functional as F
import  random
import cv2
from sklearn.cluster import KMeans
from skimage.color import lab2rgb, rgb2lab

import cv2

'''将labelme格式(.json)转化为yolo格式(.txt)'''

def get_bbox_opencv(polygon):
    """使用OpenCV计算多边形的垂直边界框"""
    # 将多边形转换为OpenCV所需的格式 (N, 1, 2)
    points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(points)

    return (x, y,  x+w, y+h)  # 转换为 (x1, y1, x2, y2) 格式

def deep_delete(obj, _seen=None):
    """
    递归地把 obj 内部所有元素都解引用，最后把 obj 本身也删掉。
    用法：
        deep_delete(my_big_dict)
    之后 my_big_dict 这个变量名也不存在。
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:          # 防止循环引用死循环
        return
    _seen.add(oid)

    # 1. 先处理最常见的容器
    if isinstance(obj, dict):
        for k, v in list(obj.items()):   # list() 避免“字典遍历时修改”
            deep_delete(v, _seen)
            del obj[k]
    elif isinstance(obj, list):
        while obj:                       # 从尾到头 pop 更快
            deep_delete(obj.pop(), _seen)
    elif isinstance(obj, tuple):         # tuple 不可变，只能解引用内部
        for v in obj:
            deep_delete(v, _seen)
    elif isinstance(obj, set):
        while obj:
            deep_delete(obj.pop(), _seen)

    # 2. 特殊对象：numpy / torch
    elif isinstance(obj, np.ndarray):
        # numpy 可能映射到 GPU，先强制解除映射
        if obj.base is not None:
            deep_delete(obj.base, _seen)
    elif isinstance(obj, torch.Tensor):
        if obj.device.type == 'cuda':
            obj.data = torch.empty(0, device='cpu')  # 把显存数据踢回 CPU 空壳
        del obj
        return                                 # Tensor 已删除，直接返回

    # 3. 自定义对象：把 __dict__ 也清掉
    elif hasattr(obj, '__dict__'):
        deep_delete(obj.__dict__, _seen)

    # 4. 最后把自己标记为“已处理”
    del obj

def get_files_from_folder(folder_path,file_type):
    file_paths = []  # 创建一个空列表用于存储文件路径
    # 遍历指定文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            if file.split('.')[-1].upper() == file_type:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)  # 将路径添加到列表中
    return file_paths

def transform_boxes(boxes, img_size, angle=0, flip_v=False,flip_h = False):
    """
    boxes: List[[xmin,ymin,xmax,ymax], ...]  绝对坐标
    img_size: (w,h)
    angle: 逆时针旋转角度（度）
    flip: 是否水平翻转
    crop: (xmin,ymin,xmax,ymax) 裁剪区域，None 表示不裁剪
    return: 同格式的新框
    """
    assert angle in {0, 90, 180, 270}, "angle must be 0, 90, 180 or 270"
    w,h = img_size
    new_boxes = []
    
    
    for x1,y1,x2,y2 in boxes:
        # 1) 水平翻转
        if flip_h:
            x1, x2 = (w - x2, w - x1)
        if flip_v:
            y1, y2=  (h - y2,h - y1)
        # 2) 旋转（绕图像中心）
        if angle == 90:
            new_x1 = y1
            new_y1 = w - x2
            new_x2 = y2
            new_y2 = w - x1
        elif angle == 180:
            new_x1 = w - x2
            new_y1 = h - y2
            new_x2 = w - x1
            new_y2 = h - y1
        elif angle == 270:
            new_x1 = h - y2
            new_y1 = x1
            new_x2 = h - y1
            new_y2 = x2
        elif angle ==0:
            new_x1,new_y1,new_x2,new_y2 = x1,y1,x2,y2
        new_boxes.append([float(new_x1),float(new_y1),float(new_x2),float(new_y2)])
    return new_boxes

def pr_ap_single_img(pred_boxes, conf, gt_boxes, iou_thresh=0.5, eps=1e-7):
    """
    单张图片、单类检测指标
    pred_boxes: (N,4)  float32  [[x1,y1,x2,y2], ...]
    conf      : (N,)   float32
    gt_boxes  : (M,4)  float32
    """
    pred_boxes = np.asarray(pred_boxes, dtype=float)
    conf       = np.asarray(conf, dtype=float)
    gt_boxes   = np.asarray(gt_boxes, dtype=float)

    # 1. 按置信度降序
    order = np.argsort(-conf)
    pred_boxes = pred_boxes[order]
    conf = conf[order]

    # 2. 匹配
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched = np.zeros(len(gt_boxes), dtype=bool)

    for i, pb in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue
        ious = np.atleast_1d(bbox_iou(pb[None, :], gt_boxes).squeeze()) 
        best = ious.argmax()
        if ious[best] >= iou_thresh and not matched[best]:
            tp[i] = 1
            matched[best] = True
        else:
            fp[i] = 1

    # 3. 累计
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_gt = len(gt_boxes)

    prec = tp_cum / (tp_cum + fp_cum + eps)
    rec  = tp_cum / (n_gt + eps)

    # 4. VOC2007 11-point AP
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.any(rec >= t):
            ap += np.max(prec[rec >= t])
    ap /= 11.

    final_p = prec[-1] if len(prec) else 0.
    final_r = rec[-1]  if len(rec)  else 0.
    return final_p, final_r, ap  # 单类 mAP == AP


# ---------- 4. 单张图增强 ----------
def augment_one(img, boxes):

    new_size = img.size
    # 随机参数
    angle = 0#random.choice([0, 90, 180, 270])
    flip_h  = False#random.random() < 0.5
    flip_v = False#random.random() < 0.5
    bright_factor = random.uniform(0.9, 1.1)
    contrast_factor = random.uniform(0.9, 1.1)
    sat_factor = random.uniform(0.9, 1.1)

    # 颜色变换（PyTorch 自带）
    img = F.adjust_brightness(img, bright_factor)
    img = F.adjust_contrast(img, contrast_factor)
    img = F.adjust_saturation(img, sat_factor)
    # 几何变换
    if flip_h:
        img = F.hflip(img)
    if flip_v:
        img = F.vflip(img)
    img = F.rotate(img, angle, expand=False)   # expand=True 防止截断
                           # 旋转后尺寸可能变了
    new_boxes = transform_boxes(boxes, new_size, angle=angle, flip_v=flip_v,flip_h =flip_h)
    return img,new_boxes

# ------------- IoU -------------
def bbox_iou(a, b):
    # a: (N,4)  b: (M,4)  -> (N,M)
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)
def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


COLORS = generate_colors(n_colors=128, n_samples=5000)

def plot_mask(mask, image, color="r", ax=None,alpha=0.5):
    COLOR = np.array(to_rgb(color))*255
    image[mask > 0] = (1 - alpha) * image[mask > 0] + alpha * COLOR
    return image.astype(np.uint8) 

def draw_bbox(
    img,    
    box,
    color="r",
    linestyle="solid",
    text=None,
):

    x1, y1, x2, y2 = box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4,5)

def save_results(img, output_path, results, conf_thresd=0.4):
    # plt.figure(figsize=(12, 8))
    img = np.array(img,dtype=np.uint8)
    ind = results["scores"]>=conf_thresd
    scores= results["scores"][ind]
    if 'masks' in results: 
        masks = results["masks"][ind]
    boxes = results["boxes"][ind]
    nb_objects = len(scores)
    # print(f"found {nb_objects} object(s)")
    # cv2.putText(img, 'org_img', (int(x1), int(x1) ),cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255), 3, cv2.LINE_AA)
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        if 'masks' in results:
            img = plot_mask(masks[i].squeeze(0).cpu(), img, color=color)
        prob = scores[i].item()
        draw_bbox(
            img,
            boxes[i].cpu(),
            text=f"(id={i}, {prob=:.2f})",
            color=np.array(to_rgb(color))*255,
        )
    plt.imsave(output_path, img)


def csv_to_dict(csv_path):
    """
    读取 Global Wheat 2020 格式的 CSV
    返回 dict: 图片名 -> List[[x1,y1,x2,y2], ...]
    """
    df = pd.read_csv(csv_path)
    # 假设列名就叫 image_name 和 boxes（如不是，自己改）
    # 如果无列名，可用 pd.read_csv(csv_path, header=None)
    # 然后 df.columns = ['image_name', 'boxes']
    name_col = df.columns[0]   # 第一列
    box_col  = df.columns[1]   # 第二列

    def parse_one_cell(cell):
        # cell 是字符串 "340 302 408 347;316 270 386 307;..."
        if cell == 'no_box':        # 空框
            return []
        return [list(map(int, box.split())) for box in str(cell).split(';')]

    return {row[name_col]: parse_one_cell(row[box_col]) for _, row in df.iterrows()}

def single_class_pr_ap(pred_boxes, gt_boxes, conf, iou_thresh=0.5, eps=1e-7):
    # 1. 生成 [(img_idx, x1,y1,x2,y2, conf), ...] 并排序
    records = []
    for img_idx, (pb, cf) in enumerate(zip(pred_boxes, conf)):
        for box, c in zip(pb, cf):
            records.append((img_idx, *box, c))
    records = sorted(records, key=lambda x: x[-1], reverse=True)  # 按 conf 降序

    all_gt = [g.astype(float) for g in gt_boxes]
    matched = [np.zeros(len(g), dtype=bool) for g in all_gt]

    tp = np.zeros(len(records))
    fp = np.zeros(len(records))

    # 2. 逐框匹配
    for det_idx, (img_idx, *box, _) in enumerate(records):
        gts = all_gt[img_idx]
        if len(gts) == 0:
            fp[det_idx] = 1
            continue
        ious = bbox_iou(np.array(box)[None, :], gts).flatten()
        best_i = ious.argmax()
        if ious[best_i] >= iou_thresh and not matched[img_idx][best_i]:
            tp[det_idx] = 1
            matched[img_idx][best_i] = True
        else:
            fp[det_idx] = 1

    # 3. PR & AP
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_pos = sum(len(g) for g in all_gt)
    prec = tp_cum / (tp_cum + fp_cum + eps)
    rec = tp_cum / (n_pos + eps)

    ap = 0
    for t in np.linspace(0, 1, 11):
        ap += np.max(prec[rec >= t]) if np.any(rec >= t) else 0
    ap /= 11.

    return prec[-1], rec[-1], ap