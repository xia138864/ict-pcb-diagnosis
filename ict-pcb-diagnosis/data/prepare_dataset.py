# -*- coding: utf-8 -*-
"""
DeepPCB 数据预处理：检测标注 -> 缺陷 patch 分类数据集
======================================================
DeepPCB 原始为检测任务（每张 640×640 图含多个缺陷 bounding box）。
本脚本将每个缺陷的 bounding box 区域裁剪为独立 patch，作为分类样本，
标签映射：1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole。

输出：
  data/patches.npz    train/test 的 patch 数组 + 标签 + 每类统计
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from tqdm import tqdm

DEEP_PCB = os.path.join(_ROOT, 'data/DeepPCB/PCBData')
OUT_FILE = os.path.join(_ROOT, 'data/patches.npz')

PATCH_SIZE = 128
CLASSES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']


def parse_annotation(txt_path):
    """解析标注文件，返回 [(x1,y1,x2,y2,type_idx), ...]，type_idx 为 0-based 类别。"""
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 标注为空白分隔：x1 y1 x2 y2 type
            parts = line.replace(',', ' ').split()
            if len(parts) != 5:
                continue
            x1, y1, x2, y2, t = [int(float(p)) for p in parts]
            if t < 1 or t > 6:
                continue
            boxes.append((x1, y1, x2, y2, t - 1))
    return boxes


def load_split(split_file):
    """读取 trainval.txt / test.txt，返回 [(test_img_path, anno_path), ...]"""
    items = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel, anno_rel = line.split()
            # img_rel: 'group20085/20085/20085291.jpg' -> 实际文件 '*_test.jpg'
            img_path = os.path.join(DEEP_PCB, img_rel[:-4] + '_test.jpg')
            anno_path = os.path.join(DEEP_PCB, anno_rel)
            items.append((img_path, anno_path))
    return items


def crop_patches(items):
    """裁剪所有缺陷 patch，返回 patches (N,128,128) uint8, labels (N,) int64。"""
    patches = []
    labels = []
    skipped = 0
    for img_path, anno_path in tqdm(items, desc='裁剪 patch'):
        if not os.path.exists(img_path):
            skipped += 1
            continue
        boxes = parse_annotation(anno_path)
        if not boxes:
            continue
        img = Image.open(img_path).convert('L')
        for x1, y1, x2, y2, t in boxes:
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img.size[0], x2); y2 = min(img.size[1], y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            patch = img.crop((x1, y1, x2, y2)).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
            patches.append(np.asarray(patch, dtype=np.uint8))
            labels.append(t)
    if not patches:
        return np.zeros((0, PATCH_SIZE, PATCH_SIZE), np.uint8), np.zeros((0,), np.int64), skipped
    return np.stack(patches), np.array(labels, dtype=np.int64), skipped


def main():
    train_items = load_split(os.path.join(DEEP_PCB, 'trainval.txt'))
    test_items = load_split(os.path.join(DEEP_PCB, 'test.txt'))
    print(f'train 图数: {len(train_items)}, test 图数: {len(test_items)}')

    train_patches, train_labels, _ = crop_patches(train_items)
    test_patches, test_labels, _ = crop_patches(test_items)

    # 统计
    def dist(labels):
        d = {}
        for i, c in enumerate(CLASSES):
            d[c] = int((labels == i).sum())
        return d

    print('\n=== 数据集统计 ===')
    print(f'train patches: {len(train_labels)}  分布: {dist(train_labels)}')
    print(f'test patches : {len(test_labels)}  分布: {dist(test_labels)}')

    np.savez_compressed(
        OUT_FILE,
        train_patches=train_patches, train_labels=train_labels,
        test_patches=test_patches, test_labels=test_labels,
    )
    print(f'\n已保存: {OUT_FILE}')


if __name__ == '__main__':
    main()
