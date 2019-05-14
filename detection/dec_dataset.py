from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import torch

def load_gt(annopath):
    """
    :return: [y1, x1, y2, x2, cls] in original sizes
    """
    bboxes = []
    labels = []
    mask_gt = cv2.imread(annopath)
    h,w,_ = mask_gt.shape
    cond1 = mask_gt[:, :, 0] != mask_gt[:, :, 1]
    cond2 = mask_gt[:, :, 1] != mask_gt[:, :, 2]
    cond3 = mask_gt[:, :, 2] != mask_gt[:, :, 0]

    r,c = np.where(np.logical_or(np.logical_or(cond1, cond2), cond3))
    unique_colors = np.unique(mask_gt[r, c, :], axis=0)

    for color in unique_colors:
        cond1 = mask_gt[:, :, 0] == color[0]
        cond2 = mask_gt[:, :, 1] == color[1]
        cond3 = mask_gt[:, :, 2] == color[2]
        r,c = np.where(np.logical_and(np.logical_and(cond1, cond2), cond3))
        y1 = np.min(r)
        x1 = np.min(c)
        y2 = np.max(r)
        x2 = np.max(c)
        if (abs(y2-y1)<=1 or abs(x2-x1)<=1):
            continue
        bboxes.append([y1, x1, y2, x2])   # 512 x 640
        labels.append([1])
    return bboxes, labels


class CellDataset(Dataset):
    def __init__(self, root, datatype, transform=None):
        super(CellDataset, self).__init__()
        self.root = root
        self.datatype = datatype
        self.transform = transform
        self._annopath = os.path.join('%s', 'mask', '%s.png')
        self._imgpath = os.path.join('%s', self.datatype, '%s.jpg')
        self.classes = {0: 'background', 1: 'cell'}
        self.img_ids = []
        for line in open(os.path.join(root, self.datatype + '.txt')):
            self.img_ids.append((root, line.strip()))

    def load_image(self, item):
        img_id = self.img_ids[item]
        img = cv2.imread(self._imgpath % img_id)
        return img

    def load_annotation(self, item):
        img_id = self.img_ids[item]
        bboxes, labels = load_gt(self._annopath % img_id)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        return bboxes, labels

    def __getitem__(self, item):
        img = self.load_image(item)
        bboxes, labels = self.load_annotation(item)
        if self.transform is not None:
            img, bboxes, labels = self.transform(img, bboxes, labels)

        return img, bboxes, labels

    def __len__(self):
        return len(self.img_ids)