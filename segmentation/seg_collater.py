import torch

def collater(data):
    imgs = []
    bboxes = []
    labels = []
    masks = []
    for sample in data:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        labels.append(sample[2])
        masks.append(sample[3])
    return torch.stack(imgs,0), bboxes, labels, masks