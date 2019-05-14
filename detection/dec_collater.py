import torch

def collater(data):
    imgs = []
    bboxes = []
    labels = []
    for sample in data:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs,0), bboxes, labels