import cv2
import numpy as np
import random
import torch

def view_dataset(dset):
    """
    :param dset:
        img:    torch.Size([3, 512, 640])
        bboxes: torch.Size([12, 4])
        labels: torch.Size([12, 1])
        masks:  torch.Size([12, 510, 621])
    :return:
    """
    cv2.namedWindow('img')
    for idx in range(len(dset)):
        img, bboxes, labels, masks = dset.__getitem__(idx)
        img = img.numpy().transpose(1,2,0)
        bboxes = bboxes.numpy()
        labels = labels.numpy()
        masks = masks.numpy()
        for i in range(bboxes.shape[0]):
            y1, x1, y2, x2 = bboxes[i,:]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=1)

            # view segmentation
            cur_gt_mask = masks[i, :, :]
            mask = np.zeros(cur_gt_mask.shape, dtype=np.float32)
            mask[cur_gt_mask == 1] = 1.
            color = (random.random(), random.random(), random.random())
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mskd = img * mask
            clmsk = np.ones(mask.shape) * mask
            clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
            clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
            clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
            img = img + 0.8 * clmsk - 0.8 * mskd
            ###########################

        cv2.imshow('img', np.uint8(img))
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()


def view_detections(inputs, detections):
    """
    :param inputs:       torch.Size([2, 3, 512, 640])
    :param detections:   torch.Size([2, 2, 200, 5])
    :return:
    """
    cv2.namedWindow('img')
    for i in range(inputs.shape[0]):
        img = inputs[i,:,:,:].data.cpu().numpy().transpose(1,2,0)
        img = np.uint8(img).copy()
        det = detections[i,1,:,:]

        mask = det[:, 0].gt(0.).expand(5, det.size(0)).t()
        det = torch.masked_select(det, mask).view(-1, 5)
        if det.shape[0] == 0:
            continue
        boxes  = det[:, 1:].cpu().numpy()
        scores = det[:, 0].cpu().numpy()
        for box, score in zip(boxes, scores):
            y1, x1, y2, x2 = box
            y1 = int(y1)
            x1 = int(x1)
            y2 = int(y2)
            x2 = int(x2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
            cv2.putText(img,
                        "%.2f" % score,
                        (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255))

        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k&0xFF==ord('q'):
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
