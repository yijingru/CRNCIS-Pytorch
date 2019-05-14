from seg_dataset import CellDataset
import seg_transforms
import config as cfg
import dec_net
import torch
from seg_net import SEG_NET
from dec_decoder import Detect
from dec_anchors import Anchors
import numpy as np
import cv2
import random


def test(dec_weights, seg_weights):
    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                              seg_transforms.ToTensor()])

    datatype = 'test'

    dsets = CellDataset(root=cfg.root,
                        datatype=datatype,
                        transform=data_transforms)


    #-----------------load detection model -------------------------
    dec_model = dec_net.resnetssd50(pretrained=False, num_classes=cfg.num_classes)
    resume_dict = torch.load(dec_weights)
    resume_dict = {k[7:]: v for k, v in resume_dict.items()}
    dec_model.load_state_dict(resume_dict)
    dec_model.to(device)
    #-----------------load segmentation model -------------------------
    seg_model =  SEG_NET(num_classes=cfg.num_classes)
    if seg_weights:
        seg_model.load_state_dict(torch.load(seg_weights))
    seg_model.to(device)

    dec_model.eval()
    seg_model.eval()

    detector = Detect(num_classes=cfg.num_classes,
                      top_k=cfg.top_k,
                      conf_thresh=cfg.conf_thresh,
                      nms_thresh=cfg.nms_thresh,
                      variance=cfg.variances)

    anchorGen = Anchors(cfg.img_sizes)
    anchors = anchorGen.forward()

    # print(dec_model)
    # print(seg_model)

    for idx_img in range(len(dsets)):
        inputs, gt_boxes, gt_classes, gt_masks = dsets.__getitem__(idx_img)
        ori_img = dsets.load_image(idx_img)
        img_copy = ori_img.copy()
        h,w,c = ori_img.shape

        x = inputs.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            locs, conf, feat_seg = dec_model(x)
            detections = detector(locs, conf, anchors)
            outputs = seg_model(detections, feat_seg)

        mask_patches, mask_dets = outputs

        for idx in range(len(mask_patches)):
            batch_mask_patches = mask_patches[idx]
            batch_mask_dets = mask_dets[idx]
            # For obj
            for idx_obj in range(len(batch_mask_patches)):
                # ori_img = img_copy
                dets = batch_mask_dets[idx_obj].data.cpu().numpy()
                box = dets[0:4]
                conf = dets[4]
                if conf < cfg.conf_thresh:
                    continue
                class_obj = dets[5]

                mask_patch = batch_mask_patches[idx_obj].data.cpu().numpy()

                [y1, x1, y2, x2] = box
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), h - 1)
                x2 = np.minimum(np.int32(np.round(x2)), w - 1)

                mask = np.zeros((h, w), dtype=np.float32)
                mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))

                mask_patch = np.where(mask_patch >= cfg.seg_thresh, 1, 0)

                mask[y1:y2, x1:x2] = mask_patch
                color = np.random.rand(3)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                mskd = ori_img * mask

                clmsk = np.ones(mask.shape) * mask
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                ori_img = ori_img + 0.7 * clmsk - 0.7 * mskd
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 0, 0), 2, 1)
                cv2.putText(ori_img, dsets.classes[int(class_obj)] + "%.2f" % conf, (x1, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255))
        cv2.imwrite("{}.jpg".format(dsets.img_ids[idx_img][1]), np.uint8(ori_img))
        cv2.imshow('img', np.uint8(ori_img))
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

    cv2.destroyAllWindows()
    exit(1)





if __name__ == '__main__':
    dec_weights = 'dec_weights/end_model.pth'
    seg_weights = 'end_model.pth'
    test(dec_weights, seg_weights)