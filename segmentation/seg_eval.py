import torch
import numpy as np
import time
import os
import shutil
import cv2
import config as cfg
from segmentation_evaluation import _py_evaluate_segmentation
import seg_transforms
from dec_decoder import Detect
from dec_anchors import Anchors
import dec_net
from seg_net import SEG_NET
from seg_dataset import CellDataset

def eval(dec_model, seg_model, dsets, device, detector, anchors):

    num_imgs = len(dsets)
    all_p_boxes = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]
    all_p_masks = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]
    all_t_boxes = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]
    all_t_masks = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]

    cachefile = "output"

    if not os.path.exists(cachefile):
        os.mkdir(cachefile)
    else:
        shutil.rmtree(cachefile)
        os.mkdir(cachefile)

    average_time = []

    for idx in range(len(dsets)):
        inputs, gt_boxes, gt_classes, gt_masks = dsets.__getitem__(idx)

        x = inputs.unsqueeze(0)
        x = x.to(device)

        since = time.time()

        locs, conf, feat_seg = dec_model(x)
        detections = detector(locs, conf, anchors)
        outputs = seg_model(detections, feat_seg)

        # print("detection time: {}".format(time.time()-since))
        if idx:
            average_time.append(time.time()-since)
        mask_patches, mask_dets = outputs

        # For batch
        for i in range(len(mask_patches)):
            batch_mask_patches = mask_patches[i]
            batch_mask_dets = mask_dets[i]
            for idx_obj in range(len(batch_mask_patches)):
                dets = batch_mask_dets[idx_obj].data.cpu().numpy()
                box = dets[0:4]
                conf = dets[4]
                if conf < cfg.conf_thresh:
                    continue
                # class_obj = dets[5]
                mask_patch = batch_mask_patches[idx_obj].data.cpu().numpy()
                [y1, x1, y2, x2] = box
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), cfg.img_sizes[1] - 1)
                x2 = np.minimum(np.int32(np.round(x2)), cfg.img_sizes[0] - 1)
                mask = np.zeros((cfg.img_sizes[1],cfg.img_sizes[0]), dtype=np.float32)
                mask_patch = cv2.resize(mask_patch, (x2 - x1, y2 - y1))
                mask_patch = np.where(mask_patch >= cfg.seg_thresh, 1, 0)
                mask[y1:y2, x1:x2] = mask_patch

                all_p_boxes[1][idx].append([float(y1), float(x1), float(y2), float(x2), conf])
                all_p_masks[1][idx].append(mask)

        for j in range(gt_boxes.shape[0]):
            y1, x1, y2, x2 = gt_boxes[j, :]
            temp_mask = gt_masks[j, :, :].numpy()
            all_t_boxes[1][idx].append([float(y1), float(x1), float(y2), float(x2)])
            all_t_masks[1][idx].append(temp_mask)


    print('average time is {:.6f}'.format(np.mean(np.asarray(average_time))))

    ap05, ap07 = _py_evaluate_segmentation(all_p_boxes, all_p_masks,all_t_boxes,all_t_masks)
    return ap05, ap07


if __name__ == '__main__':
    dec_weights = 'dec_weights/end_model.pth'
    seg_weights = 'end_model.pth'
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

    with torch.no_grad():
        eval(dec_model=dec_model, seg_model=seg_model, dsets=dsets, device=device, detector=detector, anchors=anchors)