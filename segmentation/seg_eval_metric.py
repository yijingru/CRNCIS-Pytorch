#################################################################################
# 2018-9-24
# Jingru Yi
# Evaluate using dice object-level metric
# pytorch 0.4.1
#################################################################################
import numpy as np
from scipy.spatial.distance import directed_hausdorff
def Precision(TP, FP):
    return float(TP)/float(TP+FP)

def Recall(TP, FN):
    return float(TP)/float(TP+FN)

def F1_score(TP, FP, FN):
    # Segmentation metric
    precision = Precision(TP, FP)
    recall = Recall(TP, FN)
    if precision+recall<1e-5:
        return 0.
    else:
        return 2*precision*recall/(precision+recall)

def Dice_Segmentation(G, S):
    # G: Groundtruth
    # S: Segmentation
    inter = np.logical_and(G,S).sum()
    union = G.sum() + S.sum()
    return 2*float(inter)/float(union)

def mask_iou(G, S):
    inter = np.logical_and(G, S).sum()
    union = G.sum() + S.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)

def Hausdorff_Dist(G, S):
    return directed_hausdorff(G,S)[0]


def calculate_TP_FP_FN(all_p_boxes, all_p_masks, all_t_boxes, all_t_masks):
    TP = 0
    FP = 0
    gt_label = np.zeros(all_t_boxes.shape[0], np.int8)
    # iterate over predicted instances
    for j in range(all_p_boxes.shape[0]):
        pred_mask = all_p_masks[j]

        gt_overlap = -1000
        gt_overlap_idx = -1
        # iterate over ground truth instance
        for i in range(all_t_boxes.shape[0]):
            gt_mask = all_t_masks[i]
            ov = mask_iou(gt_mask, pred_mask)
            if ov > gt_overlap:
                gt_overlap = ov
                gt_overlap_idx  = i

        if gt_overlap >= 0.5:
            if gt_label[gt_overlap_idx]:
                # repetitive detection
                FP += 1
            else:
                TP += 1
                gt_label[gt_overlap_idx]= 1
        else:
            FP += 1
    FN = all_t_boxes.shape[0] - gt_label.sum()
    return TP, FP, FN


def Metric_A_B(masks_A, masks_B):

    area_a = np.asarray([masks_A[i,:,:].sum() for i in range(masks_A.shape[0])], np.float32)
    area_b = np.asarray([masks_B[i,:,:].sum() for i in range(masks_B.shape[0])], np.float32)

    dice_a_b = np.zeros(shape=(masks_A.shape[0], masks_B.shape[0]), dtype=np.float32)
    ious_a_b = np.zeros(shape=(masks_A.shape[0], masks_B.shape[0]), dtype=np.float32)
    dist_a_b = np.zeros(shape=(masks_A.shape[0], masks_B.shape[0]), dtype=np.float32)

    for i in range(masks_A.shape[0]):
        for j in range(masks_B.shape[0]):
            dice_a_b[i,j] = Dice_Segmentation(masks_A[i], masks_B[j])
            ious_a_b[i,j] = mask_iou(masks_A[i], masks_B[j])
            dist_a_b[i,j] = Hausdorff_Dist(masks_A[i], masks_B[j])


    out_dice_a = np.sum(np.max(dice_a_b, axis=1)*area_a)/np.sum(area_a)
    out_ious_a = np.sum(np.max(ious_a_b, axis=1)*area_a)/np.sum(area_a)
    out_dist_a = np.sum(np.min(dist_a_b, axis=1)*area_a)/np.sum(area_a)


    out_dice_b = np.sum(np.max(dice_a_b, axis=0)*area_b)/np.sum(area_b)
    out_ious_b = np.sum(np.max(ious_a_b, axis=0)*area_b)/np.sum(area_b)
    out_dist_b = np.sum(np.min(dist_a_b, axis=0)*area_b)/np.sum(area_b)

    obj_level_dice = (out_dice_a + out_dice_b)/2
    obj_level_iou  = (out_ious_a + out_ious_b)/2
    obj_level_dist  = (out_dist_a + out_dist_b)/2

    return obj_level_dice, obj_level_iou, obj_level_dist


def Dice_Object_level(all_p_boxes, all_p_masks, all_t_boxes, all_t_masks):
    """
    For every image
    TP: segment area that shares more than 50% areas with the ground truth
    FP: segment area that shares less than 50% areas with the ground truth
    FN: ground truth without corresponding prediction
    all_p_boxes: list of [y1,x1,y2,x2,score], 640x512
    all_t_boxes: list of [y1,x1,y2,x2],       640x512
    """
    # step1: relate the predictions and ground truth
    # 1. align the predictions and ground truth
    all_p_boxes = np.asarray(all_p_boxes, np.float32)    #(4, 5)
    all_p_masks = np.asarray(all_p_masks, np.float32)    #(4, 512, 640)
    all_t_boxes = np.asarray(all_t_boxes, np.float32)    #(4, 4)
    all_t_masks = np.asarray(all_t_masks, np.float32)    #(4, 512, 640)
    # 2. sort the predictions according to their scores
    seg_scores = all_p_boxes[:, -1]
    keep_inds = np.argsort(-seg_scores)   # high -> low
    all_p_boxes = all_p_boxes[keep_inds, :]
    all_p_masks = all_p_masks[keep_inds, :, :]
    # 3. calculate TP, FP, FN
    TP, FP, FN = calculate_TP_FP_FN(all_p_boxes, all_p_masks, all_t_boxes, all_t_masks)
    # 4. F1 score
    F1 = F1_score(TP, FP, FN)
    # 5. object-level-dice-index
    obj_level_dice, obj_level_iou, obj_level_dist = Metric_A_B(all_t_masks, all_p_masks)

    return F1, obj_level_dice, obj_level_iou, obj_level_dist

