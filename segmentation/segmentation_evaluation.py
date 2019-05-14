import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)

def voc_eval_sds(all_p_boxes, all_p_masks,all_t_boxes,all_t_masks, ov_thresh):
    # 3. Pre-compute number of total instances to allocate memory
    #print 'step3'
    box_num = 0
    for im_i in range(len(all_p_boxes)):
        box_num += len(all_p_boxes[im_i])

    # 4. Re-organize all the predicted boxes
    #print 'step4'
    size_h, size_w = all_p_masks[0][0].shape
    new_boxes = np.zeros((box_num, 5))
    new_masks = np.zeros((box_num, size_h, size_w))
    new_image = []
    cnt = 0
    for image_ind in range(len(all_p_boxes)):
        boxes = all_p_boxes[image_ind]
        masks = all_p_masks[image_ind]
        num_instance = len(boxes)
        for box_ind in range(num_instance):
            new_boxes[cnt] = boxes[box_ind]
            new_masks[cnt] = masks[box_ind]
            new_image.append(image_ind)
            cnt += 1

    # 5. Rearrange boxes according to their scores
    #print 'step5'
    seg_scores = new_boxes[:, -1]
    keep_inds = np.argsort(-seg_scores)
    new_boxes = new_boxes[keep_inds, :]
    new_masks = new_masks[keep_inds, :, :]
    num_pred = new_boxes.shape[0]

    # 6. Calculate t/f positive
    #print 'step6'
    fp = np.zeros((num_pred, 1))
    tp = np.zeros((num_pred, 1))
    temp_overlaps = []
    all_t_flags = [[] for _ in range(len(all_p_boxes))]
    num_pos = 0
    for i in range(len(all_t_boxes)):
        for each in range(len(all_t_boxes[i])):
            all_t_flags[i].append(0)
            num_pos+=1
    for i in range(num_pred):
        #pred_box = np.round(new_boxes[i, :4]).astype(int)
        pred_mask = new_masks[i]
        image_index = new_image[keep_inds[i]]

        #gt_boxes = all_t_boxes[image_index]
        gt_masks = all_t_masks[image_index]
        gt_dict_list = all_t_flags[image_index]

        # calculate max region overlap
        cur_overlap = -1000
        cur_overlap_ind = -1
        for ind2 in range(len(gt_masks)):
            if gt_dict_list[ind2] == 0:
                gt_mask = gt_masks[ind2]
                ov = mask_iou(gt_mask, pred_mask)
                if ov > cur_overlap:
                    cur_overlap = ov
                    cur_overlap_ind = ind2

        if cur_overlap >= ov_thresh:
            if gt_dict_list[cur_overlap_ind]:
                fp[i] = 1
            else:
                temp_overlaps.append(cur_overlap)
                tp[i] = 1
                gt_dict_list[cur_overlap_ind]= 1
        else:
            fp[i] = 1

    # 7. Calculate precision
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_pos)
    # avoid divide by zero in case the first matches a difficult gt
    prec = tp / np.maximum(fp+tp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True)
    temp_overlaps = np.mean(temp_overlaps)
    print('temp_overlaps is {}'.format(temp_overlaps))
    return ap


def _py_evaluate_segmentation(all_p_boxes, all_p_masks,all_t_boxes,all_t_masks):
    classes = ['__background__', 'cell']

    # print('~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~')
    # aps = []
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        ap = voc_eval_sds(all_p_boxes[i], all_p_masks[i],all_t_boxes[i],all_t_masks[i], ov_thresh=0.5)
        # aps += [ap]
        print('AP@0.5 for {} = {:.2f}'.format(cls, ap * 100))
    # print('Mean AP@0.5 = {:.2f}'.format(np.mean(aps) * 100))
        ap_05 = ap
    # print('~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~')
    # aps = []
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        ap = voc_eval_sds(all_p_boxes[i], all_p_masks[i],all_t_boxes[i],all_t_masks[i], ov_thresh=0.7)
        # aps += [ap]
        print('AP@0.7 for {} = {:.2f}'.format(cls, ap * 100))
    # print('Mean AP@0.7 = {:.2f}'.format(np.mean(aps) * 100))
        ap_07 = ap
    return ap_05, ap_07


