import config as cfg
import os
import numpy as np
import pickle
import cv2


def get_voc_results_file_template(image_set, cls):
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(cfg.output_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_rec(annopath, imagename):
    objects = []
    mask_gt = cv2.imread(annopath % (imagename))
    h,w,_ = mask_gt.shape
    cond1 = mask_gt[:, :, 0] != mask_gt[:, :, 1]
    cond2 = mask_gt[:, :, 1] != mask_gt[:, :, 2]
    cond3 = mask_gt[:, :, 2] != mask_gt[:, :, 0]

    r,c = np.where(np.logical_or(np.logical_or(cond1, cond2), cond3))
    unique_colors = np.unique(mask_gt[r, c, :], axis=0)

    for color in unique_colors:
        obj_struct = {}
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
        obj_struct['name'] = 'cell'
        obj_struct['truncated'] = 0
        obj_struct['difficult'] = 0
        obj_struct['bbox'] = [y1-1, x1-1, y2-1, x2-1]
        objects.append(obj_struct)

    return objects


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=True):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    f.close()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath, imagename)
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
        f.close()
    else:
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)
        f.close()

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    f.close()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.shape[0] > 0:
                # compute overlaps
                # intersection
                iymin = np.maximum(BBGT[:, 0], bb[0])
                ixmin = np.maximum(BBGT[:, 1], bb[1])
                iymax = np.minimum(BBGT[:, 2], bb[2])
                ixmax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def do_python_eval(datatype, output_dir='output', use_07=True):
    annopath = os.path.join(cfg.root, 'mask', '%s.png')
    imgsetpath = os.path.join(cfg.root, '{:s}.txt')

    cachedir = os.path.join(cfg.output_dir, 'annotations_cache')

    ap_07 = 0.
    ap_05 = 0.

    use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)

    # print("AP@0.5")
    for i, cls in enumerate(cfg.labelmap):
        # print("i:{}, cls:{}".format(i,cls))
        filename = get_voc_results_file_template(datatype, cls)
        rec, prec, ap = voc_eval(filename,
                                 annopath,
                                 imgsetpath.format(datatype),
                                 cls,
                                 cachedir,
                                 ovthresh=0.5,
                                 use_07_metric=use_07_metric)
        # aps += [ap]
        print('AP@0.5 for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        ap_05 = ap

    # print("AP@0.7")
    for i, cls in enumerate(cfg.labelmap):
        # print("i:{}, cls:{}".format(i,cls))
        filename = get_voc_results_file_template(datatype, cls)
        rec, prec, ap = voc_eval(filename,
                                 annopath,
                                 imgsetpath.format(datatype),
                                 cls,
                                 cachedir,
                                 ovthresh=0.7,
                                 use_07_metric=use_07_metric)
        # aps += [ap]
        print('AP@0.7 for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        ap_07 = ap
    return ap_05, ap_07