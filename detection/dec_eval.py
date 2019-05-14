import config as cfg
import os
import shutil
import time
import torch
import numpy as np
import pickle
from dec_anchors import Anchors
import python_evaluation
import dec_transform
from dec_anchors import Anchors
import dec_net
from dec_decoder import Detect
from dec_dataset import CellDataset


def write_detection_results(detector, model, dsets, device, output_dir, datatype):
    num_imgs = len(dsets)

    det_file = os.path.join(output_dir, 'detections.pkl')
    all_boxes = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]

    anchorGen = Anchors(cfg.img_sizes)
    anchors = anchorGen.forward()
    evaluate_time = []
    for i in range(num_imgs):
        img, bboxes, labels = dsets.__getitem__(i)

        x = img.unsqueeze(0)
        x = x.to(device)

        begin = time.time()
        bboxes, conf = model(x)
        detect_time = time.time()-begin

        if i:
            evaluate_time.append(detect_time)

        # bboxes: torch.Size([1, 30080, 4])   device(type='cuda', index=0)
        # conf:   torch.Size([1, 30080, 2])   device(type='cuda', index=0)
        detections = detector(bboxes, conf, anchors)

        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.shape[0] == 0:
                continue
            boxes = dets[:, 1:]
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        # print('img-detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_imgs, detect_time))

    print("average time is {:.4f}".format(np.mean(evaluate_time)))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    for cls_ind, cls in enumerate(cfg.labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = python_evaluation.get_voc_results_file_template(datatype, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dsets.img_ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def eval(model, dsets, device, detector, datatype):
    output_dir = cfg.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    write_detection_results(detector=detector,
                            model=model,
                            dsets=dsets,
                            device=device,
                            output_dir=output_dir,
                            datatype=datatype)


    ap05, ap07 = python_evaluation.do_python_eval(datatype=datatype, output_dir=output_dir, use_07=True)
    return ap05, ap07



if __name__ == '__main__':
    resume = "end_model.pth"
    data_transforms = dec_transform.Compose([dec_transform.ConvertImgFloat(),
                                             dec_transform.ToTensor()])

    datatype = 'test'

    dsets = CellDataset(root=cfg.root,
                        datatype=datatype,
                        transform=data_transforms)


    model = dec_net.resnetssd50(pretrained=True, num_classes=cfg.num_classes)
    if resume is not None:
        resume_dict = torch.load(resume)
        resume_dict = {k[7:]: v for k, v in resume_dict.items()}
        model.load_state_dict(resume_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    detector = Detect(num_classes=cfg.num_classes,
                      top_k=cfg.top_k,
                      conf_thresh=cfg.conf_thresh,
                      nms_thresh=cfg.nms_thresh,
                      variance=cfg.variances)

    num_imgs = len(dsets)
    anchorGen = Anchors(cfg.img_sizes)
    anchors = anchorGen.forward()

    eval(model=model, dsets=dsets, device=device, detector=detector, datatype=datatype)