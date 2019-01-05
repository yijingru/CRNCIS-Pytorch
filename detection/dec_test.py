import cv2
import dec_net
import dec_transform
from dec_dataset import CellDataset
import config as cfg
import torch
from dec_decoder import Detect
from dec_anchors import Anchors


def test(resume=None):
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

    cv2.namedWindow('img')
    for i in range(num_imgs):
        img, bboxes, labels = dsets.__getitem__(i)
        ori_img = dsets.load_image(i)
        x = img.unsqueeze(0)
        x = x.to(device)

        bboxes, conf = model(x)

        detections = detector(bboxes, conf, anchors)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:,0].gt(0.).expand(5,dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1,5)
            if dets.dim()==0:
                continue
            if j:
                boxes = dets[:,1:]
                scores = dets[:,0].cpu().numpy()
                for box, score in zip(boxes,scores):
                    y1,x1,y2,x2 = box
                    y1 = int(y1)
                    x1 = int(x1)
                    y2 = int(y2)
                    x2 = int(x2)

                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                    cv2.putText(ori_img,
                                dsets.classes[int(j)] + "%.2f" % score,
                                (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255))
        cv2.imshow('img', ori_img)
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
    exit()


if __name__ == '__main__':
    resume = "end_model.pth"
    test(resume)