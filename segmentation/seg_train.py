import torch
import numpy as np
import os
import dec_net
import seg_transforms
import config as cfg
from seg_dataset import CellDataset
import viewDatasets
import seg_collater
from dec_decoder import Detect
from dec_anchors import Anchors
from seg_net import SEG_NET
import torch.optim as optim
from torch.optim import lr_scheduler
from seg_loss import SEG_loss
import seg_eval

def train(dec_weights, seg_weights=None):

    if not os.path.exists('weights'):
        os.mkdir('weights')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    ##--------------------------------------------------------------

    data_transforms = {
        'train': seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                         seg_transforms.PhotometricDistort(),
                                         seg_transforms.Expand(max_scale=2, mean=(0.485, 0.456, 0.406)),
                                         seg_transforms.RandomSampleCrop(),
                                         seg_transforms.RandomMirror_w(),
                                         seg_transforms.RandomMirror_h(),
                                         seg_transforms.Resize(cfg.img_sizes),
                                         seg_transforms.ToTensor()]),

        'val': seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                       seg_transforms.ToTensor()])
    }

    dsets = {x: CellDataset(root=cfg.root,
                            datatype=x,
                            transform=data_transforms[x])
             for x in ['train', 'val']}

    ## Visualization of input data and GT ###################
    # viewDatasets.view_dataset(dsets['train'])
    #########################################################


    dataloader_train = torch.utils.data.DataLoader(dsets['train'],
                                                   batch_size=cfg.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   collate_fn=seg_collater.collater,
                                                   pin_memory=True)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, seg_model.parameters()), lr=cfg.init_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
    criterion = SEG_loss()

    detector = Detect(num_classes=cfg.num_classes,
                      top_k=cfg.top_k,
                      conf_thresh=cfg.conf_thresh,
                      nms_thresh=cfg.nms_thresh,
                      variance=cfg.variances)

    anchorGen = Anchors(cfg.img_sizes)
    anchors = anchorGen.forward()

    #-------------------------------------------------------------------
    dec_model.eval()        # detector set to 'evaluation' mode
    for param in dec_model.parameters():
        param.requires_grad = False
    #-------------------------------------------------------------------
    train_loss_dict = []
    ap05_dict = []
    ap07_dict = []
    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))
        print('-' * 10)

        for phase in [ 'train','val']:
            if phase == 'train':
                scheduler.step()
                seg_model.train()
                running_loss = 0.0
                for inputs, bboxes, labels, masks in dataloader_train:
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        locs, conf, feat_seg = dec_model(inputs)
                        detections = detector(locs, conf, anchors)
                        # viewDatasets.view_detections(inputs, detections)

                    optimizer.zero_grad()
                    with torch.enable_grad():
                        outputs = seg_model(detections, feat_seg)
                        loss = criterion(outputs, bboxes, labels, masks)
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dsets[phase])

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                train_loss_dict.append(epoch_loss)
                np.savetxt('train_loss.txt', train_loss_dict, fmt='%.6f')
                if epoch % 5 == 0:
                    torch.save(seg_model.state_dict(),
                               os.path.join('weights', '{:d}_{:.4f}_model.pth'.format(epoch, epoch_loss)))
                torch.save(seg_model.state_dict(), os.path.join('weights', 'end_model.pth'))

            else:
                seg_model.eval()   # Set model to evaluate mode
                with torch.no_grad():
                    ap05, ap07 = seg_eval.eval(dec_model=dec_model,
                                               seg_model=seg_model,
                                               dsets=dsets[phase],
                                               device=device,
                                               detector=detector,
                                               anchors=anchors)
                    ap05_dict.append(ap05)
                    np.savetxt('ap_05.txt', ap05_dict, fmt='%.6f')
                    ap07_dict.append(ap07)
                    np.savetxt('ap_07.txt', ap07_dict, fmt='%.6f')

if __name__ == '__main__':
    dec_weights = 'dec_weights/end_model.pth'
    seg_weights = None
    train(dec_weights, seg_weights)
