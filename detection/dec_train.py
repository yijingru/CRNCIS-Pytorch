from dec_dataset import CellDataset
import dec_transform
import config as cfg
import torch
from view_datasets import view_dataset
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import dec_net
import dec_collater
from dec_loss import LossModule
from dec_eval import eval
from dec_decoder import Detect
import numpy as np
import os


def train(resume=None):

    if not os.path.exists('weights'):
        os.mkdir('weights')

    data_transforms = {
        'train': dec_transform.Compose([dec_transform.ConvertImgFloat(),
                                        dec_transform.PhotometricDistort(),
                                        dec_transform.Expand(max_scale=2, mean = (0.485, 0.456, 0.406)),
                                        dec_transform.RandomSampleCrop(),
                                        dec_transform.RandomMirror_w(),
                                        dec_transform.RandomMirror_h(),
                                        dec_transform.Resize(cfg.img_sizes),
                                        dec_transform.ToTensor()]),
        
        'val': dec_transform.Compose([dec_transform.ConvertImgFloat(),
                                      dec_transform.ToTensor()])
    }

    dsets = {x: CellDataset(root=cfg.root,
                            datatype=x,
                            transform=data_transforms[x])
             for x in ['train', 'val']}


    # view_dataset(dsets['train'])

    dataloader_train = torch.utils.data.DataLoader(dsets['train'],
                                                   batch_size=cfg.batch_size,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   collate_fn=dec_collater.collater,
                                                   pin_memory=True)

    model = dec_net.resnetssd50(pretrained=True, num_classes=cfg.num_classes)

    if resume is not None:
        resume_dict = torch.load(resume)
        resume_dict = {k[7:]: v for k, v in resume_dict.items()}
        model.load_state_dict(resume_dict)

    # data parallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)


    optimizer = optim.SGD(model.parameters(), lr=cfg.init_lr, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)

    criterion = LossModule(img_size=cfg.img_sizes, num_classes=cfg.num_classes, device=device)

    # for validation data
    detector = Detect(num_classes=cfg.num_classes,
                      top_k=cfg.top_k,
                      conf_thresh=cfg.conf_thresh,
                      nms_thresh=cfg.nms_thresh,
                      variance=cfg.variances)

    train_loss_dict = []
    ap05_dict = []
    ap07_dict = []
    for epoch in range(cfg.num_epochs):
        print('Epoch {}/{}'.format(epoch, cfg.num_epochs - 1))
        print('-' * 10)

        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()
                running_loss = 0.0
                for inputs, bboxes, labels in dataloader_train:
                    inputs = inputs.to(device)
                    # labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        loss_locs, loss_conf = criterion(outputs, bboxes, labels)
                        loss = loss_locs + loss_conf
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dsets[phase])

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                train_loss_dict.append(epoch_loss)
                np.savetxt('train_loss.txt', train_loss_dict, fmt='%.6f')
                if epoch % 5 == 0:
                    torch.save(model.state_dict(),
                               os.path.join('weights', '{:d}_{:.4f}_model.pth'.format(epoch, epoch_loss)))
                torch.save(model.state_dict(), os.path.join('weights', 'end_model.pth'))

            else:
                model.eval()   # Set model to evaluate mode
                with torch.no_grad():
                    ap05, ap07 = eval(model=model, dsets=dsets[phase], device=device, detector=detector, datatype=phase)
                    ap05_dict.append(ap05)
                    np.savetxt('ap_05.txt', ap05_dict, fmt='%.6f')
                    ap07_dict.append(ap07)
                    np.savetxt('ap_07.txt', ap07_dict, fmt='%.6f')

if __name__ == '__main__':
    train()