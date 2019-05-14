import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContextRefinementModule(nn.Module):
    def __init__(self, c):
        super(ContextRefinementModule,self).__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(output_size=(1,1),return_indices=False)
        self.avgpooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc1 = nn.Linear(in_features=c, out_features=c//8)  #squeeze
        self.fc2 = nn.Linear(in_features=c//8, out_features=c)

    def forward(self, x):
        maxpool = self.maxpooling(x).squeeze(-1).squeeze(-1)
        avgpool = self.avgpooling(x).squeeze(-1).squeeze(-1)

        maxpool_fc = self.fc1(maxpool)
        avgpool_fc = self.fc1(avgpool)

        maxpool_fc = self.fc2(maxpool_fc)
        avgpool_fc = self.fc2(avgpool_fc)

        pool_add = maxpool_fc+avgpool_fc
        pool_add = F.sigmoid(pool_add)

        out = x *pool_add.unsqueeze(2).unsqueeze(3).expand_as(x)
        return out



class CombinationModule(nn.Module):
    def __init__(self, in_size, out_size, cat_size):
        super(CombinationModule, self).__init__()
        self.up =  nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, stride=1),
                                 nn.ReLU(inplace=True))
        self.cat_conv =  nn.Sequential(nn.Conv2d(cat_size, out_size, kernel_size=1, stride=1),
                                       nn.ReLU(inplace=True))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(F.upsample_bilinear(inputs2,inputs1.size()[2:]))
        return self.cat_conv(torch.cat((inputs1, outputs2), 1))


def make_skip_layers():
    layers = []
    layers += [CombinationModule(64, 64, 128)]
    layers += [CombinationModule(256, 64, 128)]
    layers += [CombinationModule(512, 256, 512)]
    layers += [CombinationModule(1024, 512, 1024)]
    return layers

def make_final_refine():
    layers = []
    conv2d1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    conv2d2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    conv2d3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    layers += [conv2d1, nn.ReLU(inplace=True), 
               conv2d2, nn.ReLU(inplace=True),
               conv2d3]
    return nn.Sequential(*layers)

def make_c0_conv():
    layers = []
    conv2d1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    conv2d2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    layers += [conv2d1, nn.ReLU(inplace=True), conv2d2, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SEG_NET(nn.Module):
    def __init__(self, num_classes):
        super(SEG_NET,self).__init__()
        self.num_classes = num_classes
        self.skip_combine = nn.ModuleList(make_skip_layers())
        self.final_refine = make_final_refine()
        self.c0_conv = make_c0_conv()
        self.crm_layer_512 = ContextRefinementModule(512)
        self.crm_layer_1024 = ContextRefinementModule(1024)

    def get_patches(self, box, feat):
        y1, x1, y2, x2 = box
        _, h, w = feat.size()
        y1 = np.maximum(0, np.int32(np.round(y1 * h)))
        x1 = np.maximum(0, np.int32(np.round(x1 * w)))
        y2 = np.minimum(np.int32(np.round(y2 * h)), h - 1)
        x2 = np.minimum(np.int32(np.round(x2 * w)), w - 1)
        if y2<y1 or x2<x1 or y2-y1<2 or x2-x1<2:
            return None
        else:
            return (feat[:, y1:y2, x1:x2].unsqueeze(0))


    def mask_forward(self, i_x):
        pre = None
        for i in range(len(i_x)-1, -1, -1):
            if pre is None:
                pre = i_x[i]
            else:
                pre = self.skip_combine[i](i_x[i], pre)
            if pre.shape[1] == 512:
                pre = self.crm_layer_512(pre)
            if pre.shape[1] == 1024:
                pre = self.crm_layer_1024(pre)
        x = self.final_refine(pre)
        x = F.sigmoid(x)
        x = torch.squeeze(x, dim=0)
        x = torch.squeeze(x, dim=0)
        return x


    def forward(self, detections, feat_seg):
        feat_seg[0] = self.c0_conv(feat_seg[0])

        # create container for output at each batch size
        mask_patches = [[] for i in range(detections.size(0))]
        mask_dets = [[] for i in range(detections.size(0))]

        # iterate batch
        for i in range(detections.size(0)):
            # iterate class
            for j in range(1, detections.size(1)):
                dects = detections[i, j, :]
                mask = dects[:, 0].gt(0.).expand(5, dects.size(0)).t()
                dects = torch.masked_select(dects, mask).view(-1, 5)
                if dects.shape[0] == 0:
                    continue
                if j:
                    for box, score in zip(dects[:, 1:], dects[:, 0]):
                        i_x = []
                        y1, x1, y2, x2 = box
                        h,w = feat_seg[0].shape[2:]
                        for i_feat in range(len(feat_seg)):
                            x = self.get_patches([y1/h,x1/w,y2/h,x2/w], feat_seg[i_feat][i, :, :, :])
                            if x is None:
                                break
                            else:
                                i_x.append(x)
                        # ~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        if len(i_x):
                            x = self.mask_forward(i_x)  # up pooled mask patch
                            mask_patches[i].append(x)
                            mask_dets[i].append(torch.Tensor(np.append(box,[score,j])))

        output = (mask_patches, mask_dets)

        return output
