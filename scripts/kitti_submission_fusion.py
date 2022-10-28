import sys
sys.path.append('.')

import os
import numpy as np
import argparse
import torch

from utils import normalize_image
from data_readers.kitti import KITTIEval
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mfuse.raft3d_bilaplacian import RAFT3D
from mfuse.fusion import Fusion


# RAFT-3D's depth scale
DEPTH_SCALE = 0.1


def prepare_images_and_depths(image0, image1, image2, depth0, depth1, depth2, disp1):
    """ padding, normalization, and scaling """

    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image0 = F.pad(image0, [0,pad_w,0,pad_h], mode='replicate')
    image1 = F.pad(image1, [0,pad_w,0,pad_h], mode='replicate')
    image2 = F.pad(image2, [0,pad_w,0,pad_h], mode='replicate')
    depth0 = F.pad(depth0[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth1 = F.pad(depth1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]
    disp1 = F.pad(disp1[:,None], [0,pad_w,0,pad_h], mode='replicate')[:,0]

    depth0 = depth0.float()
    depth1 = depth1.float()
    depth2 = depth2.float()
    image0 = normalize_image(image0.float())
    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    disp1 = disp1.float()

    return image0, image1, image2, depth0, depth1, depth2, disp1, (pad_w, pad_h)


@torch.no_grad()
def make_kitti_submission(model, disp_path="disp_ganet"):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 1, 'drop_last': False}
    test_loader = DataLoader(KITTIEval(disp_path=disp_path), **loader_args)

    for i_batch, data_blob in enumerate(test_loader):
        image0, image1, image2, disp0, disp1, disp2, intrinsics = [item.cuda() for item in data_blob]

        depth0 = DEPTH_SCALE * (intrinsics[0,0] / disp0)
        depth1 = DEPTH_SCALE * (intrinsics[0,0] / disp1)
        depth2 = DEPTH_SCALE * (intrinsics[0,0] / disp2)

        ht, wd = image1.shape[2:]
        image0, image1, image2, depth0, depth1, depth2, disp1, _ = prepare_images_and_depths(image0, image1, image2, depth0, depth1, depth2, disp1)

        flow_pred, _ = model(image0, image1, image2, depth0, depth1, depth2, intrinsics, disp1, iters=16)

        disp1 = disp1[0,:ht,:wd].cpu().numpy()
        dispch_pred = flow_pred[0, 2, :ht, :wd]
        dispch_pred = dispch_pred.cpu().numpy()
        disp2_pred = disp1 + dispch_pred

        flow_pred = flow_pred[0, :2, :ht, :wd].permute(1,2,0).cpu().numpy()

        disp2_pred = np.maximum(disp2_pred, 0.1)

        KITTIEval.write_prediction(i_batch, disp1, disp2_pred, flow_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path the model weights')

    parser.add_argument('--disp_path', default="disp_lea")

    parser.add_argument('--inputs', default="d1,dch,fl,corrCost,dispRes,embVec")

    parser.add_argument('--fusion_module', default="MFUSE")
    parser.add_argument('--fusion_join', default="add")
    parser.add_argument('--fusion_inbetween', default="conv")

    args = parser.parse_args()


    r3d_model = torch.nn.DataParallel(RAFT3D(args))
    model = Fusion(r3d_model, None, inputs=args.inputs, fusion_module=args.fusion_module,
            fusion_join=args.fusion_join, fusion_inbetween=args.fusion_inbetween)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    if not os.path.isdir('kitti_submission'):
        os.mkdir('kitti_submission')
        os.mkdir('kitti_submission/disp_0')
        os.mkdir('kitti_submission/disp_1')
        os.mkdir('kitti_submission/flow')

    make_kitti_submission(model, disp_path=args.disp_path)
