import sys
sys.path.append('.')

from tqdm import tqdm
import numpy as np
import argparse
import torch

from utils import normalize_image
from data_readers.kitti import KITTIEval
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
from data_readers import flow_IO
from mfuse.fusion import Fusion
from mfuse.raft3d_bilaplacian import RAFT3D
from data_readers import flow_errors


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
def test_kitti(model, disp_path="disp_ganet", split=None, additional_errors=False, area=None):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'drop_last': False}
    train_dataset = KITTIEval(do_augment=False, mode="training", disp_path=disp_path, split=split)
    train_loader = DataLoader(train_dataset, **loader_args)

    errors = np.zeros(8)

    for i_batch, test_data_blob in enumerate(tqdm(train_loader)):
        image0, image1, image2, disp0, disp1, disp2, intrinsics, seq_name = test_data_blob
        image0, image1, image2, disp0, disp1, disp2, intrinsics = [data_item.cuda() for data_item in [image0, image1, image2, disp0, disp1, disp2, intrinsics]]

        depth0 = DEPTH_SCALE * (intrinsics[0,0] / disp0)
        depth1 = DEPTH_SCALE * (intrinsics[0,0] / disp1)
        depth2 = DEPTH_SCALE * (intrinsics[0,0] / disp2)

        ht, wd = image1.shape[2:]
        # pad and normalize images
        image0, image1, image2, depth0, depth1, depth2, disp1, _ = prepare_images_and_depths(image0, image1, image2, depth0, depth1, depth2, disp1)

        # run the model
        flow_pred, _ = model(image0, image1, image2, depth0, depth1, depth2, intrinsics, disp1, iters=16)

        disp1 = disp1[0,:ht,:wd].cpu().numpy()
        dispch_pred = flow_pred[0, 2, :ht, :wd]
        dispch_pred = dispch_pred.cpu().numpy()
        disp2_pred = disp1 + dispch_pred

        flow_pred = flow_pred[0, :2, :ht, :wd].permute(1,2,0).cpu().numpy()

        disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        disp2_pred = np.pad(disp2_pred, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        flow_pred = np.pad(flow_pred, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        gt_d1 = flow_IO.readDispFile(os.path.join(os.getenv("DATASETS"), "kitti15", "training", f"disp_occ_0/{seq_name[0]}_10.png"))
        gt_d2 = flow_IO.readDispFile(os.path.join(os.getenv("DATASETS"), "kitti15", "training", f"disp_occ_1/{seq_name[0]}_10.png"))
        gt_fl = flow_IO.readFlowFile(os.path.join(os.getenv("DATASETS"), "kitti15", "training", f"flow_occ/{seq_name[0]}_10.png"))

        if area == "bg" or area == "fg":
            kitti_obj_map = flow_IO.readKITTIObjMap(os.path.join(os.getenv("DATASETS"), "kitti15", "training", f"obj_map/{seq_name[0]}_10.png"))
            if area == "fg":
                kitti_obj_map = ~kitti_obj_map

            gt_d1[kitti_obj_map] = np.nan
            gt_d2[kitti_obj_map] = np.nan
            gt_fl[kitti_obj_map, :] = np.nan

        errors += np.asarray(flow_errors.compute_SF(disp1, disp2_pred, flow_pred, gt_d1, gt_d2, gt_fl))

    errors = 100 * errors[::2] / errors[1::2]
    errors_dict = dict(zip(["D1", "D2", "Fl", "SF"], errors))

    return errors_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path the model weights')

    parser.add_argument('--disp_path', default="disp_lea")
    parser.add_argument('--split', default=None)

    parser.add_argument('--inputs', default="d1,dch,fl,corrCost,dispRes,embVec")

    parser.add_argument('--fusion_module', default="MFUSE")
    parser.add_argument('--fusion_join', default="add")
    parser.add_argument('--fusion_inbetween', default="conv")

    parser.add_argument('--area', default=None)


    args = parser.parse_args()

    r3d_model = torch.nn.DataParallel(RAFT3D(args))

    model = Fusion(r3d_model, inputs=args.inputs, fusion_module=args.fusion_module,
        fusion_join=args.fusion_join, fusion_inbetween=args.fusion_inbetween)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))

    print(count_parameters(model))

    model.cuda()
    model.eval()

    result = test_kitti(model, disp_path=args.disp_path, split=args.split, area=args.area)
    print(result)
