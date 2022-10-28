
import numpy as np
import torch
import torch.utils.data as data

import os
import cv2
import random
import csv
import os.path as osp

from . import frame_utils
from .augmentation import SparseAugmentor

from . import flow_IO


SPLITS = {
            "validation_cv0": [102, 161, 21, 80, 104, 158, 172, 64, 184, 85, 71, 35, 181, 47, 90, 146, 82, 98, 2, 99, 75, 79, 43, 128, 173,
                        109, 127, 33, 137, 31, 114, 81, 9, 63, 7, 100, 120, 147, 189, 129, 72, 131, 169, 52, 17, 74, 45, 136, 144, 27],
            "train_cv0": [59, 26, 138, 86, 142, 56, 73, 28, 190, 111, 126, 1, 153, 50, 193, 60, 151, 19, 3, 180, 160, 121, 112, 53,
                        175, 84, 66, 24, 159, 178, 108, 188, 48, 107, 68, 139, 106, 115, 122, 91, 13, 164, 118, 87, 70, 113, 97, 130,
                        185, 187, 140, 61, 15, 94, 165, 6, 162, 125, 23, 148, 76, 89, 5, 29, 77, 22, 116, 62, 46, 14, 155, 110, 105, 88,
                        42, 65, 11, 25, 145, 69, 141, 149, 166, 55, 78, 186, 16, 44, 133, 168, 41, 38, 67, 196, 171, 8, 101, 152, 132, 93,
                        51, 36, 95, 135, 4, 143, 191, 134, 199, 183, 150, 195, 170, 58, 96, 10, 49, 119, 32, 40, 192, 156, 167, 103, 163,
                        20, 54, 176, 34, 194, 30, 154, 123, 92, 124, 39, 197, 174, 177, 198, 83, 37, 117, 179, 18, 57, 157, 12, 0, 182],
            "validation_cv1": [59, 26, 138, 86, 142, 56, 73, 28, 190, 111, 126, 1, 153, 50, 193, 60, 151, 19, 3, 180, 160, 121, 112, 53, 175,
                        84, 66, 24, 159, 178, 108, 188, 48, 107, 68, 139, 106, 115, 122, 91, 13, 164, 118, 87, 70, 113, 97, 130, 185, 187],
            "train_cv1": [102, 161, 21, 80, 104, 158, 172, 64, 184, 85, 71, 35, 181, 47, 90, 146, 82, 98, 2, 99, 75, 79, 43, 128, 173,
                        109, 127, 33, 137, 31, 114, 81, 9, 63, 7, 100, 120, 147, 189, 129, 72, 131, 169, 52, 17, 74, 45, 136, 144, 27, 140,
                        61, 15, 94, 165, 6, 162, 125, 23, 148, 76, 89, 5, 29, 77, 22, 116, 62, 46, 14, 155, 110, 105, 88, 42, 65, 11, 25,
                        145, 69, 141, 149, 166, 55, 78, 186, 16, 44, 133, 168, 41, 38, 67, 196, 171, 8, 101, 152, 132, 93, 51, 36, 95, 135,
                        4, 143, 191, 134, 199, 183, 150, 195, 170, 58, 96, 10, 49, 119, 32, 40, 192, 156, 167, 103, 163, 20, 54, 176, 34,
                        194, 30, 154, 123, 92, 124, 39, 197, 174, 177, 198, 83, 37, 117, 179, 18, 57, 157, 12, 0, 182],
            "validation_cv2": [140, 61, 15, 94, 165, 6, 162, 125, 23, 148, 76, 89, 5, 29, 77, 22, 116, 62, 46, 14, 155, 110, 105, 88, 42, 65,
                        11, 25, 145, 69, 141, 149, 166, 55, 78, 186, 16, 44, 133, 168, 41, 38, 67, 196, 171, 8, 101, 152, 132, 93],
            "train_cv2": [102, 161, 21, 80, 104, 158, 172, 64, 184, 85, 71, 35, 181, 47, 90, 146, 82, 98, 2, 99, 75, 79, 43, 128,
                        173, 109, 127, 33, 137, 31, 114, 81, 9, 63, 7, 100, 120, 147, 189, 129, 72, 131, 169, 52, 17, 74, 45, 136, 144,
                        27, 59, 26, 138, 86, 142, 56, 73, 28, 190, 111, 126, 1, 153, 50, 193, 60, 151, 19, 3, 180, 160, 121, 112, 53,
                        175, 84, 66, 24, 159, 178, 108, 188, 48, 107, 68, 139, 106, 115, 122, 91, 13, 164, 118, 87, 70, 113, 97, 130,
                        185, 187, 51, 36, 95, 135, 4, 143, 191, 134, 199, 183, 150, 195, 170, 58, 96, 10, 49, 119, 32, 40, 192, 156,
                        167, 103, 163, 20, 54, 176, 34, 194, 30, 154, 123, 92, 124, 39, 197, 174, 177, 198, 83, 37, 117, 179, 18, 57,
                        157, 12, 0, 182],
            "validation_cv3": [51, 36, 95, 135, 4, 143, 191, 134, 199, 183, 150, 195, 170, 58, 96, 10, 49, 119, 32, 40, 192, 156, 167, 103,
                        163, 20, 54, 176, 34, 194, 30, 154, 123, 92, 124, 39, 197, 174, 177, 198, 83, 37, 117, 179, 18, 57, 157, 12, 0, 182],
            "train_cv3": [102, 161, 21, 80, 104, 158, 172, 64, 184, 85, 71, 35, 181, 47, 90, 146, 82, 98, 2, 99, 75, 79, 43, 128, 173,
                        109, 127, 33, 137, 31, 114, 81, 9, 63, 7, 100, 120, 147, 189, 129, 72, 131, 169, 52, 17, 74, 45, 136, 144, 27, 59,
                        26, 138, 86, 142, 56, 73, 28, 190, 111, 126, 1, 153, 50, 193, 60, 151, 19, 3, 180, 160, 121, 112, 53, 175, 84, 66,
                        24, 159, 178, 108, 188, 48, 107, 68, 139, 106, 115, 122, 91, 13, 164, 118, 87, 70, 113, 97, 130, 185, 187, 140, 61,
                        15, 94, 165, 6, 162, 125, 23, 148, 76, 89, 5, 29, 77, 22, 116, 62, 46, 14, 155, 110, 105, 88, 42, 65, 11, 25, 145,
                        69, 141, 149, 166, 55, 78, 186, 16, 44, 133, 168, 41, 38, 67, 196, 171, 8, 101, 152, 132, 93]
        }

class KITTIEval(data.Dataset):

    crop = 80

    def __init__(self, image_size=None, root=osp.join(os.getenv("DATASETS"), "kitti15"), do_augment=True, mode="testing", disp_path="disp_lea", split=None):
        self.init_seed = None
        self.istest = mode == "testing"

        self.image0_list = []
        self.image1_list = []
        self.image2_list = []
        self.disp0_list = []
        self.disp1_list = []
        self.disp2_list = []
        self.calib_list = []

        included = range(200)
        if split is not None:
            included = sorted(SPLITS[split])

        for i in included:
            self.image0_list.append(osp.join(root, mode, f"image_2/{i:06d}_09.png"))
            self.image1_list.append(osp.join(root, mode, f"image_2/{i:06d}_10.png"))
            self.image2_list.append(osp.join(root, mode, f"image_2/{i:06d}_11.png"))
            self.disp0_list.append(osp.join(root, mode, f"{disp_path}/{i:06d}_09.png".format(mode)))
            self.disp1_list.append(osp.join(root, mode, f"{disp_path}/{i:06d}_10.png".format(mode)))
            self.disp2_list.append(osp.join(root, mode, f"{disp_path}/{i:06d}_11.png".format(mode)))
            self.calib_list.append(osp.join(root, mode, f"calib_cam_to_cam/{i:06d}.txt"))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)

    @staticmethod
    def write_prediction(index, disp1, disp2, flow):
        disp1 = np.pad(disp1, ((KITTIEval.crop,0),(0,0)), mode='edge')
        disp2 = np.pad(disp2, ((KITTIEval.crop, 0), (0,0)), mode='edge')
        flow = np.pad(flow, ((KITTIEval.crop, 0), (0,0),(0,0)), mode='edge')

        disp1_path = 'kitti_submission/disp_0/%06d_10.png' % index
        disp2_path = 'kitti_submission/disp_1/%06d_10.png' % index
        flow_path = 'kitti_submission/flow/%06d_10.png' % index

        flow_IO.writeDispFile(disp1, disp1_path)
        flow_IO.writeDispFile(disp2, disp2_path)
        flow_IO.writeFlowFile(flow, flow_path)

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):

        intrinsics = self.intrinsics_list[index]
        image0 = cv2.imread(self.image0_list[index])
        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        seq_name = self.image0_list[index][-13:-7]

        disp0 = cv2.imread(self.disp0_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp1 = cv2.imread(self.disp1_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_list[index], cv2.IMREAD_ANYDEPTH) / 256.0

        image0 = image0[self.crop:]
        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp0 = disp0[self.crop:]
        disp1 = disp1[self.crop:]
        disp2 = disp2[self.crop:]
        intrinsics[3] -= self.crop

        image0 = torch.from_numpy(image0).float().permute(2,0,1)
        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)
        disp0 = torch.from_numpy(disp0).float()
        disp1 = torch.from_numpy(disp1).float()
        disp2 = torch.from_numpy(disp2).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        if self.istest:
            return image0, image1, image2, disp0, disp1, disp2, intrinsics

        return image0, image1, image2, disp0, disp1, disp2, intrinsics, seq_name


class KITTI(data.Dataset):
    def __init__(self, image_size=None, root=osp.join(os.getenv("DATASETS"), "kitti15"), do_augment=True, disp_path="disp_lea", split=None):

        self.init_seed = None
        self.crop = 80

        if do_augment:
            self.augmentor = SparseAugmentor(image_size)
        else:
            self.augmentor = None

        self.image0_list = []
        self.image1_list = []
        self.image2_list = []

        self.disp1_gt_list = []
        self.disp2_gt_list = []

        self.disp0_list = []
        self.disp1_list = []
        self.disp2_list = []

        self.flow_list = []
        self.calib_list = []
        self.obj_list = []

        included = range(200)
        if split is not None:
            included = sorted(SPLITS[split])

        for i in included:
            self.image0_list.append(osp.join(root, "training", f"image_2/{i:06d}_09.png"))
            self.image1_list.append(osp.join(root, "training", f"image_2/{i:06d}_10.png"))
            self.image2_list.append(osp.join(root, "training", f"image_2/{i:06d}_11.png"))

            self.disp1_gt_list.append(osp.join(root, "training", f"disp_occ_0/{i:06d}_10.png"))
            self.disp2_gt_list.append(osp.join(root, "training", f"disp_occ_1/{i:06d}_10.png"))

            self.disp0_list.append(osp.join(root, "training", f"{disp_path}/{i:06d}_09.png"))
            self.disp1_list.append(osp.join(root, "training", f"{disp_path}/{i:06d}_10.png"))
            self.disp2_list.append(osp.join(root, "training", f"{disp_path}/{i:06d}_11.png"))

            self.flow_list.append(osp.join(root, "training", f"flow_occ/{i:06d}_10.png"))
            self.calib_list.append(osp.join(root, "training", f"calib_cam_to_cam/{i:06d}.txt"))
            self.obj_list.append(osp.join(root, "training", f"obj_map/{i:06d}_10.png"))

        self.intrinsics_list = []
        for calib_file in self.calib_list:
            with open(calib_file) as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    if row[0] == 'K_02:':
                        K = np.array(row[1:], dtype=np.float32).reshape(3,3)
                        kvec = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
                        self.intrinsics_list.append(kvec)
        print(len(self.image1_list))

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        image0 = cv2.imread(self.image0_list[index])
        image1 = cv2.imread(self.image1_list[index])
        image2 = cv2.imread(self.image2_list[index])

        disp1 = cv2.imread(self.disp1_gt_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2 = cv2.imread(self.disp2_gt_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp0_dense = cv2.imread(self.disp0_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp1_dense = cv2.imread(self.disp1_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        disp2_dense = cv2.imread(self.disp2_list[index], cv2.IMREAD_ANYDEPTH) / 256.0
        obj_map = flow_IO.readKITTIObjMap(self.obj_list[index])

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        intrinsics = self.intrinsics_list[index]

        SCALE = np.random.uniform(0.08, 0.15)

        # crop top 80 pixels, no ground truth information
        image0 = image0[self.crop:]
        image1 = image1[self.crop:]
        image2 = image2[self.crop:]
        disp1_ = disp1[self.crop:]
        disp2_ = disp2[self.crop:]
        flow = flow[self.crop:]
        valid = valid[self.crop:]
        disp0_dense = disp0_dense[self.crop:]
        disp1_dense_ = disp1_dense[self.crop:]
        disp2_dense = disp2_dense[self.crop:]
        obj_map = obj_map[self.crop:]
        intrinsics[3] -= self.crop

        image0 = torch.from_numpy(image0).float().permute(2,0,1)
        image1 = torch.from_numpy(image1).float().permute(2,0,1)
        image2 = torch.from_numpy(image2).float().permute(2,0,1)

        #disp1 = torch.from_numpy(disp1_ / intrinsics[0]) / SCALE
        disp2 = torch.from_numpy(disp2_ / intrinsics[0]) / SCALE
        disp0_dense = torch.from_numpy(disp0_dense / intrinsics[0]) / SCALE
        disp1_dense = torch.from_numpy(disp1_dense_ / intrinsics[0]) / SCALE
        disp2_dense = torch.from_numpy(disp2_dense / intrinsics[0]) / SCALE

        disp1_dense_unscaled = torch.from_numpy(disp1_dense_)
        disp1_unscaled = torch.from_numpy(disp1_)
        disp2_unscaled = torch.from_numpy(disp2_)
        obj_map = torch.from_numpy(obj_map)

        dz = (disp2 - disp1_dense).unsqueeze(dim=-1)
        depth0 = 1.0 / disp0_dense.clamp(min=0.01).float()
        depth1 = 1.0 / disp1_dense.clamp(min=0.01).float()
        depth2 = 1.0 / disp2_dense.clamp(min=0.01).float()

        intrinsics = torch.from_numpy(intrinsics)
        valid = torch.from_numpy(valid)
        flow = torch.from_numpy(flow)

        valid = valid * (disp2 > 0).float()
        flow = torch.cat([flow, dz], -1)

        if self.augmentor is not None:
            image0, image1, image2, depth0, depth1, depth2, flow, valid, intrinsics, disp1_unscaled, disp2_unscaled, disp1_dense_unscaled, obj_map = \
                self.augmentor(image0, image1, image2, depth0, depth1, depth2, flow, valid, intrinsics, disp1_unscaled, disp2_unscaled, disp1_dense_unscaled, obj_map)
        return image0, image1, image2, depth0, depth1, depth2, flow, valid, intrinsics, disp1_unscaled, disp2_unscaled, disp1_dense_unscaled, obj_map
