import sys
sys.path.append('.')

import argparse
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Logger, normalize_image

from data_readers.kitti import KITTI
from mfuse.fusion import Fusion
from mfuse.raft3d_bilaplacian import RAFT3D
from evaluation_fusion import test_kitti


VAL_FREQ = 200
TRAIN_EVAL_FREQ = 1000
SAVE_FREQ = 5000


def loss_r3d(flow2d_est, flow2d_rev, flow_gt, valid_mask, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """
    RV_WEIGHT = 0.2
    DZ_WEIGHT = 250.0

    N = len(flow2d_est)
    loss = 0.0

    for i in range(N):
        w = gamma**(N - i - 1)
        fl_rev = flow2d_rev[i]

        fl_est, dz_est = flow2d_est[i].split([2,1], dim=-1)
        fl_gt, dz_gt = flow_gt.split([2,1], dim=-1)

        loss += w * (valid_mask * (fl_est - fl_gt).abs()).mean()
        loss += w * DZ_WEIGHT * (valid_mask * (dz_est - dz_gt).abs()).mean()
        loss += w * RV_WEIGHT * (valid_mask * (fl_rev - fl_gt).abs()).mean()

    epe_2d = (fl_est - fl_gt).norm(dim=-1)
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe2d': epe_2d.mean().item(),
        'epedz': epe_dz.mean().item(),
        '1px': (epe_2d < 1).float().mean().item(),
        '3px': (epe_2d < 3).float().mean().item(),
        '5px': (epe_2d < 5).float().mean().item(),
    }

    return loss, metrics


def loss_fn(flow, disp1_dense, disp2_gt, flow_gt, valid_mask, flow_est_list, flow_rev_list, alpha=1.0, gamma=0.4, mu=0.1):
    flow = flow.permute(0,2,3,1)
    flow, dz_est = flow.split([2,1], dim=-1)
    flow_gt_ = flow_gt[:,:,:,:2]

    disp2_est = disp1_dense + dz_est[...,0]

    est = torch.cat((disp2_est[..., None], flow), dim=-1)
    gt = torch.cat((disp2_gt[..., None], flow_gt_), dim=-1)

    loss = torch.abs(est-gt)
    loss[:,:,:,0] *= alpha
    loss = (loss.sum(dim=-1) + 0.01)**gamma

    valid_mask_ = valid_mask[..., 0]

    loss = (valid_mask_ * loss).sum()
    loss = loss / valid_mask_.sum()

    lossr3d, _ = loss_r3d(flow_est_list, flow_rev_list, flow_gt, valid_mask)
    loss = loss + mu * lossr3d

    metrics = {"loss": float(loss.float().item())}

    return loss, metrics


def fetch_dataloader(args):
    train_dataset = KITTI(do_augment=True, image_size=args.crop_size, disp_path=args.disp_path, split=args.split)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    return train_loader


def train(args, logger=None, add_steps=0):
    r3d_model = torch.nn.DataParallel(RAFT3D(args))
    if args.ckpt_r3d is not None:
        r3d_model.load_state_dict(torch.load(args.ckpt_r3d))

    model = Fusion(r3d_model, freeze_baseline=args.freeze_baseline, inputs=args.inputs, fusion_module=args.fusion_module, 
                    fusion_join=args.fusion_join, fusion_inbetween=args.fusion_inbetween)
    model = torch.nn.DataParallel(model)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    model.cuda()
    model.eval()

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if logger is None:
        logger = Logger(name=args.name, start_step=args.start_step)

    train_loader = fetch_dataloader(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.001, cycle_momentum=False, anneal_strategy="linear")

    with warnings.catch_warnings():
        # suppress scheduler warning
        warnings.simplefilter("ignore")
        for _ in range(args.start_step):
            scheduler.step()

    if args.split == "train" or args.split is None:
        trainsplit_name = "train"
        valsplit_name = "validation"
    elif args.split == "train2":
        trainsplit_name = "train2"
        valsplit_name = "validation2"
    elif args.split.startswith("train_cv"):
        trainsplit_name = args.split
        valsplit_name = "validation_cv" + args.split[-1:]

    total_steps = args.start_step
    should_keep_training = True
    while should_keep_training:
        for data_blob in train_loader:
            optimizer.zero_grad()
            image0, image1, image2, depth0, depth1, depth2, flow_gt, valid, intrinsics, disp1_gt, disp2_gt, disp1_dense, obj_map = [x.cuda() for x in data_blob]

            image0 = normalize_image(image0.float())
            image1 = normalize_image(image1.float())
            image2 = normalize_image(image2.float())

            flow_pred, additional_r3d = model(image0, image1, image2, depth0, depth1, depth2, intrinsics, disp1_dense)
            flow_est_list, flow_rev_list = additional_r3d

            valid_mask = valid.unsqueeze(-1) > 0.5

            loss, metrics = loss_fn(flow_pred, disp1_dense, disp2_gt, flow_gt, valid_mask, flow_est_list, flow_rev_list,
                alpha=args.alpha, gamma=args.gamma, mu=args.mu)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            logger.push(metrics)

            total_steps += 1

            if total_steps % SAVE_FREQ == 0:
                torch.save(model.state_dict(), f"checkpoints/{args.name}_{total_steps+add_steps:06d}.pth")

            if total_steps % VAL_FREQ == 0:
                results_val = test_kitti(model, split=valsplit_name, disp_path=args.disp_path)
                results = {}
                for k in ["D1", "D2", "Fl", "SF"]:
                    results[f"{k}/validation"] = results_val[k]
                print(results, flush=True)
                logger.write_dict(results)

            if total_steps % TRAIN_EVAL_FREQ == 0:
                results_train = test_kitti(model, split=trainsplit_name, disp_path=args.disp_path)
                results = {}
                for k in ["D1", "D2", "Fl", "SF"]:
                    results[f"{k}/train"] = results_train[k]
                print(results, flush=True)
                logger.write_dict(results)

            end_steps = args.num_steps
            if args.end_steps != -1:
                end_steps = args.end_steps
            if total_steps >= end_steps:
                should_keep_training = False
                break

    torch.save(model.state_dict(), f"checkpoints/{args.name}_{total_steps+add_steps:06d}.pth")


def train_wrapper(args):
    logger = Logger(name=args.name, start_step=args.start_step)

    if args.start_step > 0:
        if args.ckpt is None:
            raise ValueError()
        if args.ckpt_r3d is not None:
            raise ValueError()

    if args.start_step == 0 and args.ckpt_r3d is None:
        print("Warning: M-FUSE should be trained with a pretrained RAFT-3D checkpoint.")

    total_steps = args.num_steps
    unfreeze_step = args.unfreeze_step
    lr = args.lr

    skip_first = False
    skip_second = False

    if args.end_steps < 0:
        end_steps_first = -1
        end_steps_second = -1
    elif args.end_steps <= unfreeze_step:
        end_steps_first = args.end_steps
        end_steps_second = None
        skip_second = True
    else:
        end_steps_first = -1
        end_steps_second = args.end_steps - unfreeze_step


    if args.start_step <= 0:
        start_step_first = 0
        start_step_second = 0
        keep_ckpt = False
    elif args.start_step < unfreeze_step:
        start_step_first = args.start_step
        start_step_second = 0
        keep_ckpt = True
    else:
        skip_first = True
        start_step_first = None
        start_step_second = args.start_step - unfreeze_step
        keep_ckpt = True

    # first stage:
    if not skip_first:
        args.freeze_baseline = True
        args.num_steps = unfreeze_step
        args.lr = lr
        args.start_step = start_step_first
        args.end_steps = end_steps_first
        print(args)
        train(args, logger=logger)

    # second stage:
    if not skip_second:
        args.freeze_baseline = False
        args.num_steps= total_steps - unfreeze_step
        args.lr = lr / 5.0
        if keep_ckpt:
            pass
        else:
            args.ckpt = f"checkpoints/{args.name}_{unfreeze_step:06d}.pth"
        args.ckpt_r3d = None
        args.start_step = start_step_second
        args.end_steps = end_steps_second
        print(args)
        train(args, logger=logger, add_steps=unfreeze_step)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--ckpt_r3d', default=None)

    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=lambda x:[int(i) for i in x.split(",")], default=[256, 960])
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--end_steps', type=int, default=-1)
    parser.add_argument('--unfreeze_step', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--mu', type=float, default=0.1)

    parser.add_argument('--inputs', default="d1,dch,fl,corrCost,dispRes,embVec")
    parser.add_argument('--fusion_module', default="MFUSE")
    parser.add_argument('--fusion_join', default="add")
    parser.add_argument('--fusion_inbetween', default="conv")

    parser.add_argument('--disp_path', default="disp_lea")
    parser.add_argument('--split', default=None)

    args = parser.parse_args()

    print(args)
    train_wrapper(args)
