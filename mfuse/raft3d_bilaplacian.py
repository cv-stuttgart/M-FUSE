import torch
import torch.nn as nn

# lietorch for tangent space backpropogation
from lietorch import SE3

from .blocks.extractor import BasicEncoder
from .blocks.resnet import FPN
from .blocks.corr import CorrBlock
from .blocks.gru import ConvGRU
from .blocks.grid import GridFactor
from .sampler_ops import depth_sampler

from . import projective_ops as pops
from . import se3_field


GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class GridSmoother(nn.Module):
    def __init__(self):
        super(GridSmoother, self).__init__()

    def forward(self, ae, wxwy):

        factor = GridFactor()
        ae = ae.permute(0,2,3,1)

        wx = wxwy[:,0].unsqueeze(-1)
        wy = wxwy[:,1].unsqueeze(-1)
        wu = torch.ones_like(wx)
        J = torch.ones_like(wu).unsqueeze(-2)

        # residual terms
        ru = ae.unsqueeze(-2)
        rx = torch.zeros_like(ru)
        ry = torch.zeros_like(ru)

        factor.add_factor([J], wu, ru, ftype='u')
        factor.add_factor([J,-J], wx, rx, ftype='h')
        factor.add_factor([J,-J], wy, ry, ftype='v')
        factor._build_factors()

        ae = factor.solveAAt().squeeze(dim=-2)
        ae = ae.permute(0,3,1,2).contiguous()

        return ae


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args

        self.gru = ConvGRU(hidden_dim, dilation=3)

        self.solver = GridSmoother()

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae_enc = nn.Conv2d(16, 3*128, 3, padding=1)

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip(),
            nn.Sigmoid())

        self.ae_wts = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, padding=0),
            GradientClip(),
            nn.Softplus())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0),
            GradientClip())


    def forward(self, net, inp, corr, flow, twist, dz, ae, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        ae = self.ae_enc(ae)
        net = self.gru(net, inp, cor, mot, ae)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        edges = 5 * self.ae_wts(net)
        ae = self.solver(ae, edges)

        return net, mask, ae, delta, weight


class RAFT3D(nn.Module):
    def __init__(self, args):
        super(RAFT3D, self).__init__()

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = FPN(output_dim=hdim+3*hdim)
        self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)

    def initializer(self, image1):
        """ Initialize coords and transformation maps """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        y0, x0 = torch.meshgrid(torch.arange(ht//8, device=device), torch.arange(wd//8, device=device))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1)

        Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device)
        ae = torch.zeros(batch_size, 16, ht//8, wd//8, device=device)

        return Ts, ae, coords0

    def features_and_correlation(self, image1, image2):
        # extract features and build correlation volume
        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # extract context features using Resnet50
        net_inp = self.cnet(image1)
        net, inp = net_inp.split([128, 128*3], dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        return corr_fn, net, inp


    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        Ts, ae, coords0 = self.initializer(image1)
        corr_fn, net, inp = self.features_and_correlation(image1, image2)

        # intrinsics and depth at 1/8 resolution
        intrinsics_r8 = intrinsics / 8.0
        depth1_r8 = depth1[:,3::8,3::8]
        depth2_r8 = depth2[:,3::8,3::8]

        flow_est_list = []
        flow_rev_list = []

        for itr in range(iters):
            Ts = Ts.detach()

            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)

            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
            zinv, _ = depth_sampler(1.0/depth2_r8, coords1)

            corr = corr_fn(coords1.permute(0,3,1,2).contiguous())
            flow = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj
            twist = Ts.log()

            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist, ae)

            target = coords1_xyz.permute(0,3,1,2) + delta
            target = target.contiguous()

            # Gauss-Newton step
            # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
            Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

            flow2d_rev = target.permute(0,2,3,1)[...,:2] - coords0
            flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

            Ts_up = se3_field.upsample_se3(Ts, mask)
            flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics)

            flow_est_list.append(flow2d_est)
            flow_rev_list.append(flow2d_rev)

        if train_mode:
            return flow_est_list, flow_rev_list

        Ts_up = se3_field.upsample_se3(Ts, mask)
        corr_up = se3_field.cvx_upsample(corr.permute(0,2,3,1)[...,24,None], mask)
        ae_up = se3_field.cvx_upsample(ae.permute(0,2,3,1), mask)

        return Ts_up, torch.cat((corr_up, ae_up), dim=-1), flow_est_list, flow_rev_list


