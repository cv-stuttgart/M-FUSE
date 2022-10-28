import torch
import torch.nn as nn

import mfuse.projective_ops as pops

from mfuse.sampler_ops import bilinear_sampler, depth_sampler


def get_deconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )


def get_conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, channels, conv_size, activation):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, conv_size, padding=conv_size//2)
        self.conv2 = nn.Conv2d(channels, channels, conv_size, padding=conv_size//2)
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x += identity
        x = self.activation(x)
        return x


def get_inbetween(inbetween_type, channels):
    if inbetween_type == "none":
        return nn.Identity()
    elif inbetween_type == "conv":
        return get_conv(channels, channels)
    elif inbetween_type == "2conv":
        return nn.Sequential(get_conv(channels, channels),get_conv(channels, channels))
    elif inbetween_type == "resblock":
        return ResBlock(channels, conv_size=3, activation="leakyrelu")
    else:
        raise ValueError()


class MFUSE(nn.Module):
    def __init__(self, in_channels=43, out_channels=3, join="add", inbetween="conv"):
        super(MFUSE,self).__init__()

        assert(join in ["cat", "add"])
        self.join = join

        assert(inbetween in ["none", "conv", "2conv", "resblock"])

        self.conv0a = get_conv(in_channels, 64)
        self.conv0b = get_inbetween(inbetween, 64)
        self.down01 = get_conv(64, 64, stride=2)
        self.conv1a = get_conv(64, 128)
        self.conv1b = get_inbetween(inbetween, 128)
        self.down12 = get_conv(128, 128, stride=2)
        self.conv2a = get_conv(128, 256)
        self.conv2b = get_inbetween(inbetween, 256)

        self.up21 = get_deconv(256, 128)
        if self.join == "cat":
            self.conv1c = get_conv(256, 128)
        elif self.join == "add":
            self.conv1c = get_conv(128, 128)
        self.conv1d = get_inbetween(inbetween, 128)

        self.up10 = get_deconv(128, 64)
        if self.join == "cat":
            self.conv0c = get_conv(128, 64)
        elif self.join == "add":
            self.conv0c = get_conv(64, 64)
        self.conv0d = get_inbetween(inbetween,64)
        self.conv0e = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out0 = self.conv0b(self.conv0a(x))
        out1 = self.conv1b(self.conv1a(self.down01(out0)))
        out2 = self.conv2b(self.conv2a(self.down12(out1)))

        up1 = self.up21(out2)
        if self.join == "cat":
            join1 = torch.cat((out1, up1), 1)
        elif self.join == "add":
            join1 = out1 + up1

        up0 = self.up10(self.conv1d(self.conv1c(join1)))
        if self.join == "cat":
            join0 = torch.cat((out0, up0), 1)
        elif self.join == "add":
            join0 = out0 + up0

        flow0 = self.conv0e(self.conv0d(self.conv0c(join0)))

        return flow0


class MFUSE_4lvl(nn.Module):
    def __init__(self, in_channels=43, out_channels=3, join="add", inbetween="conv"):
        super(MFUSE_4lvl, self).__init__()

        assert(join in ["cat", "add"])
        self.join = join

        assert(inbetween in ["none", "conv", "2conv", "resblock"])

        self.conv0a = get_conv(in_channels, 64)
        self.conv0b = get_inbetween(inbetween, 64)
        self.down01 = get_conv(64, 64, stride=2)
        self.conv1a = get_conv(64, 128)
        self.conv1b = get_inbetween(inbetween, 128)
        self.down12 = get_conv(128, 128, stride=2)
        self.conv2a = get_conv(128, 256)
        self.conv2b = get_inbetween(inbetween, 256)
        self.down23 = get_conv(256, 256, stride=2)
        self.conv3a = get_conv(256, 512)
        self.conv3b = get_inbetween(inbetween, 512)

        self.up32 = get_deconv(512, 256)
        if self.join == "cat":
            self.conv2c = get_conv(512, 256)
        elif self.join == "add":
            self.conv2c = get_conv(256, 256)
        self.conv2d = get_inbetween(inbetween, 256)

        self.up21 = get_deconv(256, 128)
        if self.join == "cat":
            self.conv1c = get_conv(256, 128)
        elif self.join == "add":
            self.conv1c = get_conv(128, 128)
        self.conv1d = get_inbetween(inbetween, 128)

        self.up10 = get_deconv(128, 64)
        if self.join == "cat":
            self.conv0c = get_conv(128, 64)
        elif self.join == "add":
            self.conv0c = get_conv(64, 64)
        self.conv0d = get_inbetween(inbetween, 64)
        self.conv0e = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out0 = self.conv0b(self.conv0a(x))
        out1 = self.conv1b(self.conv1a(self.down01(out0)))
        out2 = self.conv2b(self.conv2a(self.down12(out1)))
        out3 = self.conv3b(self.conv3a(self.down23(out2)))

        up2 = self.up32(out3)
        if self.join == "cat":
            join2 = torch.cat((out2, up2), 1)
        elif self.join == "add":
            join2 = out2 + up2

        up1 = self.up21(self.conv2d(self.conv2c(join2)))

        if self.join == "cat":
            join1 = torch.cat((out1, up1), 1)
        elif self.join == "add":
            join1 = out1 + up1

        up0 = self.up10(self.conv1d(self.conv1c(join1)))
        if self.join == "cat":
            join0 = torch.cat((out0, up0), 1)
        elif self.join == "add":
            join0 = out0 + up0

        flow0 = self.conv0e(self.conv0d(self.conv0c(join0)))

        return flow0


class MFUSE_2lvl(nn.Module):
    def __init__(self, in_channels=43, out_channels=3, join="add", inbetween="conv"):
        super(MFUSE_2lvl,self).__init__()

        assert(join in ["cat", "add"])
        self.join = join

        assert(inbetween in ["none", "conv", "2conv", "resblock"])

        self.conv0a = get_conv(in_channels, 64)
        self.conv0b = get_inbetween(inbetween, 64)
        self.down01 = get_conv(64, 64, stride=2)
        self.conv1a = get_conv(64, 128)
        self.conv1b = get_inbetween(inbetween, 128)

        self.up10 = get_deconv(128, 64)
        if self.join == "cat":
            self.conv0c = get_conv(128, 64)
        elif self.join == "add":
            self.conv0c = get_conv(64, 64)
        self.conv0d = get_inbetween(inbetween, 64)
        self.conv0e = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out0 = self.conv0b(self.conv0a(x))
        out1 = self.conv1b(self.conv1a(self.down01(out0)))

        up0 = self.up10(out1)
        if self.join == "cat":
            join0 = torch.cat((out0, up0), 1)
        elif self.join == "add":
            join0 = out0 + up0

        flow0 = self.conv0e(self.conv0d(self.conv0c(join0)))

        return flow0


class Fusion(nn.Module):
    def __init__(self, baseline_model, freeze_baseline=False, inputs=["d1,dch,fl,corrCost,dispRes,embVec"],
            fusion_module="fusion1", fusion_join="cat", fusion_inbetween="conv"):

        super(Fusion, self).__init__()
        self.fw_model = baseline_model
        self.bw_model = baseline_model

        allowed_inputs = ["d1", "dch", "fl", "corrCost", "dispRes", "embVec", "img1", "bce"]

        sizes = {"d1": 1, "dch": 2, "fl": 4, "corrCost": 2, "dispRes": 2, "embVec": 32, "img1": 3, "bce": 2}

        inputs = inputs.split(",")
        assert all([i in allowed_inputs for i in inputs])
        self.inputs = inputs
        in_channels = sum([sizes[i] for i in inputs])

        if fusion_module == "MFUSE":
            self.fusion_model = MFUSE(in_channels=in_channels, out_channels=3, join=fusion_join, inbetween=fusion_inbetween)
        elif fusion_module == "MFUSE_4lvl":
            self.fusion_model = MFUSE_4lvl(in_channels=in_channels, out_channels=3, join=fusion_join, inbetween=fusion_inbetween)
        elif fusion_module == "MFUSE_2lvl":
            self.fusion_model = MFUSE_2lvl(in_channels=in_channels, out_channels=3, join=fusion_join, inbetween=fusion_inbetween)
        else:
            raise ValueError()

        if freeze_baseline:
            for param in self.fw_model.parameters():
                param.requires_grad = False
            for param in self.bw_model.parameters():
                param.requires_grad = False


    def forward(self, image0, image1, image2, depth0, depth1, depth2, intrinsics, disp1_dense, iters=12):
        Ts_FW, add_FW, flow_est_list, flow_rev_list = self.fw_model(image1, image2, depth1, depth2, intrinsics, iters=iters, train_mode=False)
        Ts_BW, add_BW, _, _ = self.bw_model(image1, image0, depth1, depth0, intrinsics, iters=iters, train_mode=False)

        _, h ,w = Ts_FW.shape

        # RAFT-3D's depth scaling
        depth_scale = 0.1

        disp1 = disp1_dense
        disp1 = disp1[:,None,:,:]

        flow_FW, _, _ = pops.induced_flow(Ts_FW, depth1, intrinsics)
        flow_FW = flow_FW.permute(0,3,1,2)
        flow_FW, dispch_FW = flow_FW.split([2,1], dim=1)
        dispch_FW = dispch_FW * intrinsics[:,None,None,None,0] * depth_scale
        disp2_FW = disp1 + dispch_FW

        flow_BW, _, _ = pops.induced_flow(Ts_BW, depth1, intrinsics)
        flow_BW = flow_BW.permute(0,3,1,2)
        flow_BW, dispch_BW = flow_BW.split([2,1], dim=1)
        dispch_BW = dispch_BW * intrinsics[:,None,None,None,0] * depth_scale
        disp2_BW = disp1 + dispch_BW

        # inversion
        Ts_BW_inv = Ts_BW.inv()
        flow_inv, _, _ = pops.induced_flow(Ts_BW_inv, depth1, intrinsics)
        flow_inv = flow_inv.permute(0,3,1,2)
        flow_inv, dispch_inv = flow_inv.split([2,1], dim=1)
        dispch_inv = dispch_inv * intrinsics[:,None,None,None,0] * depth_scale

        add_FW = add_FW.permute(0,3,1,2)
        corrCost_FW, embVec_FW = add_FW.split([1,16], dim=1)
        add_BW = add_BW.permute(0,3,1,2)
        corrCost_BW, embVec_BW = add_BW.split([1,16], dim=1)

        y1, x1 = torch.meshgrid(
            torch.arange(h, device=flow_FW.device).float(),
            torch.arange(w, device=flow_FW.device).float())

        x2 = x1 + flow_FW[:,0,:,:]
        y2 = y1 + flow_FW[:,1,:,:]
        targets2 = torch.stack((x2,y2), dim=-1)
        warped2, _ = depth_sampler(1.0 / depth2, targets2)
        warped2 =  warped2 * intrinsics[:,None,None,0] * depth_scale

        x0 = x1 + flow_BW[:,0,:,:]
        y0 = y1 + flow_BW[:,1,:,:]
        targets0 = torch.stack((x0,y0), dim=-1)
        warped0, _ = depth_sampler(1.0 / depth0, targets0)
        warped0 = warped0 * intrinsics[:,None,None,0] * depth_scale

        dispRes_FW = warped2[:,None,:,:] - disp2_FW
        dispRes_BW = warped0[:,None,:,:] - disp2_BW

        # brightness constancy errors
        warped_img_FW = bilinear_sampler(image2, targets2)
        bce_FW = (image1 - warped_img_FW).abs().mean(dim=1)[:,None]

        x2_inv = x1 + flow_inv[:,0,:,:]
        y2_inv = y1 + flow_inv[:,1,:,:]
        targets2_inv = torch.stack((x2_inv,y2_inv), dim=-1)
        warped_img_inv = bilinear_sampler(image2, targets2_inv)
        bce_inv = (image1 - warped_img_inv).abs().mean(dim=1)[:,None]

        fusion_inp = []
        for inp in self.inputs:
            if inp == "d1":
                fusion_inp += [disp1]
            elif inp == "dch":
                fusion_inp += [dispch_FW, dispch_inv]
            elif inp == "fl":
                fusion_inp += [flow_FW, flow_inv]
            elif inp == "corrCost":
                fusion_inp += [corrCost_FW, corrCost_BW]
            elif inp == "dispRes":
                fusion_inp += [dispRes_FW, dispRes_BW]
            elif inp == "embVec":
                fusion_inp += [embVec_FW, embVec_BW]
            elif inp == "img1":
                fusion_inp += [image1]
            elif inp == "bce":
                fusion_inp += [bce_FW, bce_inv]
            else:
                raise ValueError()

        fusion_inp = torch.cat(fusion_inp, dim=1)

        fusion_result = self.fusion_model(fusion_inp.float())

        # output must be flow+dispchange tensor with shape [b,2+1,h,w]

        additional_r3d = [flow_est_list, flow_rev_list]

        return fusion_result, additional_r3d
