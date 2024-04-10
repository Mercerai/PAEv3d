import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
from nerf_network import DummyEmbedder, Embedder, MLPNet
import os
import logging

logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of nerfnet
######################################################################################
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


class NerfNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        if args.use_pe:
            self.fg_embedder_position = Embedder(input_dim=3,
                                                 max_freq_log2=args.max_freq_log2 - 1,
                                                 N_freqs=args.max_freq_log2,
                                                 N_anneal=args.N_anneal,
                                                 N_anneal_min_freq=args.N_anneal_min_freq,
                                                 use_annealing=args.use_annealing)
            self.fg_embedder_viewdir = Embedder(input_dim=3,
                                                max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                                N_freqs=args.max_freq_log2_viewdirs,
                                                N_anneal=args.N_anneal,
                                                N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
                                                use_annealing=args.use_annealing)
        else:
            self.fg_embedder_position = DummyEmbedder(input_dim=3)
            self.fg_embedder_viewdir = DummyEmbedder(input_dim=3)

        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs,
                             act=args.activation,
                             garf_sigma=args.garf_sigma,
                             crop_y=(args.crop_y_min, args.crop_y_max),
                             crop_r=args.crop_r,
                             init_gain=args.init_gain)
        # # background; bg_pt is (x, y, z, 1/r)
        # self.bg_embedder_position = Embedder(input_dim=4,
        #                                      max_freq_log2=args.max_freq_log2 - 1,
        #                                      N_freqs=args.max_freq_log2,
        #                                      N_anneal=args.N_anneal,
        #                                      N_anneal_min_freq=args.N_anneal_min_freq,
        #                                      use_annealing=args.use_annealing)
        # self.bg_embedder_viewdir = Embedder(input_dim=3,
        #                                     max_freq_log2=args.max_freq_log2_viewdirs - 1,
        #                                     N_freqs=args.max_freq_log2_viewdirs,
        #                                     N_anneal=args.N_anneal,
        #                                     N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
        #                                     use_annealing=args.use_annealing)
        # self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
        #                      input_ch=self.bg_embedder_position.out_dim,
        #                      input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
        #                      use_viewdirs=args.use_viewdirs,
        #                      act=args.activation)

        self.with_bg = args.with_bg
        self.with_ldist = args.use_ldist_reg

        self.bg_color = args.bg_color

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, iteration):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals: [..., N_samples]
        :return
        '''
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm  # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        # input = torch.cat((self.fg_embedder_position(fg_pts, iteration),
        #                    self.fg_embedder_viewdir(fg_viewdirs, iteration)), dim=-1)
        # fg_raw = self.fg_net(input)
        fg_raw = self.fg_net(fg_pts, fg_viewdirs, iteration=iteration,
                             embedder_position=self.fg_embedder_position,
                             embedder_viewdir=self.fg_embedder_viewdir)
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]),
                                          dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)  # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T  # [..., N_samples]
        AO = fg_weights.sum(dim=1)
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]

        fg_mask = torch.sum(fg_weights, -1)
        norm_depth = fg_depth_map
        norm_depth[norm_depth<1e-6]=1e-6
        norm_depth = norm_depth/norm_depth.max()
        norm_depth[norm_depth < 1e-6] = 1e-6
        norm_depth_mask = (norm_depth < 1e-5)
        norm_disp = 1/norm_depth
        norm_disp[norm_disp < 1e-6] = 1e-6
        assert(norm_disp.max() != 0)
        norm_disp = norm_disp/norm_disp.max()
        norm_disp[norm_depth_mask] = 1e-6

        # raw_depth = torch.max(1e-10 * torch.ones_like(fg_depth_map), fg_depth_map / (torch.sum(fg_weights, -1)+1e-10))
        # raw_depth[raw_depth==0] = 1e-10
        # fg_disp_map = 1. / raw_depth
        fg_midpoint = (fg_z_vals[..., 1:] + fg_z_vals[..., :-1])/2
        fg_midpoint = ray_d_norm * torch.cat((fg_midpoint, (fg_z_max.unsqueeze(-1) + fg_z_vals[..., -1:])/2),
                                          dim=-1)

        fg_midpointdist = abs(fg_midpoint.unsqueeze(-1) - fg_midpoint.unsqueeze(-2))

        if self.with_ldist:
            fg_ldist1 = torch.sum(fg_weights.unsqueeze(-1)*fg_weights.unsqueeze(-2)*fg_midpointdist, (-2, -1))
            fg_ldist2 = torch.sum(1/3*(fg_weights**2)*fg_dists, -1)
            fg_ldist = fg_ldist1+fg_ldist2

        # # render background
        # N_samples = bg_z_vals.shape[-1]
        # bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        # bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        # input = torch.cat((self.bg_embedder_position(bg_pts, iteration),
        #                    self.bg_embedder_viewdir(bg_viewdirs, iteration)), dim=-1)
        # # near_depth: physical far; far_depth: physical near
        # input = torch.flip(input, dims=[-2, ])
        # bg_z_vals = torch.flip(bg_z_vals, dims=[-1, ])  # 1--->0
        # bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        # bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        # bg_raw = self.bg_net(input)
        # bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # # Eq. (3): T
        # # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        # T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        # T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        # bg_weights = bg_alpha * T  # [..., N_samples]

        # bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]
        # bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # # composite foreground and background
        # bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        # bg_depth_map = bg_lambda * bg_depth_map
        if self.with_bg:
            rgb_map = fg_rgb_map + bg_rgb_map
        else:
            # rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1)*(159./255.)  # hard coded value of background in sRGB = 159/255
            rgb_map = fg_rgb_map + bg_lambda.unsqueeze(-1)*(self.bg_color/255.)  # hard coded value of background in sRGB = 159/255

        ret = OrderedDict([('rgb', rgb_map),            # loss
                           ('fg_weights', fg_weights),  # importance sampling
                           ('fg_mask', fg_mask),
                           # ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map.detach()),      # below are for logging
                           ('fg_depth', fg_depth_map.detach()),
                           # ('bg_rgb', bg_rgb_map.detach()),
                           # ('bg_depth', bg_depth_map.detach()),
                           ('bg_lambda', bg_lambda),
                           ('norm_disp',norm_disp),
                           ('norm_depth', norm_depth),
                           ('AO', AO),
                           ])
        if self.with_ldist:
            ret['fg_ldist'] = fg_ldist      # distortion regularizer
        return ret


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]


class NerfNetWithAutoExpo(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()
        self.nerf_net = NerfNet(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert(img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, iteration, img_name=None):  
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals: [..., N_samples]
        :return
        '''
        if img_name is not None:
            img_name = remap_name(img_name)

        ret = self.nerf_net(ray_o, ray_d, fg_z_max, fg_z_vals, iteration)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret
