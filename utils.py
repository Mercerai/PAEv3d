import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

torch.set_default_dtype(torch.float32)
HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


# misc utils
def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)
    
def img2mae(x, y, mask=None):
    if mask is None:
        return torch.nn.functional.l1_loss(x, y)
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)
    
img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


def DisLoss(dis_map, optFlow_map, AO_mask):  
    mask = AO_mask
    # loss = torch.mean(((dis_map[mask] - optFlow_map[mask]))**2)
    loss = torch.nn.functional.l1_loss(dis_map[mask] , optFlow_map[mask])
    if mask.sum() == 0:
        loss = torch.tensor(0)
        loss.item = lambda: 0
        return loss
    return loss

def TransLoss(bg_lambda, c):
    return torch.mean(torch.log(1+(bg_lambda**2/c)))

def normalize(x):
    min = x.min()
    max = x.max()

    return (x - min) / ((max - min) + TINY_NUMBER)

def MaskLoss(fg_mask, mask_gt, AO_mask):
    mask = mask_gt == 1

    # loss = torch.nn.functional.binary_cross_entropy(fg_mask[mask], mask_gt[mask], reduction='mean')
    mask = torch.logical_and(mask , AO_mask)
    if mask.sum() == 0:
        loss = torch.tensor(0)
        loss.item = lambda: 0
        return loss
    # loss = torch.nn.functional.l1_loss(fg_mask[mask], mask_gt[mask])
    loss = torch.nn.functional.binary_cross_entropy(fg_mask[mask], mask_gt[mask], reduction='mean')
    # loss = torch.mean((fg_mask[mask] - mask_gt[mask])**2)
    return loss

def FlowLoss_patches(curr_rgb, prev_rgb, optical_flow, events_gt, AO_mask):
    events = reshape_patches_edited(curr_rgb, prev_rgb, optical_flow)
    # mask = torch.all(optical_flow != 0, dim=1).unsqueeze(1).expand(-1, 3)
    # mask = AO >= 0.2
    # mask = mask.unsqueeze(-1).expand(-1, 3)
    # if mask.sum() == 0:
    #     loss = torch.tensor(0)
    #     loss.item = lambda: 0
    #     return loss
    return torch.nn.functional.l1_loss(events, events_gt, reduction='mean')

def reshape_patches_edited(curr_rgb, prev_rgb, optical_flow):
    curr_rgb = curr_rgb.reshape((256, 2, 2, 3))
    prev_rgb = prev_rgb.reshape((256, 2, 2, 3))
    prev_rgb = prev_rgb*255
    optical_flow = optical_flow.reshape((256, 2, 2, 2))
    previous_gradient_x = torch.zeros_like(prev_rgb)
    previous_gradient_y = torch.zeros_like(prev_rgb)
    previous_gradient_x[:,:-1,:,:] = prev_rgb[:,1:,:,:] - prev_rgb[:,:-1,:,:]
    previous_gradient_y[:,:,:-1,:] = prev_rgb[:,:,1:,:] - prev_rgb[:,:,:-1,:]
    previous_gradient_x[:,-1,:,:] = previous_gradient_x[:,-2,:,:]
    previous_gradient_y[:,:,-1,:] = previous_gradient_y[:,:,-2,:]
    events = -previous_gradient_x * optical_flow[:,:,:,0].unsqueeze(-1) - previous_gradient_y * optical_flow[:,:,:,1].unsqueeze(-1)
    return events.reshape((1024, 3))


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
# gray2rgb = lambda x: np.tile(x[:,:,np.newaxis], (1, 1, 3))
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


########################################################################################################################
#
########################################################################################################################
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(1.2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = ['{:3.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, append_cbar=False):
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + TINY_NUMBER

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar


# tensor
def colorize(x, cmap_name='jet', append_cbar=False, mask=None):
    x = x.numpy()
    if mask is not None:
        mask = mask.numpy().astype(dtype=np.bool)
    x, cbar = colorize_np(x, cmap_name, mask)

    if append_cbar:
        x = np.concatenate((x, np.zeros_like(x[:, :5, :]), cbar), axis=1)

    x = torch.from_numpy(x)
    return x
