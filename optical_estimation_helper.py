import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import cv2
import imageio
import numba
import matplotlib
import logging

class OpticalEstimationHelper(object):
    def __init__(self, events, H, W):
        super().__init__()
        self.events = events
        self.H = H
        self.W = W
        self.xs, self.ys, self.ts, self.ps = self.events
        self.hot_events = torch.zeros([self.H,self.W])
        self.hot_idx = 0
        self.event_formatting()
        self.event_cnt = self.events_to_channels(sensor_size=(self.H, self.W))
        self.event_voxel = self.events_to_voxel(sensor_size=(self.H, self.W))
        self.create_polarity_mask()
        self.create_mask_encoding()
        hot_mask = self.create_hot_mask(self.event_cnt)
        hot_mask_voxel = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
        self.event_voxel = self.event_voxel * hot_mask_voxel
        self.event_cnt = self.event_cnt * hot_mask_cnt
        self.event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))


    def visualize_flowmap(self, estimated_flow):
        flow_npy = estimated_flow.reshape((self.H, self.W, 2))
        flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
        flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('flow_map.png', flow_npy)
        return flow_npy

    def flow_to_image(self, flow_x, flow_y):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
        flows = np.stack((flow_x, flow_y), axis=2)
        mag = np.linalg.norm(flows, axis=2)
        min_mag = np.min(mag)
        mag_range = np.max(mag) - min_mag

        ang = np.arctan2(flow_y, flow_x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag - min_mag
        if mag_range != 0.0:
            hsv[:, :, 2] /= mag_range

        flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (255 * flow_rgb).astype(np.uint8)


    def events_to_voxel(self, num_bins=2, sensor_size=None, round_ts=False):
        """
        Generate a voxel grid from input events using temporal bilinear interpolation.
        """
        num_bins=2
        voxel = []
        ts = self.ts * (num_bins - 1)

        if round_ts:
            ts = torch.round(self.ts)

        zeros = torch.zeros(self.ts.size())
        for b_idx in range(num_bins):
            weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
            # print(weights)
            # print('image max:', weights.max(), 'image min:', weights.min())
            voxel_bin = self.events_to_image(ps=self.ps * weights, sensor_size=sensor_size)
            # cv2.imwrite('voxel_{}.png'.format(b_idx), voxel_bin.numpy()*255)
            voxel.append(voxel_bin)

        return torch.stack(voxel)

    def events_to_image(self, ps, sensor_size=None, accumulate=True):
        """
        Accumulate events into an image.
        """

        device = self.xs.device
        img_size = list(sensor_size)
        img = torch.zeros(img_size).to(device)

        if self.xs.dtype is not torch.long:
            self.xs = self.xs.long().to(device)
        if self.ys.dtype is not torch.long:
            self.ys = self.ys.long().to(device)
        img.index_put_((self.ys, self.xs), ps, accumulate=accumulate)
        # print('image max:', ps.max(), 'image min:', ps.min())
        # cv2.imwrite('events.png', img.numpy()*255)
        img = torch.clamp(img, -5, 5)
        # print(img.max(), img.min())
        return img
    
    def events_to_channels(self, sensor_size=None):
        """
        Generate a two-channel event image containing per-pixel event counters.
        """
        mask_pos = self.ps.clone()
        mask_neg = self.ps.clone()
        mask_pos[self.ps < 0] = 0
        mask_neg[self.ps > 0] = 0

        pos_cnt = self.events_to_image(self.ps * mask_pos, sensor_size=sensor_size)
        neg_cnt = self.events_to_image(self.ps * mask_neg, sensor_size=sensor_size)
        # cv2.imwrite('pos_events.png', pos_cnt.numpy()*255)
        # cv2.imwrite('neg_events.png', neg_cnt.numpy()*255)
        return torch.stack([pos_cnt, neg_cnt])
    
    def event_formatting(self):
        """
        Reset sequence-specific variables.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        self.xs = torch.from_numpy(self.xs.astype(np.float32))
        self.ys = torch.from_numpy(self.ys.astype(np.float32))
        self.ts = torch.from_numpy(self.ts.astype(np.float32))
        self.ps = torch.from_numpy(self.ps.astype(np.float32))
        if self.ts.shape[0] > 0:
            self.ts = (self.ts - self.ts[0]) / (self.ts[-1] - self.ts[0])

    def create_mask_encoding(self):
        """
        Creates a per-pixel and per-polarity event count and average timestamp representation.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [1 x H x W] event representation
        """

        event_mask = self.events_to_image(self.ps.abs(), sensor_size=(self.H,self.W), accumulate=False)
        self.event_mask = event_mask.view((1, event_mask.shape[0], event_mask.shape[1]))
    
    def compute_masked_window_flow(self):

        if self.overwrite_intermediate:
            return self._flow_map[-1] * self._event_mask
        else:
            avg_flow = self._flow_map[0] * self._event_mask[:, 0:1, :, :]
            for i in range(1, self._event_mask.shape[1]):
                avg_flow += self._flow_map[i] * self._event_mask[:, i : i + 1, :, :]
            avg_flow /= torch.sum(self._event_mask, dim=1, keepdim=True) + 1e-9
            return avg_flow
    
    def create_polarity_mask(self):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """

        event_list_pol_mask = torch.stack([self.ps, self.ps], dim = 1)
        event_list_pol_mask[0, :][event_list_pol_mask[0, :] < 0] = 0
        event_list_pol_mask[1, :][event_list_pol_mask[1, :] > 0] = 0
        event_list_pol_mask[1, :] *= -1
        self.event_list_pol_mask = event_list_pol_mask

    def create_list_encoding(self):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] event representation
        """

        return torch.stack([self.ts, self.ys, self.xs, self.ps], dim = 1)
    
    def vis_iwe(self, iwe):
        iwe = iwe.detach()
        iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((self.H, self.W, 2))
        iwe_npy = self.warped_events_to_image(iwe_npy)
        # cv2.namedWindow("Image of Warped Events", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Image of Warped Events", int(self.px), int(self.px))
        # cv2.imwrite("iwe.png", iwe_npy*255)

    def warped_events_to_image(self, event_cnt, color_scheme="green_red"):
        """
        Visualize the input events.
        :param event_cnt: [batch_size x 2 x H x W] per-pixel and per-polarity event count
        :param color_scheme: green_red/gray
        :return event_image: [H x W x 3] color-coded event image
        """
        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        return event_image
    
    def create_hot_mask(self, event_cnt):
        """
        Creates a one channel tensor that can act as mask to remove pixel with high event rate.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [H x W] binary mask
        """
        hot_update = torch.sum(event_cnt, dim=0)
        hot_update[hot_update > 0] = 1
        self.hot_events += hot_update
        self.hot_idx += 1
        event_rate = self.hot_events / self.hot_idx
        return  self.get_hot_event_mask(
            event_rate,
            self.hot_idx,
            max_px=100,
            min_obvs=5,
            max_rate=0.8,
        )
    
    def get_hot_event_mask(self, event_rate, idx, max_px=100, min_obvs=5, max_rate=0.8):
        """
        Returns binary mask to remove events from hot pixels.
        """

        mask = torch.ones(event_rate.shape).to(event_rate.device)
        if idx > min_obvs:
            for i in range(max_px):
                argmax = torch.argmax(event_rate)
                index = (argmax // event_rate.shape[1], argmax % event_rate.shape[1])
                if event_rate[index] > max_rate:
                    event_rate[index] = 0
                    mask[index] = 0
                else:
                    break
        return mask