import os
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleEventStream
import glob
import h5py
import sys
import torch
sys.path.append('./PAEv3d/event_flow/')
from load_blender import load_data
import mlflow
import configparser
from event_flow.models.model import (
    LIFFireNet,
    PLIFFireNet,
    ALIFFireNet,
    XLIFFireNet,
    LIFFireFlowNet,
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from event_flow.utils.utils import load_model, create_model_dir
from event_flow.models.model import (
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,
)
from event_flow.utils.iwe import compute_pol_iwe
from optical_estimation_helper import OpticalEstimationHelper
from utils import *

logger = logging.getLogger(__package__)

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg', '*.JPG', '*.PNG']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def load_event_data_split(args, camera_mgr,skip, split, max_winsize=1, win_constant_count=0):
    
    base_folder = args.datadir
    scene = args.scene
    skip=skip
    use_ray_jitter=args.use_ray_jitter
    is_colored=args.is_colored
    polarity_offset=args.polarity_offset
    cycle=args.is_cycled
    is_rgb_only=args.is_rgb_only
    randomize_winlen=args.use_random_window_len

    def parse_txt(filename, shape):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape(shape).astype(np.float32)
    
    def find_pose(npz_files, idx):
        npz = np.load(npz_files[idx], allow_pickle=True)
        pose = npz['object_poses']
        for obj in pose:
            obj_name = obj['name']
            obj_mat = obj['pose']
            if obj_name == 'Camera':
                pose = obj_mat.astype(np.float32)
                break
        return pose

    event_files_path = os.path.join(base_folder, "H5_Files", scene, 'event.h5')
    scene_files_path = os.path.join(base_folder, "Render_info", scene)
    # pose_files = find_files('{}/Poses'.format(base_folder), exts=['*.txt'])
    intrinsic_files = find_files('{}/Intrinsics'.format(base_folder), exts=['*.txt'])
    npz_files = find_files('{}'.format(scene_files_path), exts=["*.npz"])
    img_files = find_files('{}'.format(scene_files_path), exts=['*.png'])
    with h5py.File(event_files_path, 'r') as h5_file:
        events = np.array(h5_file['events'])
        ts, xs, ys, ps = events[:,0], events[:,1], events[:,2], events[:,3]
        xs = xs.astype(np.int32)
        ys = ys.astype(np.int32)
        ps = ps.astype(np.int32)
        ps = (ps * 2) - 1
        ts -= ts[0]
        ts = ts/ts.max()
    
    cam_cnt = len(npz_files)
    T = np.diag([1, -1, -1, 1])

    for i in range(cam_cnt): 
        curr_file = img_files[i]
        if not camera_mgr.contains(curr_file):
            pose = find_pose(npz_files, i)
            pose = np.dot(pose, T)
            pose = pose.astype(np.float32)
            camera_mgr.add_camera(curr_file, pose)

    # create ray samplers
    ray_samplers = []
    # 1 for the initial event batch spoiling everything
    # max_winsize more for previous pose not getting into this trap too
    start_range = 0 if cycle else 1+max_winsize
    model_config = {'name': 'RecEVFlowNet', 'encoding': 'cnt', 'round_encoding': False, 'norm_input': False, 'num_bins': 2, 'base_num_channels': 32, 'kernel_size': 3, 'activations': ['relu', None], 'mask_output': True, 'spiking_neuron':None}
    estimator_model = RecEVFlowNet(model_config).to(0)
    model_dir = './PAEv3d/event_flow/mlruns/mlruns/mds/EVFlowNet/artifacts/model/data/model.pth'


    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir)
        estimator_model.load_state_dict(model_loaded.state_dict())
    estimator_model.eval()
    ray_samplers = []
    start_range = 0 if cycle else 1+max_winsize
    with torch.no_grad():
        for i in range(start_range, cam_cnt):
            index = str(i).zfill(4)
            try:
                intrinsics = parse_txt(intrinsic_files[0], (5, 4))
            except ValueError:
                intrinsics = parse_txt(intrinsic_files[0], (4, 4))
                intrinsics = intrinsics.astype(np.float32)
                # concat unity distortion coefficients
                intrinsics = np.concatenate((intrinsics, np.zeros((1,4), dtype=np.float32)), 0)

            if randomize_winlen:
                winsize = np.random.randint(1, max_winsize+1)
            else:
                winsize = max_winsize

            start_time = (i-winsize)/(cam_cnt-1)
            if start_time < 0:
                start_time += 1
            end_time = (i)/(cam_cnt-1)

            end = np.searchsorted(ts, end_time*ts.max())

            if win_constant_count != 0:
                # TODO: there could be a bug with windows in the start, e.g., end-win_constant_count<0
                #       please, check if the windows are correctly composed in that case
                start_time = ts[end-win_constant_count]/ts.max()

                if win_constant_count > end:
                    start_time = start_time - 1

                winsize = int(i-start_time*(cam_cnt-1))
                assert(winsize>0)
                start_time = (i-winsize)/(cam_cnt-1)

                if start_time < 0:
                    start_time += 1

            start = np.searchsorted(ts, start_time*ts.max())

            if start <= end:
                # normal case: take the interval between
                events = (xs[start:end], ys[start:end], ts[start:end], ps[start:end])
            else:
                # loop over case: compose tail with head events
                events = (np.concatenate((xs[start:], xs[:end])),
                        np.concatenate((ys[start:], ys[:end])),
                        np.concatenate((ts[start:], ts[:end])),
                        np.concatenate((ps[start:], ps[:end])),
                        )

            H, W = 480, 640

            prev_file = img_files[(i-winsize+len(img_files))%len(img_files)]
            curr_file = img_files[i]
            curr_mask = None

            opticalHelper = OpticalEstimationHelper(events, H, W)
            optical_flow = estimator_model(opticalHelper.event_voxel.to(0), opticalHelper.event_cnt.to(0))
            estimated_flow = optical_flow["flow"][-1]*opticalHelper.event_mask.to(0)
            estimated_flow = estimated_flow.detach().squeeze(0).cpu().permute(1,2,0).numpy().reshape((-1, 2))

            if win_constant_count != 0:
                print('cnt:', len(events[0]), 'request:', win_constant_count)
            ray_samplers.append(RaySamplerSingleEventStream(H=H, W=W, intrinsics=intrinsics,
                                                            events=events,
                                                            rgb_path=curr_file,
                                                            prev_rgb_path=prev_file,
                                                            estimated_flow= estimated_flow,
                                                            mask_path=curr_mask,
                                                            end_idx=i,
                                                            use_ray_jitter=use_ray_jitter,
                                                            is_colored=is_colored,
                                                            polarity_offset=polarity_offset,
                                                            is_rgb_only=is_rgb_only))

        logger.info('Split {}, # views: {}, # effective views: {}'.format(split, cam_cnt, len(ray_samplers)))
        del estimator_model
        torch.cuda.empty_cache()
        return ray_samplers
