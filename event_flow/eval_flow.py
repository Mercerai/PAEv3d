import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import FWL, RSAT, AEE
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
from utils.iwe import compute_pol_iwe
from utils.utils import load_model, create_model_dir
from utils.mlflow import log_config, log_results
from utils.visualization import Visualization, vis_activity


def test(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    run = mlflow.get_run(args.runid)
    config = config_parser.merge_configs(run.data.params)

    if not args.debug:
        # create directory for inference results
        path_results = create_model_dir(args.path_results, args.runid)
        # store validation settings
        eval_id = log_config(path_results, args.runid, config)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id, path_results=path_results)

    # model initialization and settings
    model = LIFFireNet(config["model"]).to(device)
    model = load_model(args.runid, model, device)
    model.eval()

    # # validation metric
    # criteria = []
    # if "metrics" in config.keys():
    #     for metric in config["metrics"]["name"]:
    #         criteria.append(eval(metric)(config, device, flow_scaling=config["metrics"]["flow_scaling"]))

    # data loader
    data = H5Loader(config, config["model"]["num_bins"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # inference loop
    idx_AEE = 0
    val_results = {}
    end_test = False
    activity_log = None
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                if data.new_seq:
                    data.new_seq = False
                    activity_log = None
                    model.reset_states()

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                x = model(
                    inputs["event_voxel"].to(device), inputs["event_cnt"].to(device), log=config["vis"]["activity"]
                )

                flow_map = x["flow"][-1]

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask:
                    flow_vis *= inputs["event_mask"].to(device)

                # image of warped events
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    inputs["event_list"].to(device),
                    config["loader"]["resolution"],
                    inputs["event_list_pol_mask"][:, :, 0:1].to(device),
                    inputs["event_list_pol_mask"][:, :, 1:2].to(device),
                    flow_scaling=config["metrics"]["flow_scaling"],
                    round_idx=True,
                )
            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="mlflow run")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites mlflow settings",
    )
    parser.add_argument(
        "--path_mlflow",
        default="",
        help="location of the mlflow ui",
    )
    parser.add_argument("--path_results", default="results_inference/")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="don't save stuff",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
