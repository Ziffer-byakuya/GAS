import os
import uuid
import types
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torch.distributions.normal import Normal

from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from mlpt_configs_dataset import ACDT_DEFAULT_CONFIG, ACDTTrainConfig
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all
# from osrl.common.net import DiagGaussianActor

from datetime import datetime
REAL_TIME = datetime.now()

# COST_MAX = 151
TIME_STEP_MAX = 200

def data_preprocess(dataset,args):
    dataset_new = [[],[],[],[],[],[]]
    traj_num = 0
    for index, data in enumerate(dataset):
        traj_num += 1
        return_list = []
        cost_list = []
        for j in range(data['returns'].shape[0]):
            k = 0
            for key, value in data.items():
                if key == "returns":
                    return_list.append(value[j]*args.reward_scale)
                elif key == "cost_returns":
                    cost_list.append(value[j]*args.cost_scale)
                # elif key =="actions":
                #     print(value[j])
                k+=1
        traj_len = j+1
        if traj_len != TIME_STEP_MAX:
            print("Early stop trajecory:", traj_len)
        return_time_list = []
        cost_time_list = []
        for l_index, l in enumerate(return_list):
            c = cost_list[l_index]
            temp_return_list = []
            temp_cost_list = []
            for m in range(TIME_STEP_MAX):
                if m <= l_index:
                    # temp_return_list.append(0)
                    temp_return_list.append(l)
                    temp_cost_list.append(c)
                    # temp_return_list.append(0)
                    # temp_cost_list.append(2000)
                elif m >= traj_len:
                    temp_return_list.append(0)
                    temp_cost_list.append(0)
                else: 
                    temp_return_list.append(l-return_list[l_index-m])
                    temp_cost_list.append(c-cost_list[l_index-m])
            # print(len(temp_return_list))
            return_time_list.append(temp_return_list)
            cost_time_list.append(temp_cost_list)
        dataset_new[4] += return_time_list
        dataset_new[5] += cost_time_list


    dataset_new[4] = np.array(dataset_new[4])
    dataset_new[5] = np.array(dataset_new[5])
        
    # dataset_new = np.hstack(dataset_new)
    print("data preprocess:",traj_num, "trans num:",dataset_new[4].shape[0])
    np.save(f"./examples/train/dataset/{args.task}_return_new.npy",dataset_new[4])
    np.save(f"./examples/train/dataset/{args.task}_cost_new.npy",dataset_new[5])
    # np.save(f"./examples/train/dataset/{args.task}_return_conservative.npy",dataset_new[4])
    # np.save(f"./examples/train/dataset/{args.task}_cost_conservative.npy",dataset_new[5])

# def data_preprocess(dataset,args):
#     dataset_new = [[],[],[],[],[],[]]
#     traj_num = 0
#     for index, data in enumerate(dataset):
#         traj_num += 1
#         return_list = []
#         cost_list = []
#         for j in range(data['returns'].shape[0]):
#             k = 0
#             for key, value in data.items():
#                 if key == "returns":
#                     return_list.append(value[j]*args.reward_scale)
#                 elif key == "cost_returns":
#                     cost_list.append(value[j]*args.cost_scale)
#                 # elif key =="actions":
#                 #     print(value[j])
#                 k+=1
#         traj_len = j+1
#         return_time_list = []
#         cost_time_list = []
#         for l_index, l in enumerate(return_list):
#             c = cost_list[l_index]
#             temp_return_list = []
#             temp_cost_list = []
#             for m in range(TIME_STEP_MAX):
#                 if m < TIME_STEP_MAX-traj_len+l_index:
#                     # temp_return_list.append(0)
#                     temp_return_list.append(l)
#                     temp_cost_list.append(c)
#                     # temp_return_list.append(0)
#                     # temp_cost_list.append(2000)
#                 elif m == TIME_STEP_MAX-traj_len+l_index:
#                     temp_return_list.append(l)
#                     temp_cost_list.append(c)
#                 else: 
#                     temp_return_list.append(l-return_list[l_index+TIME_STEP_MAX-traj_len-m])
#                     temp_cost_list.append(c-cost_list[l_index+TIME_STEP_MAX-traj_len-m])
#             # print(len(temp_return_list))
#             return_time_list.append(temp_return_list)
#             cost_time_list.append(temp_cost_list)
#         dataset_new[4] += return_time_list
#         dataset_new[5] += cost_time_list


#     dataset_new[4] = np.array(dataset_new[4])
#     dataset_new[5] = np.array(dataset_new[5])
        
#     # dataset_new = np.hstack(dataset_new)
#     print("data preprocess:",traj_num, "trans num:",dataset_new[4].shape[0])
#     np.save(f"./examples/train/dataset/{args.task}_return_new.npy",dataset_new[4])
#     np.save(f"./examples/train/dataset/{args.task}_cost_new.npy",dataset_new[5])
#     # np.save(f"./examples/train/dataset/{args.task}_return_conservative.npy",dataset_new[4])
#     # np.save(f"./examples/train/dataset/{args.task}_cost_conservative.npy",dataset_new[5])

def data_preprocess_augmentation(dataset,args):
    dataset_new = [[],[],[],[],[],[]]
    traj_num = 0
    for index, data in enumerate(dataset):
        traj_num += 1
        return_list = []
        cost_list = []
        for j in range(data['returns'].shape[0]):
            k = 0
            for key, value in data.items():
                if key == "returns":
                    return_list.append(value[j]*args.reward_scale)
                elif key == "cost_returns":
                    cost_list.append(value[j]*args.reward_scale)
                # elif key =="actions":
                #     print(value[j])
                k+=1
        traj_len = j+1
        return_time_list = []
        cost_time_list = []
        for l_index, l in enumerate(return_list):
            c = cost_list[l_index]
            temp_return_list = []
            temp_cost_list = []
            for m in range(TIME_STEP_MAX):
                if m < TIME_STEP_MAX-traj_len+l_index:
                    # temp_return_list.append(0)
                    temp_return_list.append(l)
                    temp_cost_list.append(c)
                    # temp_return_list.append(0)
                    # temp_cost_list.append(2000)
                elif m == TIME_STEP_MAX-traj_len+l_index:
                    temp_return_list.append(l)
                    temp_cost_list.append(c)
                else: 
                    temp_return_list.append(l-return_list[l_index+TIME_STEP_MAX-traj_len-m])
                    temp_cost_list.append(c-cost_list[l_index+TIME_STEP_MAX-traj_len-m])
            # print(len(temp_return_list))
            return_time_list.append(temp_return_list)
            cost_time_list.append(temp_cost_list)
        dataset_new[4] += return_time_list
        dataset_new[5] += cost_time_list


    dataset_new[4] = np.array(dataset_new[4])
    dataset_new[5] = np.array(dataset_new[5])
        
    # dataset_new = np.hstack(dataset_new)
    print("data preprocess:",traj_num, "trans num:",dataset_new[4].shape[0])
    np.save(f"./examples/train/dataset/{args.task}_return_aug.npy",dataset_new[4])
    np.save(f"./examples/train/dataset/{args.task}_cost_aug.npy",dataset_new[5])
    # np.save(f"./examples/train/dataset/{args.task}_return_conservative.npy",dataset_new[4])
    # np.save(f"./examples/train/dataset/{args.task}_cost_conservative.npy",dataset_new[5])

@pyrallis.wrap()
def train(args: ACDTTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(ACDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    cfg = asdict(ACDT_DEFAULT_CONFIG[args.task]())
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    default_cfg = asdict(ACDT_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    # if "Metadrive" in args.task:
    #     import gym
    env = gym.make(args.task)

    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                args.outliers_percent,
                                args.noise_scale,
                                args.inpaint_ranges,
                                args.epsilon,
                                args.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)

    # wrapper
    env = wrap_env(
        env=env,
        reward_scale=args.reward_scale,
    )
    env = OfflineEnvWrapper(env)

    # def checkpoint_fn():
    #     return {"model_state": model.state_dict()}

    # logger.setup_checkpoint_fn(checkpoint_fn)

    ct = lambda x: 70 - x if args.linear else 1 / (x + 10)

    dataset_origin = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        deg=args.deg,
        pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease,
        beta=args.beta,
        augment_percent=0,
        cost_reverse=args.cost_reverse,
        max_reward=args.max_reward,
        min_reward=args.min_reward,
        pf_only=args.pf_only,
        rmin=args.rmin,
        cost_bins=args.cost_bins,
        npb=args.npb,
        cost_sample=args.cost_sample,
        cost_transform=ct,
        start_sampling=args.start_sampling,
        prob=args.prob,
        random_aug=args.random_aug,
        aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax,
        aug_cmin=args.aug_cmin,
        aug_cmax=args.aug_cmax,
        cgap=args.cgap,
        rstd=args.rstd,
        cstd=args.cstd,
    )

    dataset = SequenceDataset(
        data,
        seq_len=args.seq_len,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        deg=args.deg,
        pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease,
        beta=args.beta,
        augment_percent=args.augment_percent,
        cost_reverse=args.cost_reverse,
        max_reward=args.max_reward,
        min_reward=args.min_reward,
        pf_only=args.pf_only,
        rmin=args.rmin,
        cost_bins=args.cost_bins,
        npb=args.npb,
        cost_sample=args.cost_sample,
        cost_transform=ct,
        start_sampling=args.start_sampling,
        prob=args.prob,
        random_aug=args.random_aug,
        aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax,
        aug_cmin=args.aug_cmin,
        aug_cmax=args.aug_cmax,
        cgap=args.cgap,
        rstd=args.rstd,
        cstd=args.cstd,
    )

    print("-------------------------Start preprocessing data on return and cost return-----------------------------")
    data_preprocess(dataset_origin.dataset,args)
    # data_preprocess_0(dataset_origin.dataset,args)
    # data_preprocess_augmentation(dataset.dataset,args)
    print("-------------------------Finish preprocessing data on return and cost return-----------------------------")
    exit()



if __name__ == "__main__":
    train()
