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

from gas_mlp_configs import ACDT_DEFAULT_CONFIG, ACDTTrainConfig
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all
# from osrl.common.net import DiagGaussianActor

from datetime import datetime
REAL_TIME = datetime.now()




class DiagGaussianActor(nn.Module):
    """
    torch.distributions implementation of an diagonal Gaussian policy.
    """

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = log_std.clamp(self.log_std_bounds[0],self.log_std_bounds[1])
        std = log_std.exp()
        return Normal(mu, std)

class Return(nn.Module):
    def __init__(self, num_states, num_actions, num_embeding, args, num_outputs=1):
        super(Return, self).__init__()
        self.hidden = args.LAYER_DIM
        

        self.embedding_s = nn.Linear(num_states,num_embeding)
        self.embedding_c = nn.Embedding(args.COST_MAX,num_embeding)
        self.embedding_r = nn.Embedding(args.REWARD_MAX,num_embeding)
        self.embedding_t = nn.Embedding(args.TIME_STEP_MAX,num_embeding)
        self.linear0 = nn.Linear(num_embeding*4, self.hidden)
        self.linear1 = nn.Linear(self.hidden, self.hidden*4)
        self.linear2 = nn.Linear(self.hidden*4, self.hidden)
        self.linear3 = nn.Linear(self.hidden, self.hidden*4)
        self.linear4 = nn.Linear(self.hidden*4, self.hidden)
        self.linear5 = nn.Linear(self.hidden, self.hidden*4)
        self.linear6 = nn.Linear(self.hidden*4, self.hidden)
        self.linear7 = nn.Linear(self.hidden, num_outputs)
        self.emb_drop = nn.Dropout(args.DROP_PROB)
        self.layer_norm = nn.LayerNorm(self.hidden)
    def forward(self, s, a, r, c, t):
        t = t.long()
        t_embed = self.embedding_t(t)
        c = c.long()
        c_embed = self.embedding_c(c)
        r = r.clamp(0).long()
        r_embed = self.embedding_r(r)

        s_embed = self.embedding_s(s)

        if c_embed.shape[0] == 1:
            c_embed = c_embed.view(-1)
        if r_embed.shape[0] == 1:
            r_embed = r_embed.view(-1)
        if t_embed.shape[0] == 1:
            t_embed = t_embed.view(-1)
        if s_embed.shape[0] == 1:
            s_embed = s_embed.view(-1)

        if a==None:
            x = torch.cat([s_embed,r_embed,c_embed, t_embed],dim=-1)
        else:
            x = torch.cat([s_embed,a,r_embed,c_embed, t_embed],dim=-1)
        # x = torch.cat([x,c],dim=-1)
        #change the net work
        x = F.gelu(self.linear0(x))
        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.emb_drop(x)
        x += residual
        x = F.gelu(x)

        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.linear3(x))
        x = self.linear4(x)
        x = self.emb_drop(x)
        x += residual
        x = F.gelu(x)

        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.linear5(x))
        x = self.linear6(x)
        x = self.emb_drop(x)
        x += residual
        x = F.gelu(x)

        # x = torch.softmax(self.linear5(x))
        x = self.linear7(x)
        return x

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, num_embeding,target_entropy, args):
        super(Actor, self).__init__()
        self.hidden = args.LAYER_DIM
        
        self.embedding_s = nn.Linear(num_states,num_embeding)
        self.embedding_c = nn.Embedding(args.COST_MAX,num_embeding)
        self.embedding_r = nn.Embedding(args.REWARD_MAX,num_embeding)
        self.embedding_cv = nn.Embedding(args.COST_MAX,num_embeding) # There may have bug for value network > reward max or cost max
        self.embedding_rv = nn.Embedding(args.REWARD_MAX,num_embeding) # There may have bug for value network > reward max or cost max
        self.embedding_t = nn.Embedding(args.TIME_STEP_MAX,num_embeding)
        self.linear0 = nn.Linear(num_embeding*6, self.hidden)
        self.linear1 = nn.Linear(self.hidden, self.hidden*4)
        self.linear2 = nn.Linear(self.hidden*4, self.hidden)
        self.linear3 = nn.Linear(self.hidden, self.hidden*4)
        self.linear4 = nn.Linear(self.hidden*4, self.hidden)
        self.linear5 = nn.Linear(self.hidden, self.hidden*4)
        self.linear6 = nn.Linear(self.hidden*4, self.hidden)
        self.linear7 = nn.Linear(self.hidden, self.hidden)
        self.dist = DiagGaussianActor(self.hidden, num_actions)

        self.log_temperature = torch.tensor(np.log(0.1))
        self.log_temperature.requires_grad = True
        self.target_entropy = target_entropy

        self.layer_norm = nn.LayerNorm(self.hidden)
        self.emb_drop = nn.Dropout(args.DROP_PROB)

        self.COST_MAX = args.COST_MAX
        self.REWARD_MAX = args.REWARD_MAX

    def forward(self, s, r, c, t, vc, vr):
        t = t.long()
        t_embed = self.embedding_t(t)

        s_embed = self.embedding_s(s)
        c = c.long()
        c_embed = self.embedding_c(c)
        r = r.long()
        r_embed = self.embedding_r(r)

        vc = torch.clamp(vc,0,self.COST_MAX-1)
        vc = vc.long()
        vc_embed = self.embedding_cv(vc)
        
        vr = torch.clamp(vr,0,self.REWARD_MAX-1)
        vr = vr.long()
        vr_embed = self.embedding_rv(vr)
        

        if s_embed.shape[0] == 1:
            s_embed = s_embed.view(-1)
        if c_embed.shape[0] == 1:
            c_embed = c_embed.view(-1)
        if r_embed.shape[0] == 1:
            r_embed = r_embed.view(-1)
        if vc_embed.shape[0] == 1:
            vc_embed = vc_embed.view(-1)
        if vr_embed.shape[0] == 1:
            vr_embed = vr_embed.view(-1)
        if t_embed.shape[0] == 1:
            t_embed = t_embed.view(-1)
        
        x = torch.cat([r_embed,c_embed, t_embed, vc_embed, vr_embed],dim=-1)
        ## x = self.emb_norm(x)
        # x = self.emb_drop(x)
        x = torch.cat([s_embed,x],dim=-1)

        x = F.gelu(self.linear0(x))
        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.emb_drop(x)
        x += residual
        x = F.gelu(x)

        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.linear3(x))
        x = self.linear4(x)
        x = self.emb_drop(x)
        x += residual
        x = F.gelu(x)

        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.linear5(x))
        x = self.linear6(x)
        x = self.emb_drop(x)
        x += residual
        x = F.gelu(x)
        
        # x = torch.softmax(self.linear5(x))
        x = self.linear7(x)
        
        dist = self.dist(x)
        return dist

    def temperature(self):
        return self.log_temperature.exp()


def train_ciql_model(VM,CM,AM,QM, VM_opt,CM_opt,AM_opt,QM_opt,lt_opt, VM_s,CM_s,AM_s, s, a, C_true,R_true,c_single,timestep,iql_tau, iql_tau_p, device, args:ACDTTrainConfig):
    for i in range(1):
        T_sim = torch.rand(timestep.shape).to(device)
        T_sim = T_sim**args.DIF_SHAPE*args.TIME_DIF*(args.TIME_STEP_MAX-timestep)+timestep
        T_sim = T_sim.clamp(0,args.TIME_STEP_MAX-1)
        T_sim = T_sim.long()
        T_sim_index = T_sim.view((-1,1))

        # T_sim = torch.rand(timestep.shape).to(device)
        # T_sim = T_sim**(1/DIF_SHAPE) * timestep
        # T_sim = T_sim.clamp(0,TIME_STEP_MAX-1)
        # T_sim = T_sim.long()
        # T_sim_index = T_sim.view((-1,1))

        CT_true = C_true.gather(1,T_sim_index).view(-1)
        RT_true = R_true.gather(1,T_sim_index).view(-1)

        C_sim = torch.rand(CT_true.shape).to(device)
        # C_sim = C_sim**args.DIF_SHAPE*args.COST_DIF*(args.COST_MAX-CT_true)*torch.sqrt((args.TIME_STEP_MAX-T_sim)/args.TIME_STEP_MAX)+CT_true
        C_sim = C_sim**args.DIF_SHAPE*args.COST_DIF*args.COST_MAX*torch.sqrt((args.TIME_STEP_MAX-T_sim)/args.TIME_STEP_MAX)+CT_true
        C_sim = C_sim.clamp(0,args.COST_MAX-1)
        C_sim = C_sim.long()

        R_sim = torch.rand(RT_true.shape).to(device)
        R_sim = R_sim*args.REWARD_DIF*2+(1-args.REWARD_DIF)
        R_sim = R_sim*RT_true
        R_sim = R_sim.clamp(0,args.REWARD_MAX-1)
        R_sim = R_sim.long()



        V_true = VM(s,None,R_sim,C_sim,T_sim).squeeze(dim=-1)
        adv = RT_true*args.REWARD_CONSERVATIVE-V_true
        loss_V = torch.mean(torch.abs(iql_tau-(adv.detach()<0).float())*adv**2)

        VM.eval()
        with torch.no_grad():
            V_true = VM(s,None,R_sim,C_sim,T_sim).squeeze(dim=-1)
            adv = RT_true-V_true    
        VM.train()

        VC_true = CM(s,None,R_sim,C_sim,T_sim).squeeze(dim=-1)
        adv_c = CT_true-VC_true
        adv_c_mask = CT_true-VC_true.clamp(0.01)
        loss_VC = torch.mean(torch.abs(iql_tau-(adv.detach()<0).float())*adv_c**2)
        # loss_VC_1 = torch.mean(torch.abs(iql_tau-(adv.detach()<0).float())*adv_c**2)
        # loss_VC_2 = torch.mean(torch.abs(iql_tau-(adv_c_mask.detach()>0).float())*adv_c**2)
        # loss_VC = (1-args.advc_weight) * loss_VC_1 + args.advc_weight * loss_VC_2
        
        CM.eval()
        with torch.no_grad():
            VC_true = CM(s,None,R_sim,C_sim,T_sim).squeeze(dim=-1)
        CM.train()
        
        policy = AM(s,R_sim,C_sim,T_sim, V_true, VC_true)

        log_likelihood = policy.log_prob(a).mean(dim=-1)
        entropy = policy.entropy().mean()
        entropy_reg = AM.temperature().detach()

        # VC_true = VC_true.detach()
        # Value_C_Q = torch.where(VC_true<=0.1,0,VC_true)

        loss_bc = torch.abs(iql_tau_p - (adv.detach() < 0).float()) * log_likelihood
        loss_bc = loss_bc.mean()
        # loss_bc = -(loss_bc + entropy_reg * entropy)

        # loss_bc_ = torch.abs(iql_tau_p - (adv_c_mask.detach() > 0).float()) * log_likelihood
        loss_bc_ = args.advc_w * (2*(adv_c_mask.detach() > 0).float() - 1) * log_likelihood
        loss_bc_ = loss_bc_.mean()

        # loss_cc = (c_single-1)*log_likelihood
        loss_cc = args.cs_w*(2 * c_single - 1*args.cost_scale)*log_likelihood
        loss_cc = loss_cc.mean()

        # loss_ac = -((1-args.advc_weight)*loss_bc + args.advc_weight*loss_bc_ +entropy_reg * entropy) + loss_cc
        loss_ac = -(loss_bc +entropy_reg * entropy) + loss_bc_ + loss_cc

        VM_opt.zero_grad()
        loss_V.backward()
        torch.nn.utils.clip_grad_norm_(VM.parameters(), args.clip_grad)
        VM_opt.step()
        VM_s.step()

        CM_opt.zero_grad()
        loss_VC.backward()
        torch.nn.utils.clip_grad_norm_(CM.parameters(), args.clip_grad)
        CM_opt.step()
        CM_s.step()

        AM_opt.zero_grad()
        loss_ac.backward()
        torch.nn.utils.clip_grad_norm_(AM.parameters(), args.clip_grad)
        AM_opt.step()
        AM_s.step()

        lt_opt.zero_grad()
        temperature_loss = (
            AM.temperature() * (entropy - AM.target_entropy).detach()
        )
        temperature_loss.backward()
        lt_opt.step()

    return float(loss_V.detach().cpu()), float(loss_VC.detach().cpu()),  float(loss_ac.detach().cpu()),

@torch.no_grad()
def eval_actor(
    env: gym.Env, args:ACDTTrainConfig, actor: nn.Module,vm: nn.Module,cm: nn.Module, max_action: float, device: str, n_episodes: int, seed: int, target_index: int
    ) -> np.ndarray:
    # env.seed(seed)
    actor.eval()
    vm.eval()
    cm.eval()
    episode_rewards = []
    episode_costs = []
    episode_lens = []
    for _ in range(n_episodes):
        temp_return = torch.Tensor([args.target_returns[target_index][0]]).to(device) * args.reward_scale + args.REWARD_MIN
        temp_constraint = torch.Tensor([args.target_returns[target_index][1]]).to(device)
        epi_len_tensor = torch.Tensor([0]).to(device)
        state, info = env.reset()

        state = torch.Tensor(state).to(device)

        epi_len = 0
        episode_reward = args.REWARD_MIN
        episode_cost = 0.0

        init_value = vm(state,None, temp_return, temp_constraint, epi_len_tensor)
        init_value_c = cm(state,None,temp_return, temp_constraint, epi_len_tensor)
        cost = 0
        for ij in range(args.episode_len):
            value = vm(state,None, temp_return, temp_constraint, epi_len_tensor).detach()
            value_c = cm(state,None,temp_return, temp_constraint, epi_len_tensor).detach()
            
            value = torch.clamp(value,0,args.REWARD_MAX-1)
            value_c = torch.clamp(value_c,0,args.COST_MAX-1)
            if temp_constraint < 0.1 * args.COST_MAX:
                action = actor(state,temp_return, temp_constraint/2, epi_len_tensor, value, value_c)
            else:
                action = actor(state,temp_return, temp_constraint, epi_len_tensor, value, value_c)
            action = action.mean
            # action = action.cpu().numpy()
            action = action.clamp(-max_action, max_action).cpu().numpy()
            
            state, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]

            epi_len_tensor += 1
            epi_len += 1
            temp_constraint -= cost
            temp_constraint = temp_constraint.clamp(0,args.COST_MAX-1)
            # temp_return -= max(reward,0)
            temp_return -= reward
            temp_return = temp_return.clamp(0,args.REWARD_MAX-1)

            state = torch.Tensor(state).to(device)

            episode_reward += reward
            episode_cost += cost

            if terminated or truncated:
                break
        # height_list = np.array(height_list)
        episode_rewards.append((episode_reward-args.REWARD_MIN)/args.reward_scale)
        episode_costs.append(episode_cost/args.cost_scale)
        episode_lens.append(epi_len)


    actor.train()
    vm.train()
    cm.train()
    print("Estimate Value:",init_value,init_value_c)
    return np.asarray(episode_rewards), np.asarray(episode_costs), np.asarray(episode_lens)

def process_dataset(dataset, args):
    dataset_new = [[],[],[],[],[],[]]
    time_step = []
    traj_num = 0
    for index, data in enumerate(dataset):
        # print(data['returns'].shape)
        if True:
            traj_num += 1
            for j in range(data['returns'].shape[0]):
                k = 0
                for key, value in data.items():
                    if key == "returns":
                        dataset_new[k].append(value[j]*args.reward_scale)
                    else:
                        dataset_new[k].append(value[j])
                    k+=1
            time_step += [i for i in range(j+1)]
    for i,j in enumerate(dataset_new):
        if i == 0 or i == 1:
            dataset_new[i] = np.vstack(dataset_new[i])
            dataset_new[i] = torch.Tensor(dataset_new[i]).to(args.device)
        else:
            dataset_new[i] = torch.Tensor(dataset_new[i]).to(args.device)
    try:
        dataset_new[4] = torch.Tensor(np.load(f"./examples/train/dataset/{args.task}_return_new.npy")).to(args.device)*args.reward_scale
        dataset_new[5] = torch.Tensor(np.load(f"./examples/train/dataset/{args.task}_cost_new.npy")).to(args.device)
    except:
        print("Fail to load Return data to GPU. Loading to CPU now.")
        dataset_new[4] = torch.Tensor(np.load(f"./examples/train/dataset/{args.task}_return_new.npy")).to(args.device_)*args.reward_scale
        dataset_new[5] = torch.Tensor(np.load(f"./examples/train/dataset/{args.task}_cost_new.npy")).to(args.device_)

    dataset_new.append(torch.Tensor(time_step).to(args.device))
    args.REWARD_MIN = round(np.ceil(np.abs(torch.min(dataset_new[4]).item())))
    args.REWARD_MAX = round(np.ceil(torch.max(dataset_new[4]).item())) + args.REWARD_MIN
    dataset_new[4] += args.REWARD_MIN
    print("get data:",traj_num, "trans num:",dataset_new[0].shape[0], "R max:",torch.max(dataset_new[4]), "R min:",torch.min(dataset_new[4]))
    assert torch.max(dataset_new[4])<args.REWARD_MAX and torch.min(dataset_new[4])>0
    return dataset_new

def good_dataset(dataset, args, threshold=450, threshold_c=[30,50]):
    dataset_new = [[],[],[],[],[],[],[]]
    zero_index = torch.where(dataset[6]==0)[0]
    zero_index_len = zero_index.shape[0]
    traj_num = 0
    for i in range(zero_index_len):
        if dataset[4][zero_index[i]][0] >= threshold*args.reward_scale + args.REWARD_MIN and dataset[5][zero_index[i]][0] >= threshold_c[0] and dataset[5][zero_index[i]][0] <= threshold_c[1]:
            traj_num += 1
            if i != zero_index_len-1:
                for j in range(7):
                    dataset_new[j] += dataset[j][zero_index[i]:zero_index[i+1]]
            else:
                for j in range(7):
                    dataset_new[j] += dataset[j][zero_index[i]:]
    for i,j in enumerate(dataset_new):
        if i in [2,3,6]:
            dataset_new[i] = torch.vstack(dataset_new[i]).view(-1)
        else:
            dataset_new[i] = torch.vstack(dataset_new[i])

    print("good data:",traj_num, "trans num:",dataset_new[0].shape[0])
    return dataset_new

def sample_data(dataset, batch_size):
    total_index = [i for i in range(dataset[0].shape[0])]
    sample_index = np.random.choice(total_index,batch_size,replace=False)
    return dataset[0][sample_index],dataset[1][sample_index],dataset[2][sample_index],dataset[3][sample_index],dataset[4][sample_index],dataset[5][sample_index],dataset[6][sample_index]


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
        # augment_percent=args.augment_percent,
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

    # print("-------------------------Start preprocessing data on return and cost return-----------------------------")
    # data_preprocess(dataset_origin.dataset,args)
    # print("-------------------------Finish preprocessing data on return and cost return-----------------------------")
    # exit()

    good_datasets = process_dataset(dataset_origin.dataset, args)
    good_datasets_0 = good_dataset(good_datasets, args, args.GOOD_R_THRESHOD[0], args.GOOD_C_THRESHOD[0])
    good_datasets_1 = good_dataset(good_datasets, args, args.GOOD_R_THRESHOD[1], args.GOOD_C_THRESHOD[1])
    good_datasets_2 = good_dataset(good_datasets, args, args.GOOD_R_THRESHOD[2], args.GOOD_C_THRESHOD[2])




    Value_model_R = Return(env.observation_space.shape[0],0, args.EMBED_DIM,args).to(args.device)
    Value_model_C = Return(env.observation_space.shape[0],0, args.EMBED_DIM,args).to(args.device)
    Value_model_Q_R = Return(env.observation_space.shape[0], env.action_space.shape[0], args.EMBED_DIM,args).to(args.device)
    Actor_model = Actor(env.observation_space.shape[0],env.action_space.shape[0], args.EMBED_DIM, -env.action_space.shape[0],args).to(args.device)


    Actor_optimizer   = torch.optim.AdamW(Actor_model.parameters(), lr=args.p_lr, weight_decay=args.weight_decay, betas=args.betas)
    Value_model_C_optimizer = torch.optim.AdamW(Value_model_C.parameters(), lr=args.vf_lr, weight_decay=args.weight_decay, betas=args.betas)
    Value_model_R_optimizer = torch.optim.AdamW(Value_model_R.parameters(), lr=args.vf_lr, weight_decay=args.weight_decay, betas=args.betas)
    Value_model_Q_R_optimizer = torch.optim.AdamW(Value_model_Q_R.parameters(), lr=args.vf_lr, weight_decay=args.weight_decay, betas=args.betas)
    log_temperature_optimizer = torch.optim.Adam([Actor_model.log_temperature],lr=1e-4,betas=[0.9, 0.999],)

    Actor_scheduler = torch.optim.lr_scheduler.LambdaLR(Actor_optimizer, lambda steps: min((steps + 1) / args.lr_warmup_steps, 1),)
    Value_model_R_scheduler = torch.optim.lr_scheduler.LambdaLR(Value_model_R_optimizer, lambda steps: min((steps + 1) / args.lr_warmup_steps, 1),)
    Value_model_C_scheduler = torch.optim.lr_scheduler.LambdaLR(Value_model_C_optimizer, lambda steps: min((steps + 1) / args.lr_warmup_steps, 1),)


    evaluations = []
    loss_lists = []
    for step in trange(args.update_steps, desc="Training"):
        loss_1, loss_2, loss_3 = 0, 0, 0,
        random_number = np.random.random()

        # good_batch = sample_data(good_datasets_0,args.batch_size)
        # if random_number < args.SAMPLE_PROB[0]:
        #     good_batch = sample_data(good_datasets,args.batch_size//2)
        #     good_batch_1 = sample_data(good_datasets_2,args.batch_size//2)
        #     good_batch = [torch.cat([good_batch[i],good_batch_1[i]],dim=0)for i in range(len(good_batch))]
        # elif random_number < args.SAMPLE_PROB[1]:
        #     good_batch = sample_data(good_datasets_0,args.batch_size//2)
        #     good_batch_1 = sample_data(good_datasets_2,args.batch_size//2)
        #     good_batch = [torch.cat([good_batch[i],good_batch_1[i]],dim=0)for i in range(len(good_batch))]
        # elif random_number < args.SAMPLE_PROB[2]:
        #     good_batch = sample_data(good_datasets_1,args.batch_size//2)
        #     good_batch_1 = sample_data(good_datasets_2,args.batch_size//2)
        #     good_batch = [torch.cat([good_batch[i],good_batch_1[i]],dim=0)for i in range(len(good_batch))]
        # else:
        #     good_batch = sample_data(good_datasets_2,args.batch_size)

        # good_batch = sample_data(good_datasets,args.batch_size)
        good_batch = sample_data(good_datasets,args.batch_size//4)
        good_batch_0 = sample_data(good_datasets_0,args.batch_size//4)
        good_batch_1 = sample_data(good_datasets_1,args.batch_size//4)
        good_batch_2 = sample_data(good_datasets_2,args.batch_size//4)
        good_batch = [torch.cat([good_batch[i],good_batch_0[i], good_batch_1[i], good_batch_2[i]],dim=0)for i in range(len(good_batch))]


        states, actions, rewards, costs, returns, costs_return, time_step = [
            b for b in good_batch
        ]
        if returns.device.type == 'cpu':
            returns = returns.to(args.device)
            costs_return = costs_return.to(args.device)
        loss_4, loss_5, loss_6 = train_ciql_model(Value_model_R,Value_model_C,Actor_model,Value_model_Q_R,\
                                                  Value_model_R_optimizer,Value_model_C_optimizer,Actor_optimizer,Value_model_Q_R_optimizer,log_temperature_optimizer,\
                                                  Value_model_R_scheduler, Value_model_C_scheduler, Actor_scheduler,\
                                                  states,actions,costs_return,returns, costs, time_step, args.iql_tau,args.iql_tau_p,args.device,args)
        if (step + 1)%args.eval_every==0:
            print("Loss:",loss_1,loss_2,loss_3,loss_4, loss_5, loss_6)
            temp_eva = []
            for target_index in range(len(args.target_returns)):
                test_return, test_cost, test_epi_len = eval_actor(env,args,Actor_model,Value_model_R,Value_model_C,env.action_space.high[0],args.device,10,args.seed, target_index)
                temp_eva += [float(np.average(test_return)), float(np.average(test_cost)), float(np.average(test_epi_len))]
            evaluations.append(temp_eva)
            loss_lists.append([loss_4, loss_5, loss_6])
            print("Average Value",temp_eva)
            # print(test_return)
            # print(test_cost)
            # print(test_epi_len)

            if (step + 1)%args.save_every==0:
                np.savetxt(f"/home/zliuft/CDT/examples/train/myresult/result_{args.task}_gas_final_iqltau_{str(round(args.iql_tau,2))}_{str(round(args.iql_tau_p,2))}_{REAL_TIME.month}_{REAL_TIME.day}_{REAL_TIME.hour}_{REAL_TIME.minute}.txt",np.array(evaluations))
                # np.savetxt(f"/home/zliuft/CDT/examples/train/myresult/loss_{args.task}_gas_final_iqltau_{str(round(args.iql_tau,2))}_{str(round(args.iql_tau_p,2))}_{REAL_TIME.month}_{REAL_TIME.day}_{REAL_TIME.hour}_{REAL_TIME.minute}.txt",np.array(loss_lists))
                # torch.save(Value_model_R.state_dict(),f"./examples/train/mlpt_value_model_r.net")
                # torch.save(Value_model_C.state_dict(),f"./examples/train/mlpt_value_model_c.net")
                # torch.save(Value_model_Q_R.state_dict(),f"./examples/train/mlpt_value_model_qr.net")
                # torch.save(Actor_model.state_dict(),f"./examples/train/mlpt_actor_model.net")


if __name__ == "__main__":
    train()
