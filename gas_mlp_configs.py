from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field
@dataclass
class ACDTTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "ACDT"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    action_head_layers: int = 1
    seq_len: int = 1
    episode_len: int = 300
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    time_emb: bool = True
    # training params
    dataset: str = None
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 2048
    update_steps: int = 400000
    lr_warmup_steps: int = 500
    reward_scale: float = 1
    cost_scale: float = 1
    cs_w: float = 0
    num_workers: int = 8
    # evaluation params
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((330.0, 10), (390.0, 30), (480.0, 80))  # reward, cost
    eval_episodes: int = 10
    eval_every: int = 5000
    save_every: int = 20000
    # general params
    seed: int = 0
    threads: int = 6
    # augmentation param
    deg: int = 4
    pf_sample: bool = False
    beta: float = 1.0
    augment_percent: float = 0.2
    # maximum absolute value of reward for the augmented trajs
    max_reward: float = 600.0
    # minimum reward above the PF curve
    min_reward: float = 1.0
    # the max drecrease of ret between the associated traj
    # w.r.t the nearest pf traj
    max_rew_decrease: float = 100.0
    # model mode params
    use_rew: bool = True
    use_cost: bool = True
    cost_transform: bool = True
    cost_prefix: bool = False
    add_cost_feat: bool = False
    mul_cost_feat: bool = False
    cat_cost_feat: bool = False
    loss_cost_weight: float = 0.02
    loss_state_weight: float = 0
    cost_reverse: bool = False
    # pf only mode param
    pf_only: bool = False
    rmin: float = 300
    cost_bins: int = 60
    npb: int = 5
    cost_sample: bool = True
    linear: bool = True  # linear or inverse
    start_sampling: bool = False
    prob: float = 0.2
    stochastic: bool = True
    init_temperature: float = 0.1
    no_entropy: bool = False
    # random augmentation
    random_aug: float = 0
    aug_rmin: float = 400
    aug_rmax: float = 500
    aug_cmin: float = -2
    aug_cmax: float = 25
    cgap: float = 5
    rstd: float = 1
    cstd: float = 0.2


    task: str = "OfflineAntRun-v0"
    device: str = "cuda:1"
    device_: str = "cpu"
    cost_limit: int = 20
    return_limit: int = 550
    # tau: float = 0.005  # Target network update rate
    # beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.8 # Coefficient for asymmetric loss
    iql_tau_p: float = 0.8  # Coefficient for asymmetric loss
    # constraint: float = cost_limit * (1 - c_discount**episode_len) / (1 - c_discount) / episode_len * cost_scale  # Conctraint
    # constraint: float = cost_limit  # Conctraint
    # iql_deterministic: bool = True  # Use deterministic actor
    # normalize: bool = True  # Normalize states
    # normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 1e-3  # V function learning rate
    p_lr: float = 1e-3  # Critic learning rate
    advc_w: float = 0.0


    DROP_PROB: float = 0.00
    COST_MAX: int = 301
    EMBED_DIM: int = 256
    LAYER_DIM: int = 512
    REWARD_MIN: int = 0
    REWARD_MAX: int = 0
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = 500

    GOOD_R_THRESHOD : List[float] = field(default=[450, 400, 300], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[61,90],[31,60],[0,30]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTCarCircleConfig(ACDTTrainConfig):

    # model params
    seq_len: int = 1
    episode_len: int = 300
    # training params
    task: str = "OfflineCarCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((360.0, 10), (370.0, 20), (410.0, 30), (480.0, 70), (480.0, 80), (480.0, 90))  # reward, cost
    # augmentation param
    deg: int = 4
    max_reward: float = 600.0
    max_rew_decrease: float = 100.0
    device: str = "cuda:0"

    advc_w: float = 0

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 101
    EMBED_DIM: int = 128
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[400, 350, 200], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[61,90],[31,60],[0,30]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)


@dataclass
class ACDTAntRunConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 200
    # training params
    task: str = "OfflineAntRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((600.0, 15), (650.0, 30), (690.0, 45), (750.0, 105), (800.0, 120), (820.0, 135))
                        #   ...] = ((400.0, 0),(450.0, 0), (500.0, 0),(550.0, 0),(600.0, 0),)
    # augmentation param
    deg: int = 3
    max_reward: float = 1000.0
    max_rew_decrease: float = 150
    device: str = "cuda:0"

    cs_w: float = 0.1
    advc_w: float = 0.1
    vf_lr: float = 1e-3  # V function learning rate
    p_lr: float = 1e-3  # Critic learning rate

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 151
    EMBED_DIM: int = 64
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.2 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[600, 550, 400], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[91,135],[16,90],[0,15]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTDroneRunConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 200
    # training params
    task: str = "OfflineDroneRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((400.0, 14), (420.0, 28), (450.0, 42), (480.0, 56), (520.0, 70), (550.0, 84), (600.0, 98), (620.0, 112), (640.0, 126))
    # augmentation param
    deg: int = 1
    max_reward: float = 700.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"

    advc_w: float = 0

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 141
    EMBED_DIM: int = 128
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[500, 400, 350], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[85,126],[43,84],[0,42]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTDroneCircleConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 300
    # training params
    task: str = "OfflineDroneCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((600.0, 10), (650.0, 20), (700.0, 30), (750.0, 40), (790.0, 50), (830.0, 60), (830.0, 70), (850.0, 80), (870.0, 90))
    # augmentation param
    deg: int = 1
    max_reward: float = 1000.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    # device: str = "cuda:5"

    advc_w: float = 1

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 101
    EMBED_DIM: int = 128
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[700, 600, 550], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[61,90],[31,60],[0,30]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTCarRunConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 200
    # training params
    task: str = "OfflineCarRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((570.0, 4), (570.0, 8), (570.0, 12), (570.0, 28), (570.0, 32), (570.0, 36))
    # augmentation param
    deg: int = 0
    max_reward: float = 600.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:5"

    advc_w: float = 1

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 41
    EMBED_DIM: int = 128
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[550, 550, 550], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[25,36],[13,24],[0,12]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTAntCircleConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 500
    # training params
    task: str = "OfflineAntCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((200.0, 20), (200.0, 40), (240.0, 60), (320.0, 140), (350.0, 160), (400.0, 180))
                        #   ...] = ((120.0, 20), (140.0, 20),(160.0, 20),(180.0, 20),(200.0, 20),(220.0, 20),)
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"

    cs_w: float = 0.1
    advc_w: float = 0.1
    vf_lr: float = 1e-3  # V function learning rate
    p_lr: float = 1e-3  # Critic learning rate

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 201
    EMBED_DIM: int = 64
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[250, 200, 100], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[121,180],[21,120],[0,20]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.2, 0.3, 0.4, 1], is_mutable=True)

@dataclass
class ACDTBallRunConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 100
    # training params
    task: str = "OfflineBallRun-v0"
    target_returns: Tuple[Tuple[float, ...],
                        #   ...] = ((340.0, 16), (360.0, 16), (380.0, 16), (400.0, 16), (420.0, 16), (440.0, 16), (460.0, 16), (500.0, 16))
                          ...] = ((420.0, 8), (430.0, 16), (480.0, 24), (580.0, 32), (720.0, 40), (820.0, 48), (920.0, 56), (1100.0, 64), (1200.0, 72))
    # augmentation param
    deg: int = 2
    max_reward: float = 1400.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    device: str = "cuda:1"

    advc_w: float = 1

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 81
    EMBED_DIM: int = 128
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[400, 350, 300], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[49,72],[25,48],[0,24]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTBallCircleConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 1
    episode_len: int = 200
    # training params
    task: str = "OfflineBallCircle-v0"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((500.0, 8), (600.0, 16), (690.0, 24), (800.0, 56), (810.0, 64), (820.0, 72))
    # augmentation param
    deg: int = 2
    max_reward: float = 1000.0
    max_rew_decrease: float = 200
    min_reward: float = 1
    device: str = "cuda:2"

    advc_w: float = 1

    #dataset parameter
    reward_scale: float = 1
    cost_scale: float = 1
    DROP_PROB: float = 0.05
    COST_MAX: int = 81
    EMBED_DIM: int = 128
    LAYER_DIM: int = 256
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = episode_len+1

    GOOD_R_THRESHOD : List[float] = field(default=[600, 500, 400], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[49,72],[25,48],[0,24]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTCarButton1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((35.0, 20), (35.0, 40), (35.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 45.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class ACDTCarButton2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 50.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class ACDTCarCircle1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((20.0, 20), (22.5, 40), (25.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 30.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class ACDTCarCircle2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((20.0, 20), (21.0, 40), (22.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 30.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:0"


@dataclass
class ACDTCarGoal1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 50.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class ACDTCarGoal2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class ACDTCarPush1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((15.0, 20), (15.0, 40), (15.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 20.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class ACDTCarPush2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((12.0, 20), (12.0, 40), (12.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 15.0
    max_rew_decrease: float = 3
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class ACDTPointButton1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 60), (30.0, 80), (30.0, 100), (30.0, 120), (30.0, 140), (30.0, 160), (30.0, 180))
    # augmentation param
    deg: int = 0
    max_reward: float = 45.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTPointButton2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 50.0
    max_rew_decrease: float = 10
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTPointCircle1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((50.0, 20), (52.5, 40), (55.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 65.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTPointCircle2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 500
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((36.0, 28), (41, 55), (42.0, 90), (43.0, 110), (43, 140), (44.0, 170), (45.0, 200), (47, 240), (48.0, 270))
    # augmentation param
    deg: int = 1
    max_reward: float = 55.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:5"

    DROP_PROB: float = 0.00
    COST_MAX: int = 301
    EMBED_DIM: int = 256
    LAYER_DIM: int = 512
    REWARD_MIN: int = 163
    REWARD_MAX: int = 55
    REWARD_DIF: float = 0.1 #reward 上下浮动 20%
    REWARD_CONSERVATIVE: float = 1
    # COST_CONSERVATIVE = 0.95
    DIF_SHAPE: int = 1
    COST_DIF: float = 0.3 #cost随机数从0-0.5, 0-1表示从cost到cost max, 0-0.5则只能到一半
    TIME_DIF: float = 0.3
    TIME_STEP_MAX: int = 500

    GOOD_R_THRESHOD : List[float] = field(default=[40,       40,    30], is_mutable=True)
    GOOD_C_THRESHOD : List[float] = field(default=[[201,300],[101,200],[0,100]], is_mutable=True)
    SAMPLE_PROB     : List[float] = field(default=[0.4, 0.6, 0.8, 1], is_mutable=True)

@dataclass
class ACDTPointGoal1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTPointGoal2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 35.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTPointPush1Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((15.0, 20), (15.0, 40), (15.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 20.0
    max_rew_decrease: float = 5
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTPointPush2Config(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    target_returns: Tuple[Tuple[float, ...], ...] = ((12.0, 20), (12.0, 40), (12.0, 80))
    # augmentation param
    deg: int = 0
    max_reward: float = 15.0
    max_rew_decrease: float = 3
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTAntVelocityConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((2800.0, 20), (2800.0, 40), (2800.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3000.0
    max_rew_decrease: float = 500
    min_reward: float = 1
    device: str = "cuda:1"


@dataclass
class ACDTHalfCheetahVelocityConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((3000.0, 20), (3000.0, 40), (3000.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3000.0
    max_rew_decrease: float = 500
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTHopperVelocityConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((1750.0, 25), (1750.0, 50), (1750.0, 75), (1750.0, 100), (1750.0, 125), (1750.0, 80), (1750.0, 20), (1750.0, 40), (1750.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 2000.0
    max_rew_decrease: float = 300
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTSwimmerVelocityConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((160.0, 20), (160.0, 40), (160.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 250.0
    max_rew_decrease: float = 50
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTWalker2dVelocityConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((2800.0, 20), (2800.0, 40), (2800.0, 80))
    # augmentation param
    deg: int = 1
    max_reward: float = 3600.0
    max_rew_decrease: float = 800
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTEasySparseConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTEasyMeanConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTEasyDenseConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 2
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTMediumSparseConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (300.0, 20), (300.0, 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:3"


@dataclass
class ACDTMediumMeanConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (300.0, 20), (300.0, 40))
    # augmentation param
    deg: int = 0
    max_reward: float = 300.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTMediumDenseConfig(ACDTTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class ACDTHardSparseConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTHardMeanConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


@dataclass
class ACDTHardDenseConfig(ACDTTrainConfig):
    # model params
    seq_len: int = 10
    episode_len: int = 1000
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    update_steps: int = 200_000
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((300.0, 10), (350.0, 20), (400.0, 40))
    # augmentation param
    deg: int = 1
    max_reward: float = 500.0
    max_rew_decrease: float = 100
    min_reward: float = 1
    device: str = "cuda:2"


ACDT_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": ACDTCarCircleConfig,
    "OfflineAntRun-v0": ACDTAntRunConfig,
    "OfflineDroneRun-v0": ACDTDroneRunConfig,
    "OfflineDroneCircle-v0": ACDTDroneCircleConfig,
    "OfflineCarRun-v0": ACDTCarRunConfig,
    "OfflineAntCircle-v0": ACDTAntCircleConfig,
    "OfflineBallCircle-v0": ACDTBallCircleConfig,
    "OfflineBallRun-v0": ACDTBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": ACDTCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": ACDTCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": ACDTCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": ACDTCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": ACDTCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": ACDTCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": ACDTCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": ACDTCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": ACDTPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": ACDTPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": ACDTPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": ACDTPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": ACDTPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": ACDTPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": ACDTPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": ACDTPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": ACDTAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": ACDTHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": ACDTHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": ACDTSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": ACDTWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": ACDTEasySparseConfig,
    "OfflineMetadrive-easymean-v0": ACDTEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": ACDTEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": ACDTMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": ACDTMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": ACDTMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": ACDTHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": ACDTHardMeanConfig,
    "OfflineMetadrive-harddense-v0": ACDTHardDenseConfig
}
