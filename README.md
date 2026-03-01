# GAS
We first prepared an initial version for reference. The code will be continually organized and updated.

## Data Augmentation
Please run `data_preprocess.py` to generate the augmented data in the dataset folder.
1. Change `mlpt_configs_dataset.py` to change the task.
2. Change the ``TIME_STEP_MAX`` parameter according to the task. (We will automate it in the future.)
3. Keep the reward scale and cost scale as 1.

## GAS training
Please run `train_gas_mlp.py` to train GAS.
1. It is recommended to keep the reward scale and cost scale as 1.
2. In GAS, we use `nn.embedding` to process return-to-goes for stability.
   1. Since `nn.embedding` can only process integers, the reward scale may need to be increased to make single-step rewards significant.
   2. Or you can change `nn.embedding` to `nn.linear`.
   

