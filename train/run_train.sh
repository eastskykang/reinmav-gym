#!/usr/bin/env bash

# arguments

#--env=walking
#--save_video_interval=30000
#--save_video_length=1000
#--reward_scale=1e-3
#--noise_mag=0.001
#--action_scale=1.
#--ent_coef=0.
#--num_hidden=32
#--num_layers=3
#--value_network=copy

python ../gym_reinmav/example/train_hovering.py \
--env=DRLMacaBlimpAisle-v0 \
--save_interval=200 \
--save_video_interval=50000 \
--save_video_length=1000 \
--reward_scale=0.01 \
--noise_std=0.3 \
--action_scale=1.0 \
--ppo_ent_coef=0.0 \
--ppo_lambda=0.95 \
--ppo_gamma=0.999 \
--num_hidden=256 \
--num_layers=2 \
--network_type=cnn \
--last_layer_activation \
--value_network=copy \
--num_step_per_update=512 \
--num_minibatch=2 \
--init_weight=0.001 \
--seed=42 \
--num_timesteps=200000000 \
--num_epoch=10 \
--lr=0.00005 \
--step_decimation=10