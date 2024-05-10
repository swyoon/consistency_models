#! /bin/bash

#### Imagenet 64
# sampling from CD model
# OPENAI_LOGDIR=cd_imagenet64_lpips mpiexec -n 4 python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler onestep --model_path ../pretrained/cd_imagenet64_lpips.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000 --resblock_updown True --use_fp16 True --weight_schedule uniform


# Training ImageNet64 CD model
# mpiexec -n 8 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed \
#     --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt \
#     --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 \
#     --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 --image_size 64 --lr 0.000008 \
#     --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform \
#     --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/data


##### LSUN Bedroom

# Sampling

mpiexec -n 8 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_distillation --sampler onestep \
    --model_path /root/consistency/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 100 --resblock_updown True --use_fp16 True --weight_schedule uniform



# Training CD
# export OPENAI_LOGDIR="lsunbed_cd_lpips"
# mpiexec -n 4 python cm_train.py --training_mode consistency_distillation --sigma_max 80 --sigma_min 0.002 --target_ema_mode fixed \
#     --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 --loss_norm lpips --lr_anneal_steps 0 \
#     --teacher_model_path ../pretrained/edm_bedroom256_ema.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
#     --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 --global_batch_size 256 --image_size 256 \
#     --lr 0.00001 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform \
#     --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir ../datasets/bedroom_train_output_dir

