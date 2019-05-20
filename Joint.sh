#!/usr/bin/env bash
export PYTHONPATH=.
#
#for rander in 55 83 5583; do
#    for set in 0.2 0.3 0.4 0.5 0.7 1.0; do
#        echo random_seed=$rander
#        echo dataset=cutoff-$set-PCNN-0.0
#    done
#done

#CUDA_VISIBLE_DEVICES=2,3 python3 joint_trainer.py \
#            --total_iterations=400 \
#            --use_replay_memory=1 \
#            --train_pcnn=1 \
#            --bfs_iteration=200 \
#            --use_joint_model=1 \
#            --pcnn_dataset_base="./datasets" \
#            --pcnn_dataset_name="text" \
#            --pcnn_max_epoch=250 \
#            --base_output_dir="./output" \
#            --gfaw_dataset_base="./datasets" \
#            --gfaw_dataset="FB60K-1.0_rev" \
#            --load_model=0 \
#            --load_pcnn_model=1 \
#            --batch_size=64 \
#            --hidden_size 100 --embedding_size 100 \
#            --random_seed=5583 \
#            --eval_every=100 \
#            --model_load_dir="./model"

CUDA_VISIBLE_DEVICES=1,0 python3 joint_trainer.py \
            --total_iterations=200 \
            --use_replay_memory=1 \
            --train_pcnn=1 \
            --bfs_iteration=0 \
            --use_joint_model=1 \
            --pcnn_dataset_base="./datasets" \
            --pcnn_dataset_name="text" \
            --pcnn_max_epoch=250 \
            --base_output_dir="./output" \
            --gfaw_dataset_base="./datasets" \
            --gfaw_dataset="FB60K-0.7_rev" \
            --load_model=0 \
            --load_pcnn_model=1 \
            --batch_size=64 \
            --hidden_size 100 --embedding_size 100 \
            --random_seed=5583 \
            --eval_every=100 \
            --model_load_dir=""

