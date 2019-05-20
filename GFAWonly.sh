#!/usr/bin/env bash
export PYTHONPATH="/home/base/GraphPath/BioRE-master/rl+PCNN/Joint"

for set in 0.5 0.7 1.0; do
    for rander in 55 83 5583; do
        CUDA_VISIBLE_DEVICES=0 python3 pure_GFAW.py \
            --total_iterations=500 \
            --use_replay_memory=0 \
            --train_pcnn=0 \
            --bfs_iteration=0 \
            --use_joint_model=0 \
            --pcnn_dataset_base="/data/base/Joint-Datasets" \
            --pcnn_dataset_name="FB15K-237+NYT/text" \
            --pcnn_max_epoch=250 \
            --base_output_dir="/data/base/Joint-output" \
            --gfaw_dataset_base="/data/base/baselines-data" \
            --gfaw_dataset="FB60K-"$set"_rev" \
            --load_model=0 \
            --load_pcnn_model=0 \
            --batch_size=64 \
            --hidden_size 100 --embedding_size 100 \
            --random_seed=$rander \
            --eval_every=125 
    done
done
