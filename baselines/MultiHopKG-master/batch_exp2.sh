#!/usr/bin/env bash

#!/bin/bash

export PYTHONPATH=`pwd`
echo $PYTHONPATH

gpu=5

#source configs/fb60kcut.sh
#
#siz=data/FB60K-Cut/FB60K-0.1
#echo "$siz"
## echo "$siz" >> results.txt
#data_dir=$siz
#
#cmd="python3 -m src.experiments \
#    --data_dir $data_dir \
#    --process_data \
#    --model $model \
#    --bandwidth $bandwidth \
#    --entity_dim $entity_dim \
#    --relation_dim $relation_dim \
#    --history_dim $history_dim \
#    --history_num_layers $history_num_layers \
#    --num_rollouts $num_rollouts \
#    --num_rollout_steps $num_rollout_steps \
#    --bucket_interval $bucket_interval \
#    --num_epochs $num_epochs \
#    --num_wait_epochs $num_wait_epochs \
#    --num_peek_epochs $num_peek_epochs \
#    --batch_size $batch_size \
#    --train_batch_size $train_batch_size \
#    --dev_batch_size $dev_batch_size \
#    --margin $margin \
#    --learning_rate $learning_rate \
#    --baseline $baseline \
#    --grad_norm $grad_norm \
#    --emb_dropout_rate $emb_dropout_rate \
#    --ff_dropout_rate $ff_dropout_rate \
#    --action_dropout_rate $action_dropout_rate \
#    --action_dropout_anneal_interval $action_dropout_anneal_interval \
#    --beta $beta \
#    --beam_size $beam_size \
#    --num_paths_per_entity $num_paths_per_entity \
#    --gpu $gpu "
#
#echo "Executing $cmd"
#$cmd
#
#siz=data/FB60K-Cut/FB60K-0.2
#echo "$siz"
## echo "$siz" >> results.txt
#data_dir=$siz
#
#cmd="python3 -m src.experiments \
#    --data_dir $data_dir \
#    --process_data \
#    --model $model \
#    --bandwidth $bandwidth \
#    --entity_dim $entity_dim \
#    --relation_dim $relation_dim \
#    --history_dim $history_dim \
#    --history_num_layers $history_num_layers \
#    --num_rollouts $num_rollouts \
#    --num_rollout_steps $num_rollout_steps \
#    --bucket_interval $bucket_interval \
#    --num_epochs $num_epochs \
#    --num_wait_epochs $num_wait_epochs \
#    --num_peek_epochs $num_peek_epochs \
#    --batch_size $batch_size \
#    --train_batch_size $train_batch_size \
#    --dev_batch_size $dev_batch_size \
#    --margin $margin \
#    --learning_rate $learning_rate \
#    --baseline $baseline \
#    --grad_norm $grad_norm \
#    --emb_dropout_rate $emb_dropout_rate \
#    --ff_dropout_rate $ff_dropout_rate \
#    --action_dropout_rate $action_dropout_rate \
#    --action_dropout_anneal_interval $action_dropout_anneal_interval \
#    --beta $beta \
#    --beam_size $beam_size \
#    --num_paths_per_entity $num_paths_per_entity \
#    --gpu $gpu "
#
#echo "Executing $cmd"
#$cmd
#
#source configs/fb60kcut-complex.sh
#
#siz=data/FB60K-Cut/FB60K-0.1
#echo "$siz"
## echo "$siz" >> results.txt
#data_dir=$siz
#
#cmd="python3 -m src.experiments \
#    --data_dir $data_dir \
#    --train \
#    --model $model \
#    --entity_dim $entity_dim \
#    --relation_dim $relation_dim \
#    --num_rollouts $num_rollouts \
#    --bucket_interval $bucket_interval \
#    --num_epochs $num_epochs \
#    --num_wait_epochs $num_wait_epochs \
#    --batch_size $batch_size \
#    --train_batch_size $train_batch_size \
#    --dev_batch_size $dev_batch_size \
#    --margin $margin \
#    --learning_rate $learning_rate \
#    --grad_norm $grad_norm \
#    --emb_dropout_rate $emb_dropout_rate \
#    --beam_size $beam_size \
#    --gpu $gpu \
#    --add_reversed_training_edges \
#    --num_negative_samples $num_negative_samples \
#    --group_examples_by_query"
#
#echo "Executing $cmd"
#$cmd
#
#
#siz=data/FB60K-Cut/FB60K-0.2
#echo "$siz"
## echo "$siz" >> results.txt
#data_dir=$siz
#
#cmd="python3 -m src.experiments \
#    --data_dir $data_dir \
#    --train \
#    --model $model \
#    --entity_dim $entity_dim \
#    --relation_dim $relation_dim \
#    --num_rollouts $num_rollouts \
#    --bucket_interval $bucket_interval \
#    --num_epochs $num_epochs \
#    --num_wait_epochs $num_wait_epochs \
#    --batch_size $batch_size \
#    --train_batch_size $train_batch_size \
#    --dev_batch_size $dev_batch_size \
#    --margin $margin \
#    --learning_rate $learning_rate \
#    --grad_norm $grad_norm \
#    --emb_dropout_rate $emb_dropout_rate \
#    --beam_size $beam_size \
#    --gpu $gpu \
#    --add_reversed_training_edges \
#    --num_negative_samples $num_negative_samples \
#    --group_examples_by_query"
#
#echo "Executing $cmd"
#$cmd


source configs/fb60kcut-rs.sh

siz=data/FB60K-Cut/FB60K-0.1
echo "$siz"
data_dir=$siz
siz2=$(echo "$siz" | cut -d'/' -f 3)


distmult_state_dict_path="model/$siz2-distmult-xavier-200-200-0.003-0.3-0.1/model_best.tar"
complex_state_dict_path="model/$siz2-complex-RV-xavier-200-200-0.003-0.3-0.1/model_best.tar"
conve_state_dict_path="model/$siz2-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
checkpoint_path="model/$siz2-point.rs.complex-xavier-n/model_best.tar"

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    --train \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    --reward_shaping_threshold $reward_shaping_threshold \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    --distmult_state_dict_path $distmult_state_dict_path \
    --complex_state_dict_path $complex_state_dict_path \
    --conve_state_dict_path $conve_state_dict_path \
    --use_action_space_bucketing \
    --gpu $gpu "

echo "$siz" >> results.txt
echo "Executing $cmd"
$cmd

cmd="python3 -m src.experiments \
--data_dir $data_dir \
--inference \
--model $model \
--bandwidth $bandwidth \
--entity_dim $entity_dim \
--relation_dim $relation_dim \
--history_dim $history_dim \
--history_num_layers $history_num_layers \
--num_rollouts $num_rollouts \
--num_rollout_steps $num_rollout_steps \
--bucket_interval $bucket_interval \
--num_epochs $num_epochs \
--num_wait_epochs $num_wait_epochs \
--num_peek_epochs $num_peek_epochs \
--batch_size $batch_size \
--train_batch_size $train_batch_size \
--dev_batch_size $dev_batch_size \
--margin $margin \
--learning_rate $learning_rate \
--baseline $baseline \
--grad_norm $grad_norm \
--emb_dropout_rate $emb_dropout_rate \
--ff_dropout_rate $ff_dropout_rate \
--action_dropout_rate $action_dropout_rate \
--action_dropout_anneal_interval $action_dropout_anneal_interval \
--reward_shaping_threshold $reward_shaping_threshold \
--beta $beta \
--beam_size $beam_size \
--num_paths_per_entity $num_paths_per_entity \
--distmult_state_dict_path $distmult_state_dict_path \
--complex_state_dict_path $complex_state_dict_path \
--conve_state_dict_path $conve_state_dict_path \
--use_action_space_bucketing \
--checkpoint_path $checkpoint_path \
--gpu $gpu "

echo "Executing $cmd"
$cmd


siz=data/FB60K-Cut/FB60K-0.2
echo "$siz"
data_dir=$siz
siz2=$(echo "$siz" | cut -d'/' -f 3)

distmult_state_dict_path="model/$siz2-distmult-xavier-200-200-0.003-0.3-0.1/model_best.tar"
complex_state_dict_path="model/$siz2-complex-RV-xavier-200-200-0.003-0.3-0.1/model_best.tar"
conve_state_dict_path="model/$siz2-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
checkpoint_path="model/$siz2-point.rs.complex-xavier-n/model_best.tar"

cmd="python3 -m src.experiments \
    --data_dir $data_dir \
    --train \
    --model $model \
    --bandwidth $bandwidth \
    --entity_dim $entity_dim \
    --relation_dim $relation_dim \
    --history_dim $history_dim \
    --history_num_layers $history_num_layers \
    --num_rollouts $num_rollouts \
    --num_rollout_steps $num_rollout_steps \
    --bucket_interval $bucket_interval \
    --num_epochs $num_epochs \
    --num_wait_epochs $num_wait_epochs \
    --num_peek_epochs $num_peek_epochs \
    --batch_size $batch_size \
    --train_batch_size $train_batch_size \
    --dev_batch_size $dev_batch_size \
    --margin $margin \
    --learning_rate $learning_rate \
    --baseline $baseline \
    --grad_norm $grad_norm \
    --emb_dropout_rate $emb_dropout_rate \
    --ff_dropout_rate $ff_dropout_rate \
    --action_dropout_rate $action_dropout_rate \
    --action_dropout_anneal_interval $action_dropout_anneal_interval \
    --reward_shaping_threshold $reward_shaping_threshold \
    --beta $beta \
    --beam_size $beam_size \
    --num_paths_per_entity $num_paths_per_entity \
    --distmult_state_dict_path $distmult_state_dict_path \
    --complex_state_dict_path $complex_state_dict_path \
    --conve_state_dict_path $conve_state_dict_path \
    --use_action_space_bucketing
    --gpu $gpu "

echo "$siz" >> results.txt
echo "Executing $cmd"
$cmd

cmd="python3 -m src.experiments \
--data_dir $data_dir \
--inference \
--model $model \
--bandwidth $bandwidth \
--entity_dim $entity_dim \
--relation_dim $relation_dim \
--history_dim $history_dim \
--history_num_layers $history_num_layers \
--num_rollouts $num_rollouts \
--num_rollout_steps $num_rollout_steps \
--bucket_interval $bucket_interval \
--num_epochs $num_epochs \
--num_wait_epochs $num_wait_epochs \
--num_peek_epochs $num_peek_epochs \
--batch_size $batch_size \
--train_batch_size $train_batch_size \
--dev_batch_size $dev_batch_size \
--margin $margin \
--learning_rate $learning_rate \
--baseline $baseline \
--grad_norm $grad_norm \
--emb_dropout_rate $emb_dropout_rate \
--ff_dropout_rate $ff_dropout_rate \
--action_dropout_rate $action_dropout_rate \
--action_dropout_anneal_interval $action_dropout_anneal_interval \
--reward_shaping_threshold $reward_shaping_threshold \
--beta $beta \
--beam_size $beam_size \
--num_paths_per_entity $num_paths_per_entity \
--distmult_state_dict_path $distmult_state_dict_path \
--complex_state_dict_path $complex_state_dict_path \
--conve_state_dict_path $conve_state_dict_path \
--use_action_space_bucketing \
--checkpoint_path $checkpoint_path
--gpu $gpu "

echo "Executing $cmd"
$cmd



