from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
import time
from pprint import pprint


def read_options():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_input_dir", default="", type=str)
    parser.add_argument("--input_file", default="train.txt", type=str)
    parser.add_argument("--create_vocab", default=0, type=int)
    parser.add_argument("--vocab_dir", default="", type=str)
    parser.add_argument("--max_num_actions", default=200, type=int)     # default=200
    parser.add_argument("--path_length", default=3, type=int)
    parser.add_argument("--hidden_size", default=100, type=int)
    parser.add_argument("--embedding_size", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--beta", default=1e-2, type=float)
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--log_dir", default="./logs/", type=str)
    parser.add_argument("--log_file_name", default="reward.txt", type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--num_rollouts", default=20, type=int)
    parser.add_argument("--test_rollouts", default=100, type=int)
    parser.add_argument("--LSTM_layers", default=1, type=int)
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--total_iterations", default=500, type=int)  # default=2000
    parser.add_argument("--Lambda", default=0.01, type=float)
    parser.add_argument("--pool", default="max", type=str)
    parser.add_argument("--eval_every", default=50, type=int)  # default=100
    parser.add_argument("--use_entity_embeddings", default=1, type=int)
    parser.add_argument("--train_entity_embeddings", default=1, type=int)   # 1
    parser.add_argument("--train_relation_embeddings", default=1, type=int)     # 1
    parser.add_argument("--model_load_dir", default="", type=str)
    parser.add_argument("--load_model", default=1, type=int)    # load pretrained model, if not, pretrain
    parser.add_argument("--nell_evaluation", default=0, type=int)
    parser.add_argument("--random_seed", default=83, type=int)
    parser.add_argument("--gfaw_dataset_base", default="/data/base2/Bio-Relation-Extract/BioRE-master/data/", type=str)
    parser.add_argument("--gfaw_dataset", default="GFAW-cutoff-1.0-PCNN-1.0", type=str)
    # parser.add_argument("--nell_query", default='all', type=str)

    # for PCNN model
    parser.add_argument("--load_pcnn_model", default=1, type=int)   # load pretrained model, if not, pretrain
    parser.add_argument("--pcnn_dataset_base", default='/data/base2/Bio-Relation-Extract/BioRE-master/rl+PCNN/Combined/data', type=str)
    parser.add_argument("--pcnn_dataset_name", default='Less200_0.7', type=str)
    parser.add_argument("--pcnn_batch_size", default=30, type=int)
    parser.add_argument("--pcnn_max_epoch", default=120, type=int)  # default=5

    # for joint model modules
    parser.add_argument("--use_replay_memory", default=1, type=int)  # 1
    parser.add_argument("--use_joint_model", default=0, type=int)  # 1
    parser.add_argument("--sample_RM_neg_ratio", default=0.8, type=float)  # 1
    parser.add_argument("--bfs_iteration", default=200, type=int)  # default=100
    parser.add_argument("--train_pcnn", default=1, type=int)  # 1


    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    ## new argument added
    # for input graph data
    parsed['data_input_dir'] = os.path.join(parsed['gfaw_dataset_base'], parsed['gfaw_dataset']) # adjust
    parsed['vocab_dir'] = os.path.join(parsed['data_input_dir'], "vocab")
    parsed['base_output_dir'] = os.path.join('../Joint-Experiments', parsed['gfaw_dataset'])
    ## ended

    parsed['input_files'] = [parsed['data_input_dir'] + '/' + parsed['input_file']]

    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)
    parsed['use_replay_memory'] = (parsed['use_replay_memory'] == 1)
    parsed['use_joint_model'] = (parsed['use_joint_model'] == 1)

    parsed['pretrained_embeddings_action'] = ""
    parsed['pretrained_embeddings_entity'] = ""

    output_name = str(parsed['path_length'])+'_'+str(parsed['hidden_size'])+'_'+str(parsed['embedding_size'])+'_'+str(parsed['Lambda'])+'_'+str(parsed['random_seed'])+'_'+str(parsed['train_entity_embeddings'])+'_'+str(parsed['max_num_actions'])+'_'+str(parsed['total_iterations'])+"_"+time.strftime("%d%H%M%S")   # +'_'+str(parsed['train_pcnn']) # +'_EntityTest'
    parsed['output_dir'] = os.path.join(parsed['base_output_dir'], output_name)
    parsed['model_dir'] = os.path.join(parsed['output_dir'], 'model/')

    parsed['load_model'] = (parsed['load_model'] == 1)
    parsed['load_pcnn_model'] = (parsed['load_pcnn_model'] == 1)
    parsed['train_pcnn'] = (parsed['train_pcnn'] == 1)

    if parsed['model_load_dir'] == '':
        parsed['model_load_dir'] = parsed['model_dir']

    ##Logger##
    parsed['path_logger_file'] = parsed['output_dir']
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
    if not os.path.isdir(parsed['output_dir']):
        os.makedirs(parsed['output_dir'])
        os.mkdir(parsed['model_dir'])
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed
