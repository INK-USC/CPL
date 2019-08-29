import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn

from evaluation import ranking_and_hits, ranking_and_hits2
from model import ConvE, DistMult, Complex, RNNDist

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(precision=3)

timer = CUDATimer()
cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = True
Config.embedding_dim = 200
#Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = 51
load = False
if Config.dataset is None:
    Config.dataset = 'ICEWS'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)


''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def main():
    if Config.process: preprocess(Config.dataset, delete_data=True)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

    train_batcher = StreamBatcher(Config.dataset, 'train', Config.batch_size, randomize=True, keys=input_keys)
    dev_rank_batcher = StreamBatcher(Config.dataset, 'dev_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)
    test_rank_batcher = StreamBatcher(Config.dataset, 'test_ranking', Config.batch_size, randomize=False, loader_threads=4, keys=input_keys)


    if Config.model_name is None:
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ConvE':
        model = ConvE(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'DistMult':
        model = DistMult(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'ComplEx':
        model = Complex(vocab['e1'].num_token, vocab['rel'].num_token)
    elif Config.model_name == 'RNNDist':
        model = RNNDist(vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        log.info('Unknown model: {0}', Config.model_name)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))

    eta = ETAHook('train', print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=100))
    if Config.dataset == 'ICEWS18':
        lengths = [1618, 956, 815, 1461, 1634, 1596, 1754, 1494, 800, 979, 1588, 1779, 1831, 1762, 1566, 812, 820, 1707, 1988, 1845, 1670, 1695, 956, 930, 1641, 1813, 1759, 1664, 1616, 1021, 998, 1668, 1589, 1720]
    else:
        lengths= [1090, 730, 646, 939, 681, 783, 546, 526, 524, 586, 656, 741, 562, 474, 493, 487, 474, 477, 460, 532, 348, 530, 402, 493, 503, 452, 668, 512, 406, 467, 524, 563, 524, 418, 441, 487, 515, 475, 478, 532, 387, 479, 485, 417, 542, 496, 487, 445, 504, 350, 432, 445, 401, 570, 554, 504, 505, 483, 587, 441, 489, 501, 487, 513, 513, 524, 655, 545, 599, 702, 734, 519, 603, 579, 537, 635, 437, 422, 695, 575, 553, 485, 429, 663, 475, 673, 527, 559, 540, 591, 558, 698, 422, 1145, 969, 1074, 888, 683, 677, 910, 902, 644, 777, 695, 571, 656, 797, 576, 468, 676, 687, 549, 482, 1007, 778, 567, 813, 788, 879, 557, 724, 850, 809, 685, 714, 554, 799, 727, 208, 946, 979, 892, 859, 1092, 1038, 999, 1477, 1126, 1096, 1145, 955, 100, 1264, 1287, 962, 1031, 1603, 1662, 1179, 1064, 1179, 1105, 1465, 1176, 1219, 1137, 1112, 791, 829, 2347, 917, 913, 1107, 960, 850, 1005, 1045, 871, 972, 921, 1019, 984, 1033, 848, 918, 699, 1627, 1580, 1354, 1119, 1065, 1208, 1037, 1134, 980, 1249, 1031, 908, 787, 819, 804, 764, 959, 1057, 770, 691, 816, 620, 788, 829, 895, 1128, 1023, 1038, 1030, 1016, 991, 866, 878, 1013, 977, 914, 976, 717, 740, 904, 912, 1043, 1117, 930, 1116, 1028, 946, 922, 1151, 1092, 967, 1189, 1081, 1158, 943, 981, 1212, 1104, 941, 912, 1347, 1241, 1479, 1188, 1152, 1164, 1167, 1173, 1280, 979, 142, 1458, 910, 1126, 1053, 1083, 897, 1021, 1075, 881, 1054, 941, 927, 860, 1081, 876, 1952, 1576, 1560, 1599, 1226, 1083, 964, 1059, 1179, 982, 1032, 933, 877, 1032, 957, 884, 909, 846, 850, 798, 843, 1183, 1108, 1185, 797, 915, 952, 1181, 744, 86, 889, 1151, 925, 1119, 1115, 1036, 772, 1052, 837, 897, 1095, 926, 1034, 1031, 995, 907, 969, 981, 1135, 915, 1161, 100, 1269, 1244, 1331, 1124, 1074, 1162, 1159, 1078, 1311, 1210, 1308, 945, 1183, 1580, 1406, 1417, 1173, 1348, 1274, 1179, 893, 1107, 950, 1028, 1055, 1059, 1244, 1082, 1179, 1011, 955, 886, 865, 857]
    if Config.cuda:
        model.cuda()
    if load:
    # if True:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        # ranks = ranking_and_hits2(model, test_rank_batcher, vocab, 'test_evaluation')
        print(len(ranks))

        mrr = []
        curr_step = 0
        for i in range(len(lengths)):
            rr = np.array(ranks[curr_step:curr_step + 2 * lengths[i]])
            mrr.append(np.mean(1 / rr))

            curr_step += 2 * lengths[i]
        with open(Config.dataset +'mrr.txt', 'w') as f:
            for i, mr in enumerate(mrr):
                print("MRR (filtered) @ {}th day: {:.6f}".format(i, mr))
                f.write(str(mr) + '\n')
        h10 = []
        curr_step = 0
        for i in range(len(lengths)):
            rr = np.array(ranks[curr_step:curr_step + 2 * lengths[i]])
            h10.append(np.mean(rr <= 10))
        with open(Config.dataset +'h10.txt', 'w') as f:
            for i, mr in enumerate(h10):
                print("h10 (filtered) @ {}th day: {:.6f}".format(i, mr))
                f.write(str(mr) + '\n')
        h10 = []
        for i in range(len(lengths)):
            rr = np.array(ranks[curr_step:curr_step + 2 * lengths[i]])
            h10.append(np.mean(rr <= 3))
        with open(Config.dataset + 'h3.txt', 'w') as f:
            for i, mr in enumerate(h10):
                print("h10 (filtered) @ {}th day: {:.6f}".format(i, mr))
                f.write(str(mr) + '\n')

        h10 = []

        for i in range(len(lengths)):
            rr = np.array(ranks[curr_step:curr_step + 2 * lengths[i]])
            h10.append(np.mean(rr <= 1))
        with open(Config.dataset + 'h1.txt', 'w') as f:
            for i, mr in enumerate(h10):
                print("h10 (filtered) @ {}th day: {:.6f}".format(i, mr))
                f.write(str(mr) + '\n')
        print("length", len(ranks))
        print("length_2", 2* sum(lengths))

        # ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))

    opt = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.L2)
    for epoch in range(epochs):
        # break
        model.train()
        for i, str2var in enumerate(train_batcher):
            opt.zero_grad()
            e1 = str2var['e1']
            rel = str2var['rel']
            e2_multi = str2var['e2_multi1_binary'].float()

            # label smoothing
            # e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.size(1))
            # print("this",Config.label_smoothing_epsilon, e2_multi.size(1))

            pred = model.forward(e1, rel)
            # loss = model.loss(pred, e2_multi)
            # #
            loss = torch.zeros(1).cuda()
            for j in range(128):
                position = torch.nonzero(e2_multi[j])[0].cuda()
                label = torch.cat([torch.ones(len(position)), torch.zeros(len(position))]).cuda()
                neg_position = torch.randint(e2_multi.shape[1], (len(position),)).long().cuda()
                position = torch.cat([position, neg_position])
                loss += model.loss(pred[j,position], label)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients
            opt.step()

            train_batcher.state.loss = loss.cpu()


        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            # ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
            if epoch == 50 :
                ranks = ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
                # ranks = ranking_and_hits2(model, test_rank_batcher, vocab, 'test_evaluation')



    # with torch.no_grad():
        #     ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
        #     if epoch % 3 == 0:
        #         if epoch > 0:
        #             ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')


if __name__ == '__main__':
    main()
