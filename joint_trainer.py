# ======================= import packages from Rl model =======================
from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import tensorflow as tf

# Export PYTHONPATH so that 'rl_code' folder can be regarded as a package
import sys

from rl_code.model.trainer import Trainer
from rl_code.model.agent import Agent
from rl_code.options import read_options
from rl_code.model.environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from rl_code.model.baseline import ReactiveBaseline
from rl_code.model.nell_eval import nell_eval
from scipy.misc import logsumexp as lse
from pprint import pprint

# ======================= import packages from PCNN model =======================
import nrekit
import numpy as np
import tensorflow as tf
import sys
import json
import time

# ======================= initialize global parameters =======================
# tf and config settings
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
config.allow_soft_placement = True

# read command line options
options = read_options()
tf.set_random_seed(options['random_seed'])
np.random.seed(options['random_seed'])

# ======================= initialize GFAW parameters =======================
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Set logging
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
logfile = logging.FileHandler(options['log_file_name'], 'w')
logfile.setFormatter(fmt)
logger.addHandler(logfile)

# read the vocab files, it will be used by many classes hence global scope
logger.info('reading vocab files...')
options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))

logger.info('Reading mid to name map')
mid_to_word = {}
logger.info('Done..')
logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))

# ======================= import code from PCNN model =======================
dataset_dir = os.path.join(options['pcnn_dataset_base'], options['pcnn_dataset_name'])
if not os.path.isdir(dataset_dir):
    raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

# The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'),
                                                        os.path.join(dataset_dir, 'word_vec.json'),
                                                        os.path.join(dataset_dir, 'rel2id.json'),
                                                        mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                        shuffle=True, batch_size=options['pcnn_batch_size'], case_sensitive=False,
                                                        reprocess=False)
test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'),
                                                       os.path.join(dataset_dir, 'word_vec.json'),
                                                       os.path.join(dataset_dir, 'rel2id.json'),
                                                       mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                       shuffle=False, batch_size=options['pcnn_batch_size'], case_sensitive=False,
                                                       reprocess=False)

framework = nrekit.framework.re_framework(train_loader, test_loader)

class pcnn_model(nrekit.framework.re_model):
    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")

        # Embedding
        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

        # Encoder
        x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
        x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)

        # Selector
        self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope,
                                                                               self.ins_label,
                                                                               self.rel_tot, True,
                                                                               keep_prob=0.5)
        self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope,
                                                                             self.ins_label,
                                                                             self.rel_tot, False,
                                                                             keep_prob=1.0)

        # Classifier
        self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label,
                                                                     self.rel_tot,
                                                                     weights_table=self.get_weights())

    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table

# TODO: Pretrain GFAW/PCNN if there doesn't exist
# ======================= Pretrain GFAW =======================
if not options['load_model']:
    trainer = Trainer(options)
    with tf.Session(config=config) as sess:
        sess.run(trainer.initialize())
        trainer.initialize_pretrained_embeddings(sess=sess)
        # trainer.test_environment = trainer.test_test_environment  # use test to show result
        trainer.train(sess)

    tf.reset_default_graph()

    # Pretrained Model Test
    trainer = Trainer(options)

    save_path = os.path.join(options['model_dir'], 'model.ckpt')  # trainer.save_path
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir

    with tf.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess)

        trainer.test_rollouts = 100
        test_set_name = 'test'
        if not os.path.isdir(path_logger_file + "/" + "test_beam_" + test_set_name):
            os.mkdir(path_logger_file + "/" + "test_beam_" + test_set_name)
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam_" + test_set_name + "/paths"
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write(test_set_name + "(beam) scores with best model from " + save_path + "\n")
        # trainer.test_environment = trainer.test_test_environment  # test_environment = dev_test_environment
        trainer.test_environment = env(trainer.params, test_set_name)
        # trainer.test_environment.test_rollouts = 100
        trainer.train_environment.grapher.array_store = np.load(file=str(options['model_dir'] + 'new_graph.npy'))
        trainer.test(sess, beam=True, print_paths=True, save_model=False)

    tf.reset_default_graph()

# Training PCNN model
if not options['load_pcnn_model']:
    framework.train(pcnn_model, ckpt_dir="checkpoint", model_name=options['pcnn_dataset_name'] + "_pcnn_att", max_epoch=options['pcnn_max_epoch'])

# TODO: Load GFAW/PCNN pretrained model and Joint Model runs
# Load pretrained PCNN model
framework = nrekit.framework.re_framework(train_loader, test_loader)
framework.partial_run_setup(model=pcnn_model, model_name='pcnn_att', pretrain_model="checkpoint/" + options['pcnn_dataset_name'] + '_pcnn_att', max_epoch=options['pcnn_max_epoch'])

# Change Output dir so that Joint Model can be saved
# output_name = str(options['path_length'])+'_'+str(options['hidden_size'])+'_'+str(options['embedding_size'])+'_'+str(options['Lambda'])+'_'+str(options['random_seed'])+'_'+str(options['train_entity_embeddings'])+'_'+str(options['max_num_actions']) # +'_EntityTest' # +'_'+str(options['batch_size'])+'_'+str(options['num_rollouts'])+'_EntityTest_200BFS_64*20'  # +'_negconstrain'
output_name = "Joint_"+time.strftime("%d%H%M%S")+str(options['path_length'])+'_'+str(options['hidden_size'])+'_'+str(options['embedding_size'])+'_'+str(options['Lambda'])+'_'+str(options['random_seed'])+'_'+str(options['train_entity_embeddings'])+'_'+str(options['max_num_actions'])+'_'+str(options['batch_size'])+'_'+str(options['num_rollouts'])+'_'+str(options['bfs_iteration'])+'_'+str(options['train_pcnn'])+'_'+str(options['use_replay_memory'])+'_'+str(options['use_joint_model'])+'_'+str(options['sample_RM_neg_ratio'])
options['output_dir'] = os.path.join(options['base_output_dir'], output_name)
options['model_dir'] = os.path.join(options['output_dir'], 'model/')
options['path_logger_file'] = options['output_dir']
options['log_file_name'] = options['output_dir'] +'/log.txt'
if not os.path.isdir(options['output_dir']):
    os.makedirs(options['output_dir'])
    os.mkdir(options['model_dir'])
with open(options['output_dir']+'/config.txt', 'w') as out:
    pprint(options, stream=out)

# Load pretrained GFAW model
train_JointModel = True
if train_JointModel:
    trainer = Trainer(options)
    with tf.Session(config=config) as sess:
        # Train Joint Model
        trainer.initialize(restore=os.path.join(options['model_load_dir'], 'model.ckpt'), sess=sess)
        trainer.initialize_pretrained_embeddings(sess=sess)
        trainer.test_environment = trainer.test_test_environment    # use test to show result
        # trainer.train_joint(sess, framework)   # input PCNN framework into GFAW trainer
        # trainer.train_joint_withoutRM(sess, framework)
        trainer.train_joint_module(sess, framework)
        # trainer.store_bfs()

    tf.reset_default_graph()

# Joint Model Test
test_JointModel = True
if test_JointModel:
    trainer = Trainer(options)

    save_path = os.path.join(options['model_dir'], 'model.ckpt')  # trainer.save_path
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir

    with tf.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess)

        trainer.test_rollouts = 100
        test_set_name = 'test'
        if not os.path.isdir(path_logger_file + "/" + "test_beam_" + test_set_name):
            os.mkdir(path_logger_file + "/" + "test_beam_" + test_set_name)
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam_" + test_set_name + "/paths"
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write(test_set_name + "(beam) scores with best model from " + save_path + "\n")
        # trainer.test_environment = trainer.test_test_environment  # test_environment = dev_test_environment
        trainer.test_environment = env(trainer.params, test_set_name)
        # trainer.test_environment.test_rollouts = 100
        trainer.train_environment.grapher.array_store = np.load(file=str(options['model_dir'] + 'new_graph.npy'))
        trainer.test(sess, beam=True, print_paths=True, save_model=False)

    tf.reset_default_graph()


