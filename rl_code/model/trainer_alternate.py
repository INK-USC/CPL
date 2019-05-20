from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import tensorflow as tf

# Export PYTHONPATH so that 'rl_code' folder can be regarded as
# a package
import sys
from rl_code.model.agent import Agent, AgentTarget
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

logger = logging.getLogger()
logging.basicConfig(filename=time.strftime("%d%H%M%S")+'output.log', level=logging.DEBUG)

import random
import numpy as np
import pandas as pd
import pickle

'''

class priorityDictionary(dict):
    def __init__(self):
        self.__heap = []
        dict.__init__(self)

    def smallest(self):
        # Find smallest item after removing deleted items from heap.
        if len(self) == 0:
            print("smallest of empty priorityDictionary")
        heap = self.__heap
        while heap[0][1] not in self or self[heap[0][1]] != heap[0][0]:
            lastItem = heap.pop()
            insertionPoint = 0
            while 1:
                smallChild = 2 * insertionPoint + 1
                if smallChild + 1 < len(heap) and \
                        heap[smallChild] > heap[smallChild + 1]:
                    smallChild += 1
                if smallChild >= len(heap) or lastItem <= heap[smallChild]:
                    heap[insertionPoint] = lastItem
                    break
                heap[insertionPoint] = heap[smallChild]
                insertionPoint = smallChild
        return heap[0][1]

    def __iter__(self):
        # Create destructive sorted iterator of priorityDictionary.

        def iterfn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]

        return iterfn()

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        heap = self.__heap
        if len(heap) > 2 * len(self):
            self.__heap = [(v, k) for k, v in self.iteritems()]
            self.__heap.sort()  # builtin sort likely faster than O(n) heapify
        else:
            newPair = (val, key)
            insertionPoint = len(heap)
            heap.append(None)
            while insertionPoint > 0 and \
                    newPair < heap[(insertionPoint - 1) // 2]:
                heap[insertionPoint] = heap[(insertionPoint - 1) // 2]
                insertionPoint = (insertionPoint - 1) // 2
            heap[insertionPoint] = newPair

    # def setdefault(self, key, val):
        # Reimplement setdefault to call our customized __setitem__. 
    #    if key not in self:
    #        self[key] = val
    #    return self[key]

# def Dijkstra(G, start, end=None):
    D = {}  # dictionary of final distances
    P = {}  # dictionary of predecessors
    Q = priorityDictionary()  # est.dist. of non-final vert.
    Q[start] = 0

    for v in Q:
        D[v] = Q[v]
        if v == end: break

        if v not in G:  # this node is not a head node in the graph
            continue

        for w in G[v]:
            if G[v][w] < 1: print(v,w)
            vwLength = D[v] + G[v][w]
            if w in D:
                if vwLength < D[w]:
                    print("Dijkstra: found better path to already-final vertex")
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v

    return (D, P)

def shortestPath(G, start, end):
    """
    Find a single shortest path from the given start vertex
    to the given end vertex.
    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along
    the shortest path.
    """

    D, P = Dijkstra(G, start, end)

    if end in D:
        distance = D[end]
        Path = []
        while 1:
            Path.append(end)
            if end == start: break
            if end not in P:
                break
            end = P[end]
        Path.reverse()
        return distance, Path
    else:
        return 'nan', []

def constructGraph(train, edge_train):
    Graph = {}
    print("Constructing graph from training set...")
    for index, row in tqdm(train.iterrows()):
        if row['e1'] not in Graph:
            Graph[row['e1']] = {}
            if row['e1'] == row['e2']:
                pass
            else:
                Graph[row['e1']][row['e2']] = 1
        else:
            if row['e2'] not in Graph[row['e1']]:
                if row['e1'] == row['e2']:
                    pass
                else:
                    Graph[row['e1']][row['e2']] = 1

    print("Constructing graph from added edges training set...")
    for index, row in tqdm(edge_train.iterrows()):
        if row['e1'] not in Graph:
            Graph[row['e1']] = {}
            if row['e1'] == row['e2']:
                pass
            else:
                Graph[row['e1']][row['e2']] = 0.5
        else:
            if row['e2'] not in Graph[row['e1']]:
                if row['e1'] == row['e2']:
                    pass
                else:
                    Graph[row['e1']][row['e2']] = 0.5

    return Graph

'''

class Memory(object):
    def __init__(self, memory_size=4000):
        self.memory = {}
        self.memory_size = memory_size

    def insert(self, key, extend_value):
        # TODO: replace old memory with new memory if it's full
        if key not in self.memory:
            self.memory[key] = []
        self.memory[key].extend(extend_value)
        random.shuffle(self.memory[key])
        self.memory[key] = self.memory[key][:self.memory_size]

    def sample(self, batch_size, keys):
        # TODO: if memory size < batch_size, deal with it!
        try:
            indices = list(range(len(self.memory[keys[0]])))
            random.shuffle(indices)
            indices = indices[:batch_size]
            res = [np.array(self.memory[key])[indices] for key in keys]
            return res
        except:
            print("Wrong Input!")

    def clear(self):
        del self.memory
        self.memory = {}


class Trainer(object):
    def __init__(self, params):

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        self.params = params
        self.save_path = None
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')   # test set: path_test.txt
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def sample_PCNN(self, framework):
        indices = np.random.randint(len(self.pcnn_experience.memory['triples']), size=self.params['pcnn_batch_size'])
        sampled = np.array(self.pcnn_experience.memory['triples'])[indices]

        # return pcnn index for e1, e2, rn
        # relation_map = framework.test_data_loader.rel2id
        # entity_map = framework.test_data_loader.word2id
        #
        # def function(row):
        #     return [entity_map[row[0]], entity_map[row[1]], relation_map[row[2]]]
        #
        # sampled = np.apply_along_axis(function, 1, sampled)
        return sampled


    def sample_GFAW(self, ratio):
        """
        :param batch_size: batch size in GFAW
        :return: feed_dict list for every step in this path
        """

        # TODO: 1. sample 数据
        indices_pos = np.random.randint(len(self.pos_experience.memory['entity_path']), size=round((1-ratio)*self.num_rollouts*self.batch_size))
        indices_neg = np.random.randint(len(self.neg_experience.memory['entity_path']), size=round(ratio*self.num_rollouts*self.batch_size))

        path_rewards_pos = np.array(self.pos_experience.memory['path_rewards'])[indices_pos]  # np.repeat( , self.num_rollouts)
        path_rewards_neg = np.array(self.neg_experience.memory['path_rewards'])[indices_neg]  # np.repeat( , self.num_rollouts)
        path_rewards = np.concatenate((path_rewards_pos, path_rewards_neg), axis=0)

        state_rewards_pos = np.array(self.pos_experience.memory['state_rewards'])[indices_pos]  # np.repeat( , self.num_rollouts, axis=0)
        state_rewards_neg = np.array(self.neg_experience.memory['state_rewards'])[indices_neg]  # np.repeat( , self.num_rollouts, axis=0)
        state_rewards = np.concatenate((state_rewards_pos, state_rewards_neg), axis=0)

        # TODO: 2. 放入 feed_dict
        path_length = len(self.pos_experience.memory['relation_path'][0])
        feed_dict = [{} for _ in range(path_length)]

        feed_dict[0][self.first_state_of_test] = False

        query_relation_neg = np.array(self.neg_experience.memory['query_relation'])[indices_neg]
        query_relation_pos = np.array(self.pos_experience.memory['query_relation'])[indices_pos]  # np.repeat( , self.num_rollouts)
          # np.repeat( , self.num_rollouts)
        query_relation_pos = np.reshape(query_relation_pos,(-1,1))
        query_relation = np.concatenate((query_relation_pos, query_relation_neg), axis=0)

        feed_dict[0][self.query_relation] = query_relation.reshape(self.batch_size * self.num_rollouts)
        feed_dict[0][self.range_arr] = np.arange(self.batch_size * self.num_rollouts)
        for i in range(path_length):
            # print(self.pos_experience.memory['entity_path'])

            entities_neg = np.array(self.neg_experience.memory['entity_path'])[indices_neg][:,i]
            entities_pos = np.array(self.pos_experience.memory['entity_path'])[indices_pos][:,i]
            entities = np.concatenate((entities_pos, entities_neg), axis=0)
            entities = np.array([ety_index if ety_index < len(self.train_environment.grapher.array_store) else
                                 self.train_environment.grapher.entity_vocab['UNK'] for ety_index in entities])

            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = self.train_environment.grapher.array_store[entities][:, :, 1]  # np.repeat( , self.num_rollouts, axis=0)   # TODO 2.2
            feed_dict[i][self.candidate_entity_sequence[i]] = self.train_environment.grapher.array_store[entities][:, :, 0]  # np.repeat( , self.num_rollouts, axis=0)  # TODO 2.3
            feed_dict[i][self.entity_sequence[i]] = entities  # np.repeat( , self.num_rollouts)  # TODO 2.4

        return feed_dict, path_rewards, state_rewards

    def sample_pos_GFAW(self):
        """
        :param batch_size: batch size in GFAW
        :return: feed_dict list for every step in this path
        """

        # TODO: 1. sample 数据
        indices = np.random.randint(len(self.pos_experience.memory['entity_path']), size=self.batch_size * self.num_rollouts)

        path_rewards = np.array(self.pos_experience.memory['path_rewards'])[indices]  # np.repeat( , self.num_rollouts)
        state_rewards = np.array(self.pos_experience.memory['state_rewards'])[indices]  # np.repeat( , self.num_rollouts, axis=0)

        # TODO: 2. 放入 feed_dict
        path_length = len(self.pos_experience.memory['relation_path'][0])
        feed_dict = [{} for _ in range(path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = np.array(self.pos_experience.memory['query_relation'])[indices].reshape(self.batch_size * self.num_rollouts)  # np.repeat( ,self.num_rollouts)  # TODO: 2.1 to update
        feed_dict[0][self.range_arr] = np.arange(self.batch_size * self.num_rollouts)
        for i in range(path_length):
            entities = np.array(self.pos_experience.memory['entity_path'])[indices][:, i]
            entities = np.array([ety_index if ety_index < len(self.train_environment.grapher.array_store) else
                                 self.train_environment.grapher.entity_vocab['UNK'] for ety_index in entities])

            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = self.train_environment.grapher.array_store[entities][:, :, 1]  # np.repeat( , self.num_rollouts, axis=0)  # TODO 2.2
            feed_dict[i][self.candidate_entity_sequence[i]] = self.train_environment.grapher.array_store[entities][:, :, 0]  # np.repeat( , self.num_rollouts, axis=0)  # TODO 2.3
            feed_dict[i][self.entity_sequence[i]] = entities  # np.repeat( , self.num_rollouts)  # TODO 2.4

        return feed_dict, path_rewards, state_rewards


    def target_model_setup(self):
        self.agent_target = AgentTarget(self.params)
        self.initialize_target()
        self.initialize_weight_update()

    def GFAW_Q_initialize(self, path_length):
        self.Q_target = tf.placeholder(tf.float32, [None, self.max_num_actions], name='Q_target')
        self.Q_eval = self.per_example_logits[i]
        self.Q_loss = tf.reduce_mean(tf.squared_difference(self.Q_target, self.Q_eval))

    def calc_reinforce_loss(self):
        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()
        # self.pp = tf.Print(self.tf_baseline)
        # multiply with rewards
        final_reward = self.cum_discounted_reward - self.tf_baseline
        # reward_std = tf.sqrt(tf.reduce_mean(tf.square(final_reward))) + 1e-5 # constant addded for numerical stability
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])
        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.div(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        self.loss_before_reg = loss

        total_loss = tf.reduce_mean(loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # scalar

        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy

    def initialize(self, restore=None, sess=None):
        with tf.device("/gpu:0"):
            self.agent = Agent(self.params)
            logger.info("Creating TF graph...")
            self.candidate_relation_sequence = []
            self.candidate_entity_sequence = []
            self.input_path = []
            self.first_state_of_test = tf.placeholder(tf.bool, name="is_first_state_of_test")
            self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
            self.range_arr = tf.placeholder(tf.int32, shape=[None, ])
            self.global_step = tf.Variable(0, trainable=False)
            self.decaying_beta = tf.train.exponential_decay(self.beta, self.global_step,
                                                       200, 0.90, staircase=False)
            self.entity_sequence = []

            # to feed in the discounted reward tensor
            self.cum_discounted_reward = tf.placeholder(tf.float32, [None, self.path_length],
                                                        name="cumulative_discounted_reward")

            for t in range(self.path_length):
                next_possible_relations = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                       name="next_relations_{}".format(t))
                next_possible_entities = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                         name="next_entities_{}".format(t))
                input_label_relation = tf.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
                start_entities = tf.placeholder(tf.int32, [None, ])
                self.input_path.append(input_label_relation)
                self.candidate_relation_sequence.append(next_possible_relations)
                self.candidate_entity_sequence.append(next_possible_entities)
                self.entity_sequence.append(start_entities)

            self.loss_before_reg = tf.constant(0.0)
            self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
                self.candidate_relation_sequence,
                self.candidate_entity_sequence, self.entity_sequence,
                self.input_path,
                self.query_relation, self.range_arr, self.first_state_of_test, self.path_length)

            self.loss_op = self.calc_reinforce_loss()

            # mark trainable_variables
            self.trainable_variables = tf.trainable_variables()

            # backprop
            self.train_op = self.bp(self.loss_op)

            # Building the test graph
            self.prev_state = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
            self.prev_relation = tf.placeholder(tf.int32, [None, ], name="previous_relation")
            self.query_embedding = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.query_relation)  # [B, 2D]
            layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
            formated_state = [tf.unstack(s, 2) for s in layer_state]
            self.next_relations = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
            self.next_entities = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])

            self.current_entities = tf.placeholder(tf.int32, shape=[None,])


            with tf.variable_scope("policy_steps_unroll") as scope:
                scope.reuse_variables()
                self.test_loss, test_state, self.test_logits, self.test_action_idx, self.chosen_relation = self.agent.step(
                    self.next_relations, self.next_entities, formated_state, self.prev_relation, self.query_embedding,
                    self.current_entities, self.input_path[0], self.range_arr, self.first_state_of_test)
                self.test_state = tf.stack(test_state)

            logger.info('TF Graph creation done..')
            self.model_saver = tf.train.Saver(max_to_keep=2)

            # return the variable initializer Op.
            if not restore:
                return tf.global_variables_initializer()
            else:
                return self.model_saver.restore(sess, restore)

    def initialize_target(self):
        """ Initialize the target model, which is a fixed model as the same as self.agent
        We proposed it for Fixed Target Update
        """
        logger.info("Creating TF graph...")
        self.candidate_relation_sequence_target = []
        self.candidate_entity_sequence_target = []
        self.input_path_target = []
        self.first_state_of_test_target = tf.placeholder(tf.bool, name="is_first_state_of_test_target")
        self.query_relation_target = tf.placeholder(tf.int32, [None], name="query_relation_target")
        self.range_arr_target = tf.placeholder(tf.int32, shape=[None, ])
        self.global_step_target = tf.Variable(0, trainable=False)
        self.decaying_beta_target = tf.train.exponential_decay(self.beta, self.global_step_target,
                                                        200, 0.90, staircase=False)
        self.entity_sequence_target = []

        # to feed in the discounted reward tensor
        self.cum_discounted_reward_target = tf.placeholder(tf.float32, [None, self.path_length],
                                                    name="cumulative_discounted_reward_target")

        for t in range(self.path_length):
            next_possible_relations_target = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_relations_{}_target".format(t))
            next_possible_entities_target = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                    name="next_entities_{}_target".format(t))
            input_label_relation_target = tf.placeholder(tf.int32, [None], name="input_label_relation_{}_target".format(t))
            start_entities_target = tf.placeholder(tf.int32, [None, ])
            self.input_path_target.append(input_label_relation_target)
            self.candidate_relation_sequence_target.append(next_possible_relations_target)
            self.candidate_entity_sequence_target.append(next_possible_entities_target)
            self.entity_sequence_target.append(start_entities_target)
        self.loss_before_reg_target = tf.constant(0.0)
        self.per_example_loss_target, self.per_example_logits_target, self.action_idx_target = self.agent_target(
            self.candidate_relation_sequence_target,
            self.candidate_entity_sequence_target, self.entity_sequence_target,
            self.input_path_target,
            self.query_relation_target, self.range_arr_target, self.first_state_of_test_target, self.path_length)

        # Building the test graph
        self.prev_state_target = tf.placeholder(tf.float32, self.agent_target.get_mem_shape(), name="memory_of_agent")
        self.prev_relation_target = tf.placeholder(tf.int32, [None, ], name="previous_relation")
        self.query_embedding_target = tf.nn.embedding_lookup(self.agent_target.relation_lookup_table, self.query_relation_target)  # [B, 2D]
        layer_state_target = tf.unstack(self.prev_state_target, self.LSTM_layers)
        formated_state_target = [tf.unstack(s, 2) for s in layer_state_target]
        self.next_relations_target = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities_target = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])

        self.current_entities_target = tf.placeholder(tf.int32, shape=[None, ])

        with tf.variable_scope("policy_steps_unroll_target") as scope:
            scope.reuse_variables()
            self.test_loss_target, test_state_target, self.test_logits_target, self.test_action_idx_target, self.chosen_relation_target = self.agent_target.step(
                self.next_relations_target, self.next_entities_target, formated_state_target, self.prev_relation_target, self.query_embedding_target,
                self.current_entities_target, self.input_path_target[0], self.range_arr_target, self.first_state_of_test_target)
            self.test_state_target = tf.stack(test_state_target)

        return tf.global_variables_initializer()

    def initialize_weight_update(self):
        """ call this function after initializing all parameters for self.agent and self.agent_target
        Update self.agent_target using variables in self.agent
        """

        variables = tf.trainable_variables()
        e_params = sorted(self.trainable_variables, key=lambda x: x.name)
        t_params = sorted(list(set(variables) - set(self.trainable_variables)), key=lambda x: x.name)
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] # extend the replace operations

    def gpu_io_setup_target(self):
        # create fetches for partial_run_setup
        fetches = self.per_example_loss_target  + self.action_idx_target + self.per_example_logits_target
        feeds =  [self.first_state_of_test_target] + self.candidate_relation_sequence_target+ self.candidate_entity_sequence_target + self.input_path_target + \
                [self.query_relation_target] + [self.cum_discounted_reward_target] + [self.range_arr_target] + self.entity_sequence_target

        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test_target] = False
        feed_dict[0][self.query_relation_target] = None
        feed_dict[0][self.range_arr_target] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path_target[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence_target[i]] = None
            feed_dict[i][self.candidate_entity_sequence_target[i]] = None
            feed_dict[i][self.entity_sequence_target[i]] = None

        return fetches, feeds, feed_dict

    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_action != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            _ = sess.run((self.agent.relation_embedding_init),
                         feed_dict={self.agent.action_embedding_placeholder: embeddings})
        if self.pretrained_embeddings_entity != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            _ = sess.run((self.agent.entity_embedding_init),
                         feed_dict={self.agent.entity_embedding_placeholder: embeddings})


    def bp(self, cost):
        self.baseline.update(tf.reduce_mean(self.cum_discounted_reward))
        tvars = self.trainable_variables
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op


    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])  # [B]
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  # [B, T]
        cum_disc_reward[:,
        self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def gpu_io_setup(self):
        # create fetches for partial_run_setup
        fetches = self.per_example_loss + self.action_idx + [self.loss_op] + self.per_example_logits + [self.dummy]
        feeds=[self.first_state_of_test] + self.candidate_relation_sequence+ self.candidate_entity_sequence + self.input_path + \
                [self.query_relation] + [self.cum_discounted_reward] + [self.range_arr] + self.entity_sequence

        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def gpu_io_setup_test(self):
        # create fetches for partial_run_setup
        fetches = self.per_example_loss + self.action_idx + self.per_example_logits
        feeds = [self.first_state_of_test] + self.candidate_relation_sequence+ self.candidate_entity_sequence + self.input_path + \
                [self.query_relation] + [self.range_arr] + self.entity_sequence

        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def train(self, sess):
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0
        for episode in self.train_environment.get_episodes():

            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relation] = episode.get_query_relation()

            # get initial state
            state = episode.get_state()
            # for each time step
            loss_before_regularization = []
            logits = []
            for i in range(self.path_length):
                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i], self.per_example_logits[i], self.action_idx[i]],
                                                  feed_dict=feed_dict[i])
                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)
                # action = np.squeeze(action, axis=1)  # [B,]
                state = episode(idx)
            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # get the final reward from the environment
            rewards = episode.get_reward()

            # computed cumulative discounted reward
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]


            # backprop
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                   feed_dict={self.cum_discounted_reward: cum_discounted_reward})

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size),
                               train_loss))

            if self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                if not os.path.exists(self.path_logger_file + "/" + str(self.batch_counter)):
                    os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def get_pcnn_predictions(self, framework, state, id_entities_dict):

        entpair_list, result_list = framework.predict(state['current_entities'], id_entities_dict)
        pcnn_edge_idx = []
        next_relations, next_entities, pcnn_confidence_recorder = [], [], []
        _, max_length = state['next_relations'].shape

        # map from PCNN to GFAW
        self.train_environment.batcher.relation_vocab['NA'] = self.train_environment.batcher.relation_vocab['UNK']  # TODO: delete it
        relation_map = {framework.test_data_loader.rel2id[k]:self.train_environment.batcher.relation_vocab[k] for k in framework.test_data_loader.rel2id}
        entity_map = self.train_environment.batcher.entity_vocab

        for i, (entpairs, results) in enumerate(zip(entpair_list, result_list)):
            current_ety_idx = state['current_entities'][i]

            # Add edge from GFAW
            if len(results) == 0:
                pcnn_edge_idx.append(self.max_num_actions)
                next_rlt = self.train_environment.grapher.array_store[current_ety_idx][:, 1]
                next_ety = self.train_environment.grapher.array_store[current_ety_idx][:, 0]
                next_relations.append(next_rlt)
                next_entities.append(next_ety)

            # Add edge from PCNN
            else:
                length = sum(np.array(entpairs) != "None#None")
                pcnn_rlt = [relation_map[j] for j in np.argmax(results, axis=1)[:length]]
                pcnn_ety = [entity_map[j.split("#")[-1]] if j.split("#")[-1] in entity_map else entity_map['UNK'] for j in entpairs[:length]]
                next_rlt = self.train_environment.grapher.array_store[current_ety_idx][:, 1]
                next_ety = self.train_environment.grapher.array_store[current_ety_idx][:, 0]
                for edge_idx in range(len(pcnn_rlt)):
                    next_rlt[-(edge_idx+1)] = pcnn_rlt[edge_idx]
                    next_ety[-(edge_idx+1)] = pcnn_ety[edge_idx]
                # if predicted idx > pcnn edge idx, it's pcnn edge
                pcnn_edge_idx.append(self.max_num_actions-len(pcnn_rlt)-1)
                next_relations.append(next_rlt)
                next_entities.append(next_ety)


        next_relations = np.array(next_relations)
        next_entities = np.array(next_entities)

        return next_relations, next_entities, pcnn_edge_idx

    '''
    # Deprecated
    
    
    def load_dicts(self):
        print("Using bfs to store positive pcnn sequences in replay memory...")
        if os.path.exists('./_processed_data/'+self.params['gfaw_dataset']+'-pre_train.pkl'):
            print("Dicts already Stored!")
            dir = './_processed_data/'
            pre_train = pickle.load(open(os.path.join(dir, self.params['gfaw_dataset']+'-pre_train.pkl'), 'rb'))
            Graph = pickle.load(open(os.path.join(dir, self.params['gfaw_dataset']+'-Graph.pkl'), 'rb'))
            entpair2rlt = pickle.load(open(os.path.join(dir, self.params['gfaw_dataset']+'-entpair2rlt.pkl'), 'rb'))
            edge_entpair2rlt = pickle.load(open(os.path.join(dir, self.params['gfaw_dataset']+'-edge_entpair2rlt.pkl'), 'rb'))
        else:
            dir = "/data/base/Joint-Datasets"
            load_dir = './_processed_data/'
            pre_dataset = os.path.join(dir, self.params['gfaw_dataset'])
            after_dataset = os.path.join(dir, self.params['gfaw_dataset'])

            pre_train = pd.read_csv(os.path.join(pre_dataset, "train.txt"), sep='\t', names=['e1', 'r', 'e2'])
            after_train = pd.read_csv(os.path.join(pre_dataset, "train.txt"), sep='\t', names=['e1', 'r', 'e2'])

            edges = after_train.append(pre_train).drop_duplicates(keep=False)
            print("Storing entity pairs to relations dict...")
            edge_entpair2rlt = {(row['e1'] + '#' + row['e2']): row['r'] for (index, row) in tqdm(edges.iterrows())}
            entpair2rlt = {(row['e1'] + '#' + row['e2']): row['r'] for (index, row) in tqdm(after_train.iterrows())}
            Graph = constructGraph(pre_train, edges)

            print("Saving Dicts for future use...")
            pickle.dump(pre_train, open(os.path.join(load_dir, self.params['gfaw_dataset']+'-pre_train.pkl'), 'wb'))
            pickle.dump(Graph, open(os.path.join(load_dir, self.params['gfaw_dataset']+'-Graph.pkl'), 'wb'))
            pickle.dump(entpair2rlt, open(os.path.join(load_dir, self.params['gfaw_dataset']+'-entpair2rlt.pkl'), 'wb'))
            pickle.dump(edge_entpair2rlt, open(os.path.join(load_dir, self.params['gfaw_dataset']+'-edge_entpair2rlt.pkl'), 'wb'))
            del after_train, edges

        return pre_train, Graph, entpair2rlt, edge_entpair2rlt

    def use_bfs(self):
        pre_train, Graph, entpair2rlt, edge_entpair2rlt = self.load_dicts()
        count = 0

        # dir = "../../data/"
        # pre_dataset = os.path.join(dir, "GFAW-cutoff-0.5-PCNN-1.0")
        # test = pd.read_csv(os.path.join(pre_dataset, "test.txt"), sep='\t', names=['e1', 'r', 'e2'])

        for index, row in tqdm(pre_train.sample(n=100).iterrows()):      # , random_state=self.params['random_seed']
            if row['e1'] == row['e2']:
                pass
            else:
                del Graph[row['e1']][row['e2']]
                distance, path = shortestPath(Graph, row['e1'], row['e2'])
                print(distance, path)
                Graph[row['e1']][row['e2']] = 1
                ## ===================== for pcnn edge:
                if len(path) and distance<(len(path)-1) and (len(path)-1) == self.params['path_length']:
                    # exist path and exist pcnn added edge
                    flag = 0
                    for e in path:
                        if e not in self.train_environment.grapher.entity_vocab:
                            print(e, " is not in vocab!")
                            flag = 1
                    if not flag:
                        count += 1
                        entity_path = [self.train_environment.grapher.entity_vocab[e] for e in path]
                        relation_path_ = [entpair2rlt[path[i] + '#' + path[i + 1]] for i in range(len(path) - 1)]
                        relation_path = [self.train_environment.grapher.relation_vocab[r] for r in relation_path_]
                        pcnn_edge = [1 if path[i] + '#' + path[i + 1] in edge_entpair2rlt else 0 for i in
                                     range(len(path) - 1)]
                        self.pos_experience.insert('entity_path', np.array([entity_path]))
                        self.pos_experience.insert('relation_path', np.array([relation_path]))
                        self.pos_experience.insert('path_rewards', np.array([1]))
                        self.pos_experience.insert('state_rewards', np.array([[1] * self.params['path_length']]))
                        self.pos_experience.insert('query_relation',
                                               np.array([[self.train_environment.grapher.relation_vocab[row['r']]]]))
                        self.pos_experience.insert('pcnn_edge', np.array([pcnn_edge]))

                        for index, is_pcnn in enumerate(pcnn_edge):
                            if is_pcnn:
                                self.pcnn_experience.insert('triples',
                                                            [[path[index], path[index + 1], relation_path_[index]]])
                del distance, path

                ## ============================ for gfaw edge:
                # if len(path) and (len(path) - 1) == self.params['path_length']:
                #     # exist path and exist pcnn added edge
                #     flag = 0
                #     for e in path:
                #         if e not in self.train_environment.grapher.entity_vocab:
                #             print(e, " is not in vocab!")
                #             flag = 1
                #     if not flag:
                #         count += 1
                #         entity_path = [self.train_environment.grapher.entity_vocab[e] for e in path]
                #         relation_path_ = [entpair2rlt[path[i]+'#'+path[i+1]] for i in range(len(path)-1)]
                #         relation_path = [self.train_environment.grapher.relation_vocab[r] for r in relation_path_]
                #         pcnn_edge = [1 if path[i]+'#'+path[i+1] in edge_entpair2rlt else 0 for i in range(len(path)-1)]
                #         self.pos_experience.insert('entity_path', np.array([entity_path]))
                #         self.pos_experience.insert('relation_path', np.array([relation_path]))
                #         self.pos_experience.insert('path_rewards', np.array([1]))
                #         self.pos_experience.insert('state_rewards', np.array([[1] * self.params['path_length']]))
                #         self.pos_experience.insert('query_relation', np.array([[self.train_environment.grapher.relation_vocab[row['r']]]]))
                #         self.pos_experience.insert('pcnn_edge', np.array([pcnn_edge]))
                #
                #         for index, is_pcnn in enumerate(pcnn_edge):
                #             if is_pcnn:
                #                 self.pcnn_experience.insert('triples', [[path[index], path[index+1], relation_path_[index]]])
                # del distance, path

        print("Done! There are ", str(count), " positive samples!")
        del pre_train, edge_entpair2rlt, entpair2rlt, Graph

    def store_bfs(self):
        self.pos_experience = Memory(memory_size=50000)
        self.pcnn_experience = Memory(memory_size=50000)
        pre_train, Graph, entpair2rlt, edge_entpair2rlt = self.load_dicts()
        count = 0

        for index, row in pre_train.sample(frac=1).iterrows():  # , random_state=self.params['random_seed']
            print(count, end='\r')
            if row['e1'] == row['e2']:
                pass
            else:
                del Graph[row['e1']][row['e2']]
                distance, path = shortestPath(Graph, row['e1'], row['e2'])
                Graph[row['e1']][row['e2']] = 1
                ## ===================== for pcnn edge:
                if len(path) and distance < (len(path) - 1) and (len(path) - 1) == self.params['path_length']:
                    # exist path and exist pcnn added edge
                    flag = 0
                    for e in path:
                        if e not in self.train_environment.grapher.entity_vocab:
                            print(e, " is not in vocab!")
                            flag = 1
                    if not flag:
                        count += 1
                        entity_path = [self.train_environment.grapher.entity_vocab[e] for e in path]
                        relation_path_ = [entpair2rlt[path[i] + '#' + path[i + 1]] for i in range(len(path) - 1)]
                        relation_path = [self.train_environment.grapher.relation_vocab[r] for r in relation_path_]
                        pcnn_edge = [1 if path[i] + '#' + path[i + 1] in edge_entpair2rlt else 0 for i in
                                     range(len(path) - 1)]
                        self.pos_experience.insert('entity_path', np.array([entity_path]))
                        self.pos_experience.insert('relation_path', np.array([relation_path]))
                        self.pos_experience.insert('path_rewards', np.array([1]))
                        self.pos_experience.insert('state_rewards', np.array([[1] * self.params['path_length']]))
                        self.pos_experience.insert('query_relation',
                                               np.array([[self.train_environment.grapher.relation_vocab[row['r']]]]))
                        self.pos_experience.insert('pcnn_edge', np.array([pcnn_edge]))

                        for index, is_pcnn in enumerate(pcnn_edge):
                            if is_pcnn:
                                self.pcnn_experience.insert('triples', [[path[index], path[index + 1], relation_path_[index]]])

            if 'entity_path' in self.pos_experience.memory and len(self.pos_experience.memory['entity_path']) == self.pos_experience.memory_size:
                print("Memory Full!")
                break

        pickle.dump(self.pos_experience, open(os.path.join('./_processed_data/', self.params['gfaw_dataset'] + '-pre_bfs.pkl'), 'wb'))
        pickle.dump(self.pcnn_experience, open(os.path.join('./_processed_data/', self.params['gfaw_dataset'] + '-pre_pcnn.pkl'), 'wb'))


    def one_path(self, one_line):
        triples = one_line.strip().split(';')
        query = eval(triples[0])
        path_triples = triples[2:-1]
        relation_path = [eval(triple)[1] for triple in path_triples]
        entity_path = [eval(triple)[0] for triple in path_triples] + [query[2]]

        entity_path.extend([0] * (self.path_length-len(path_triples)))   # add padding
        relation_path.extend([0] * (self.path_length-len(path_triples)))  # add padding

        self.pos_experience.insert('entity_path', np.array([entity_path]))
        self.pos_experience.insert('relation_path', np.array([relation_path]))
        self.pos_experience.insert('path_rewards', np.array([1]))
        self.pos_experience.insert('state_rewards', np.array([[1] * self.path_length]))
        self.pos_experience.insert('query_relation', np.array([[query[1]]]))
        # self.pos_experience.insert('pcnn_edge', np.array([pcnn_edge]))

    def use_bfs_new(self):
        for _ in tqdm(range(self.batch_size * self.num_rollouts)):
            self.one_path(self.bfs_path[random.randint(0, len(self.bfs_path) - 1)])
            
    '''

    def load_bfs_file(self):
        print("Using bfs to store positive pcnn sequences in replay memory...")
        if os.path.exists('./_processed_data/FB60K-pre_train.pkl'):
            print("Dicts already Stored!")
            dir = './_processed_data/'
            pre_train = pickle.load(open(os.path.join(dir, 'FB60K-pre_train.pkl'), 'rb'))
            self.edge_entpair2rlt = pickle.load(open(os.path.join(dir, 'FB60K-edge_entpair2rlt.pkl'), 'rb')) # self.params['gfaw_dataset']+
        else:
            dir = "/data/base/Joint-Datasets"
            load_dir = './_processed_data/'
            new_paths = os.path.join("/data/base/Joint-Datasets/FB15K-237+NYT/text/")

            pre_train = pd.read_csv(os.path.join(new_paths, "top1.txt"), sep='\t', names=['e1', 'e2', 'r', 'pos'])
            print("Storing entity pairs to relations dict...")
            self.edge_entpair2rlt = {}
            for (index, row) in tqdm(pre_train.iterrows()):
                self.edge_entpair2rlt.update({(str(row['e1']) + '#' + str(row['e2'])): str(row['r'])})
                self.edge_entpair2rlt.update({(str(row['e2']) + '#' + str(row['e1'])): str(row['r'])})

            print("Saving Dicts for future use...")
            pickle.dump(pre_train, open(os.path.join(load_dir, 'FB60K-pre_train.pkl'), 'wb'))
            pickle.dump(self.edge_entpair2rlt, open(os.path.join(load_dir, 'FB60K-edge_entpair2rlt.pkl'), 'wb'))

        data_dir = self.params['data_input_dir'] # Change to PCNN dataset


        self.paths = {}
        self.stats = {}
        self.discovered = []

        num_rollout_steps = 3
        le1, le2, lr = 0, 0, 0
        current = []

        if os.path.exists('./_processed_data/FB60K-distribution.json'):
            self.stats = json.load(open('./_processed_data/FB60K-distribution.json',"r"))
        else:
            dir = "../FB60K-1.0_rev"  # Change to GFAW dataset
            loader = open(os.path.join(dir, 'dfspaths2.txt'))
            for line in tqdm(loader.readlines()):
                p1 = line.split(":")
                p2 = p1[1].split(";")
                e1, e2, r = p1[0].split(",")
                e1, e2 = int(e1), int(e2)
                if le1 != e1 or le2 != e2 or lr != r:
                    current = []
                path = ""
                for element in p2[1:-1]:
                    e11, e22, rr = element.split(",")
                    path += "-" + rr
                u = len(p2)
                while u <= num_rollout_steps:
                    path += "-0"
                    u += 1

                if r not in self.stats:
                    self.stats.update({r: {}})
                if e1 not in self.paths:
                    self.paths.update({e1: {}})
                if e2 not in self.paths[e1]:
                    self.paths[e1].update({e2: []})

                if path not in self.stats[r]: self.stats[r].update({path: 0.0})
                if path not in current:
                    self.stats[r][path] += 1.0
                    current.append(path)

            for rel in self.stats.keys():
                top_ = sorted(self.stats[rel].items(), key=lambda kv: kv[1], reverse=True)
                if len(top_) > 10: top_ = top_[:10]
                # with open("paths.log", "a") as path: print(ent, ttl, top_20, file=path)
                self.stats[rel] = [kv[0] for kv in top_]
            json.dump(self.stats,open('./_processed_data/FB60K-distribution.json',"w"))

        if os.path.exists('./_processed_data/' + self.params['gfaw_dataset'] + '-paths.pkl'):
            self.paths = pickle.load(open(os.path.join("./_processed_data/", self.params['gfaw_dataset']+'-paths.pkl'), 'rb'))
            self.discovered = pickle.load(open(os.path.join("./_processed_data/", self.params['gfaw_dataset'] + '-discovered.pkl'), 'rb'))
        else:
            loader = open(os.path.join(data_dir, 'pcnnpaths.txt'))
            for line in tqdm(loader.readlines()):
                p1 = line.split(":")
                p2 = p1[1].split(";")
                e1, e2, r = p1[0].split(",")
                e1, e2 = int(e1), int(e2)
                if le1 != e1 or le2 != e2 or lr != r:
                    current = []
                path = {"rel": "", "triple": []}
                Existed = True
                for element in p2[1:-1]:
                    e11, e22, rr = element.split(",")
                    path["rel"] += "-" + rr
                    e11, e22, rr = int(e11), int(e22), int(rr)
                    path["triple"].append((e11, rr, e22))
                u = len(p2)
                while u <= num_rollout_steps:
                    path["rel"] += "-0"
                    u += 1
                    path["triple"].append((e2, 0, e2))

                if str(r) not in self.stats: continue
                if path["rel"] not in self.stats[str(r)]: continue

                if e1 not in self.paths:
                    self.paths.update({e1:{}})
                if e2 not in self.paths[e1]:
                    self.paths[e1].update({e2:[]})

                self.discovered.append((e1,e2))
                if path["rel"] not in current:
                    current.append(path["rel"])
                self.paths[e1][e2].append((r,path))

            pickle.dump(self.paths,open(os.path.join("./_processed_data/", self.params['gfaw_dataset']+'-paths.pkl'), 'wb'))
            pickle.dump(self.discovered,open(os.path.join("./_processed_data/", self.params['gfaw_dataset'] + '-discovered.pkl'), 'wb'))



    def bfs_action(self):
        count = 0
        for (e1,e2) in tqdm(random.sample(self.discovered,100)):
            ## Add PCNN Edges
            scount = 0
            path = self.paths[e1][e2]
            for (r,each) in path:
                if str(r) not in self.stats: continue
                if each["rel"] not in self.stats[str(r)]:
                    continue
                count += 1
                scount += 1
                if len(each["triple"])<self.path_length:
                    while len(each["triple"])<self.path_length: each["triple"].append((each["triple"][-1][2],0,each["triple"][-1][2]))
                entity_path = [t[0] for t in each["triple"]]
                entity_path.append(each["triple"][-1][2])
                relation_path = [t[1] for t in each["triple"]]
                pcnn_edge = [1 if str(self.train_environment.grapher.rev_entity_vocab[i[0]]) + '#' + str(self.train_environment.grapher.rev_entity_vocab[i[2]]) in self.edge_entpair2rlt else 0 for i in each["triple"]]
                self.pos_experience.insert('entity_path', np.array([entity_path]))
                self.pos_experience.insert('relation_path', np.array([relation_path]))
                self.pos_experience.insert('path_rewards', np.array([1]))
                self.pos_experience.insert('state_rewards', np.array([[1] * self.params['path_length']]))
                self.pos_experience.insert('query_relation',
                                           np.array([r]))
                self.pos_experience.insert('pcnn_edge', np.array([pcnn_edge]))
                if scount > 12: break

        print("Done! There are ", str(count), " positive samples!")

    def train_joint_withoutRM(self, sess, framework):
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()
        # fetches_test, feeds_test, feed_dict_test = self.gpu_io_setup_test()

        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0

        id_entities_dict = {self.train_environment.batcher.entity_vocab[k]: k
                            for k in self.train_environment.batcher.entity_vocab}

        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relation] = episode.get_query_relation()

            # get initial state
            state = episode.get_state()
            # for each time step
            loss_before_regularization = []
            logits = []
            for i in range(self.path_length):
                next_relations, next_entities, pcnn_edge_idx = self.get_pcnn_predictions(framework, state,
                                                                                         id_entities_dict)

                # Switch to GFAW
                with sess.as_default():
                    with sess.graph.as_default():
                        # TODO: Adapt GFAW feed_dict according to the PCNN prediction
                        feed_dict[i][self.candidate_relation_sequence[i]] = next_relations
                        feed_dict[i][self.candidate_entity_sequence[i]] = next_entities
                        feed_dict[i][self.entity_sequence[i]] = state['current_entities']  # [batch_size*num_rollouts, ]

                        # GFAW predict next action
                        # TODO: sess.partial_run
                        per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i],
                                                                                         self.per_example_logits[i],
                                                                                         self.action_idx[i]],
                                                                                     feed_dict=feed_dict[i])

                        loss_before_regularization.append(per_example_loss)
                        logits.append(per_example_logits)
                        # action = np.squeeze(action, axis=1)  # [B,]

                        # GFAW return next state
                        # episode.state['next_entities'] = np.array(next_entities)
                        # state = episode(idx)    # __call__(self, action) return state
                        state['current_entities'] = np.array(next_entities)[np.arange(self.batch_size * self.num_rollouts), idx]

            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # get the final reward from the environment
            # rewards = episode.get_reward()
            reward = (state['current_entities'] == episode.end_entities)
            condlist = [reward == True, reward == False]
            choicelist = [episode.positive_reward, episode.negative_reward]
            rewards = np.select(condlist, choicelist)  # [B,]

            # computed cumulative discounted reward
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

            # backprop
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                   feed_dict={self.cum_discounted_reward: cum_discounted_reward})

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size),
                               train_loss))

            if self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                if not os.path.exists(self.path_logger_file + "/" + str(self.batch_counter)):
                    os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    '''

    def train_joint(self, sess, framework):
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()
        fetches_test, feeds_test, feed_dict_test = self.gpu_io_setup_test()
        self.load_bfs_file()

        # setup target model
        # self.target_model_setup()
        # sess.run(self.replace_target_op)

        # if self.params['bfs_iteration']:
        #     with open(os.path.join(self.params['data_input_dir'], 'path_8r.txt'), 'r') as f:
        #         self.bfs_path = f.readlines()

        train_loss = 0.0
        pcnn_pos_edge_appearance = {}
        self.batch_counter = 0
        self.pos_experience = Memory()
        self.neg_experience = Memory()
        self.pcnn_experience = Memory()

        id_entities_dict = {self.train_environment.batcher.entity_vocab[k]: k
                                for k in self.train_environment.batcher.entity_vocab}

        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            loss_before_regularization = []
            logits = []

            if self.batch_counter <= self.params['bfs_iteration']:     # 200
                # TODO: use replay memory to train gfaw
                # TODO: use bfs to store positive pcnn sequence to replay memory
                # self.use_bfs()
                # self.use_bfs_new()
                self.bfs_action()
            else:
                h = sess.partial_run_setup(fetches=fetches_test, feeds=feeds_test)
                feed_dict_test[0][self.query_relation] = episode.get_query_relation()

                # get initial state
                state = episode.get_state()

                # store path
                entity_trajectory = []
                relation_trajectory = []
                is_pcnn_edge = []
                pcnn_edge_idx = []
                pcnn_confidence_recorder = []
                all_query_relation = []

                for i in range(self.path_length):
                    # =============== get from PCNN  =============== #

                    # setup run_array and feed_dict for framework
                    # Switch to PCNN
                    # res = framework.get_results()
                    next_relations, next_entities, pcnn_edge_idx = self.get_pcnn_predictions(framework, state, id_entities_dict)

                    # Switch to GFAW
                    with sess.as_default():
                        with sess.graph.as_default():

                            # TODO: Adapt GFAW feed_dict according to the PCNN prediction
                            # feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']     # [batch_size*num_rollouts, 200]
                            feed_dict_test[i][self.candidate_relation_sequence[i]] = next_relations
                            # feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']    # [batch_size*num_rollouts, 200]
                            feed_dict_test[i][self.candidate_entity_sequence[i]] = next_entities

                            feed_dict_test[i][self.entity_sequence[i]] = state['current_entities']      # [batch_size*num_rollouts, ]

                            # GFAW predict next action
                            # TODO: sess.partial_run
                            per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i],
                                                                                             self.per_example_logits[i],
                                                                                             self.action_idx[i]],
                                                                                         feed_dict=feed_dict_test[i])
                            # per_example_logits, shape = (batch_size*rollouts, max_num_actions)
                            # per_example_loss, shape = (batch_size*rollouts, )
                            # idx: the next chosen action of the list index,  shape = (batch_size*rollouts, )

                            # TODO: Store the predicted path
                            # use predicted idx to get the
                            relations = np.array(next_relations)[np.arange(self.batch_size*self.num_rollouts), idx]
                            # pcnn_conf = np.array(pcnn_confidence)[np.arange(self.batch_size*self.num_rollouts), idx]
                            entity_trajectory.append(state['current_entities'])
                            relation_trajectory.append(relations)
                            # pcnn_confidence_recorder.append(pcnn_conf)

                            is_pcnn_edge_ = [1 if idx[edge] > pcnn_edge_idx[edge] else 0 for edge in range(len(idx))]
                            is_pcnn_edge.append(is_pcnn_edge_)

                            # GFAW return next state
                            # episode.state['next_entities'] = np.array(next_entities)
                            # state = episode(idx)    # __call__(self, action) return state
                            state['current_entities'] = np.array(next_entities)[np.arange(self.batch_size*self.num_rollouts), idx]

                    del next_relations, next_entities, is_pcnn_edge_       # , pcnn_confidence

                # Store the end entities
                entity_trajectory.append(state['current_entities'])
                all_query_relation.append(episode.get_query_relation())

                # Reshape
                entity_trajectory = np.column_stack(entity_trajectory)      # shape = (batch_size*rollouts, path_length+1)
                relation_trajectory = np.column_stack(relation_trajectory)  # shape = (batch_size*rollouts, path_length)
                # pcnn_confidence_recorder = np.column_stack(pcnn_confidence_recorder)   # shape = (batch_size*rollouts, path_length)
                is_pcnn_edge = np.column_stack(is_pcnn_edge)        # shape = (batch_size*rollouts, path_length)
                path_pcnn_sum = np.sum(is_pcnn_edge, axis=1)        # shape = (batch_size*rollouts, )
                all_query_relation = np.column_stack(all_query_relation)

                # get the final reward from the environment
                # rewards = episode.get_reward()     # [batch_size*num_rollouts, ]
                reward = (state['current_entities'] == episode.end_entities)
                condlist = [reward == True, reward == False]
                choicelist = [episode.positive_reward, episode.negative_reward]
                rewards = np.select(condlist, choicelist)  # [B,]

                # computed cumulative discounted reward
                cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

                # TODO: Find if it's positive sentence or not
                positive_sequences = [i for i in range(len(rewards)) if rewards[i] == 1]
                negative_sequences = [i for i in range(len(rewards)) if rewards[i] == 0]

                # TODO: If from PCNN (positive & negative)
                positive_pcnn_sequences = [index for index in positive_sequences if path_pcnn_sum[index] > 0]
                positive_gfaw_sequences = [index for index in positive_sequences if path_pcnn_sum[index] == 0]
                negative_pcnn_sequences = [index for index in negative_sequences if path_pcnn_sum[index] > 0]
                negative_gfaw_sequences = [index for index in negative_sequences if path_pcnn_sum[index] == 0]
                positive_sequences = positive_pcnn_sequences + positive_gfaw_sequences    # positive_pcnn_sequences +
                negative_sequences = negative_pcnn_sequences + negative_gfaw_sequences    # negative_pcnn_sequences +

                # not add too much negative pcnn sequences
                # negative_pcnn_sequences_sampled = [negative_pcnn_sequences[x] for x in random.sample(range(len(negative_pcnn_sequences)), 2*len(positive_pcnn_sequences))]
                # pos_neg_pcnn_seq_idx = positive_pcnn_sequences + negative_pcnn_sequences_sampled
                # print(len(positive_pcnn_sequences), " Positive samples, ", len(negative_pcnn_sequences_sampled), " Negative samples.")
                # pos_neg_pcnn_seq_idx = positive_pcnn_sequences + negative_pcnn_sequences

                # TODO: enlarge edges in KG
                # add edges to self.train_environment.grapher.array_store
                if len(positive_pcnn_sequences):
                    entity_paths = entity_trajectory[positive_pcnn_sequences]
                    relation_paths = relation_trajectory[positive_pcnn_sequences]
                    paths_pcnn_egde = is_pcnn_edge[positive_pcnn_sequences]
                    for i in range(len(paths_pcnn_egde)):   # [0, 0, 1]
                        for j in range(len(paths_pcnn_egde[i])):
                            if paths_pcnn_egde[i][j]:      # this is pcnn edge
                                e1 = entity_paths[i][j]
                                r = relation_paths[i][j]
                                e2 = entity_paths[i][j+1]
                                edge_name = str(e1)+'#'+str(r)+'#'+str(e2)
                                if edge_name not in pcnn_pos_edge_appearance:
                                    pcnn_pos_edge_appearance[edge_name] = 1
                                else:
                                    pcnn_pos_edge_appearance[edge_name] += 1
                                if pcnn_pos_edge_appearance[edge_name] <5:
                                    # at most add 5 repeated edges
                                    # TODO: Store e1, e2, r in the PCNN Memory
                                    self.pcnn_experience.insert('triples', [[self.rev_entity_vocab[e1], self.rev_entity_vocab[e2], self.rev_relation_vocab[r]]])
                                    # Store e1, e2, r in the graph
                                    for row in range(self.max_num_actions):
                                        if self.train_environment.grapher.array_store[e1][row].sum() == 0:
                                            # change the first nozero edge into (e1, e2, r)
                                            self.train_environment.grapher.array_store[e1][row] = [e2, r]
                                            continue
                                        elif row == (self.max_num_actions-1):
                                            # e1's outgoint degree > max_num_action, random replace one adge with (e1, e2, r)
                                            self.train_environment.grapher.array_store[e1][np.random.randint(self.max_num_actions)] = [e2, r]
                    del entity_paths, relation_paths, paths_pcnn_egde

                # TODO: Add replay memory

                if len(positive_sequences):
                    self.pos_experience.insert('entity_path', entity_trajectory[positive_sequences])
                    self.pos_experience.insert('relation_path', relation_trajectory[positive_sequences])
                    # self.pos_experience.insert('pcnn_confidence_recorder', pcnn_confidence_recorder[pos_neg_pcnn_seq_idx])
                    self.pos_experience.insert('pcnn_edge', is_pcnn_edge[positive_sequences])
                    self.pos_experience.insert('path_rewards', rewards[positive_sequences])
                    self.pos_experience.insert('state_rewards', cum_discounted_reward[positive_sequences])
                    self.pos_experience.insert('query_relation', all_query_relation[positive_sequences])

                if len(negative_sequences):
                    self.neg_experience.insert('entity_path', entity_trajectory[negative_sequences])
                    self.neg_experience.insert('relation_path', relation_trajectory[negative_sequences])
                    # self.pos_experience.insert('pcnn_confidence_recorder', pcnn_confidence_recorder[pos_neg_pcnn_seq_idx])
                    self.neg_experience.insert('pcnn_edge', is_pcnn_edge[negative_sequences])
                    self.neg_experience.insert('path_rewards', rewards[negative_sequences])
                    self.neg_experience.insert('state_rewards', cum_discounted_reward[negative_sequences])
                    self.neg_experience.insert('query_relation', all_query_relation[negative_sequences])


                # TODO: Clean Memory
                del entity_trajectory, relation_trajectory, is_pcnn_edge, pcnn_edge_idx, path_pcnn_sum, pcnn_confidence_recorder, all_query_relation
                del rewards, cum_discounted_reward
                del positive_sequences, negative_sequences, positive_pcnn_sequences, negative_pcnn_sequences  # , pos_neg_pcnn_seq_idx

            # TODO: Update PCNN
            if self.params['train_pcnn']:
                if 'triples' not in self.pcnn_experience.memory or len(self.pcnn_experience.memory['triples']) < self.params['pcnn_batch_size']:
                    print("PCNN replay memory size < PCNN batch size, skip PCNN backward!")
                    pass
                else:
                    print("PCNN backward...")
                    batch_pcnn_triples = self.sample_PCNN(framework)
                    batch_data = framework.test_data_loader.batch_gen_train(batch_pcnn_triples)
                    framework.backward(batch_data)
                    del batch_data, batch_pcnn_triples

            # TODO: Update GFAW
            if 'entity_path' not in self.neg_experience.memory or (len(self.neg_experience.memory['entity_path']) < self.batch_size* self.num_rollouts):
                if (len(self.pos_experience.memory['entity_path']) < self.batch_size* self.num_rollouts):
                    print("Replay memory size < GFAW batch size, skip GFAW backward!")
                    pass
                else:
                    # use positive pcnn samples to update
                    with sess.as_default():
                        with sess.graph.as_default():
                            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                            feed_dict, rewards, cum_discounted_reward = self.sample_pos_GFAW()

                            for i in range(self.path_length):
                                per_example_loss, per_example_logits, idx = sess.partial_run(h,
                                                                                             [self.per_example_loss[i],
                                                                                              self.per_example_logits[
                                                                                                  i],
                                                                                              self.action_idx[i]],
                                                                                             feed_dict=feed_dict[i])
                                loss_before_regularization.append(per_example_loss)
                                logits.append(per_example_logits)
                                del per_example_logits, per_example_loss, idx

                            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

                            # get the final reward from the environment
                            # rewards = episode.get_reward()

                            # computed cumulative discounted reward
                            # cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

                            # backprop
                            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                                   feed_dict={
                                                                       self.cum_discounted_reward: cum_discounted_reward})

                            # print statistics
                            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
                            avg_reward = np.mean(rewards)
                            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
                            # entity pair, atleast one of the path get to the right answer
                            reward_reshape = np.reshape(rewards, (
                            self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
                            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
                            reward_reshape = (reward_reshape > 0)
                            num_ep_correct = np.sum(reward_reshape)

                    if np.isnan(train_loss):
                        raise ArithmeticError("Error in computing loss")

                    logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                                "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                                format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                       (num_ep_correct / self.batch_size),
                                       train_loss))

                    if self.batch_counter % self.eval_every == 0:
                        with open(self.output_dir + '/scores.txt', 'a') as score_file:
                            score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                        if not os.path.exists(self.path_logger_file + "/" + str(self.batch_counter)):
                            os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                        self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                        self.test(sess, beam=True, print_paths=False)

                    logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                    del rewards, cum_discounted_reward, loss_before_regularization, logits
                    del batch_total_loss, avg_reward, reward_reshape, num_ep_correct
                    gc.collect()
            else:
                # use both positive & negative samples to update
                with sess.as_default():
                    with sess.graph.as_default():
                        h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                        feed_dict, rewards, cum_discounted_reward = self.sample_GFAW()

                        for i in range(self.path_length):
                            per_example_loss, per_example_logits, idx = sess.partial_run(h,
                                    [self.per_example_loss[i], self.per_example_logits[i], self.action_idx[i]],
                                                              feed_dict=feed_dict[i])
                            loss_before_regularization.append(per_example_loss)
                            logits.append(per_example_logits)
                            del per_example_logits, per_example_loss, idx

                        loss_before_regularization = np.stack(loss_before_regularization, axis=1)

                        # get the final reward from the environment
                        # rewards = episode.get_reward()

                        # computed cumulative discounted reward
                        # cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

                        # backprop
                        batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                               feed_dict={self.cum_discounted_reward: cum_discounted_reward})

                        # print statistics
                        train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
                        avg_reward = np.mean(rewards)
                        # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
                        # entity pair, atleast one of the path get to the right answer
                        reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
                        reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
                        reward_reshape = (reward_reshape > 0)
                        num_ep_correct = np.sum(reward_reshape)

                if np.isnan(train_loss):
                    raise ArithmeticError("Error in computing loss")

                logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                            "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                            format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                   (num_ep_correct / self.batch_size),
                                   train_loss))

                if self.batch_counter%self.eval_every == 0:
                    with open(self.output_dir + '/scores.txt', 'a') as score_file:
                        score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                    if not os.path.exists(self.path_logger_file + "/" + str(self.batch_counter)):
                        os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                    self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                    self.test(sess, beam=True, print_paths=False)

                logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                del rewards, cum_discounted_reward, loss_before_regularization, logits
                del batch_total_loss, avg_reward, reward_reshape, num_ep_correct
                gc.collect()

            if self.batch_counter >= self.total_iterations:
                break
                
    '''

    def train_joint_module(self, sess, framework):

        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0
        self.load_bfs_file()

        # =========================== initialize =========================
        if self.params['use_replay_memory']:
            fetches_test, feeds_test, feed_dict_test = self.gpu_io_setup_test()
            fetches, feeds, feed_dict = self.gpu_io_setup()
            self.pos_experience = Memory()
            self.neg_experience = Memory()
            self.pcnn_experience = Memory()
        else:
            fetches, feeds, feed_dict = self.gpu_io_setup()

        # =========================== prepare the index dict for matching PCNN & MINERVA =========================
        if self.params['use_joint_model']:
            id_entities_dict = {self.train_environment.batcher.entity_vocab[k]: k
                                for k in self.train_environment.batcher.entity_vocab}
            pcnn_pos_edge_appearance = {}

        # =========================== start main loop =========================
        for episode in self.train_environment.get_episodes():
            print("Epoch ",self.batch_counter)
            self.batch_counter += 1                     # for each bach
            state = episode.get_state()                  # get initial state
            loss_before_regularization = []              # for each time step, backprop
            logits = []

            # start bfs at first several hundreds iteration if use bfs
            if self.batch_counter - 1 < self.params['bfs_iteration']:
                # use bfs to store positive pcnn sequence to replay memory
                print("This iteration uses BFS")
                self.bfs_action()
            else:
                if self.batch_counter - 1 == self.params['bfs_iteration']:
                    # use bfs to store positive pcnn sequence to replay memory
                    self.pos_experience.clear()
                    self.neg_experience.clear()
                    print(self.pos_experience.memory)

                # use joint model
                print("This iteration uses replay memory")
                if self.params['use_replay_memory']:    # use replay memory to backprop
                    h = sess.partial_run_setup(fetches=fetches_test, feeds=feeds_test)
                    feed_dict_test[0][self.query_relation] = episode.get_query_relation()
                    entity_trajectory = []
                    relation_trajectory = []
                    all_query_relation = []
                    if self.params['use_joint_model']:
                        is_pcnn_edge = []
                else:
                    h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                    feed_dict[0][self.query_relation] = episode.get_query_relation()

                # =========================== one iteration inference from gfaw or joint model =========================
                for i in range(self.path_length):
                    if self.params['use_joint_model']:
                        # use pcnn to get predictions based on entities
                        next_relations, next_entities, pcnn_edge_idx = self.get_pcnn_predictions(framework, state,
                                                                                                 id_entities_dict)

                    with sess.as_default():
                        with sess.graph.as_default():
                            if not self.params['use_replay_memory'] and not self.params['use_joint_model']:
                                # use MINERVA to update as normal
                                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations'] # [batch_size*num_rollouts, 200]
                                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']    # [batch_size*num_rollouts, 200]
                                feed_dict[i][self.entity_sequence[i]] = state['current_entities']   # [batch_size*num_rollouts, ]
                                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i],
                                                                                                 self.per_example_logits[i],
                                                                                                 self.action_idx[i]],
                                                                                             feed_dict=feed_dict[i])
                                # per_example_logits, shape = (batch_size*rollouts, max_num_actions)
                                # per_example_loss, shape = (batch_size*rollouts, )
                                # idx: the next chosen action of the list index,  shape = (batch_size*rollouts, )

                                loss_before_regularization.append(per_example_loss)
                                logits.append(per_example_logits)
                            elif not self.params['use_replay_memory'] and self.params['use_joint_model']:
                                # use joint model but do not use replay memory
                                feed_dict[i][self.candidate_relation_sequence[i]] = next_relations
                                feed_dict[i][self.candidate_entity_sequence[i]] = next_entities
                                feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i],
                                                                                                 self.per_example_logits[i],
                                                                                                 self.action_idx[i]],
                                                                                             feed_dict=feed_dict[i])
                                loss_before_regularization.append(per_example_loss)
                                logits.append(per_example_logits)
                            elif self.params['use_replay_memory'] and not self.params['use_joint_model']:
                                # use MINERVA and replay memory
                                feed_dict_test[i][self.candidate_relation_sequence[i]] = state['next_relations']
                                feed_dict_test[i][self.candidate_entity_sequence[i]] = state['next_entities']
                                feed_dict_test[i][self.entity_sequence[i]] = state['current_entities']
                                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i],
                                                                                                 self.per_example_logits[i],
                                                                                                 self.action_idx[i]],
                                                                                             feed_dict=feed_dict_test[i])
                                # store the chosen relation and entity at this time step, later store this into replay memory
                                relations = np.array(state['next_relations'])[np.arange(self.batch_size * self.num_rollouts), idx]
                                relation_trajectory.append(relations)
                                entity_trajectory.append(state['current_entities'])
                            elif self.params['use_replay_memory'] and self.params['use_joint_model']:
                                # use joint model and use replay memory
                                feed_dict_test[i][self.candidate_relation_sequence[i]] = next_relations
                                feed_dict_test[i][self.candidate_entity_sequence[i]] = next_entities
                                feed_dict_test[i][self.entity_sequence[i]] = state['current_entities']
                                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i],
                                                                                                 self.per_example_logits[i],
                                                                                                 self.action_idx[i]],
                                                                                             feed_dict=feed_dict_test[i])
                                relations = np.array(next_relations)[np.arange(self.batch_size * self.num_rollouts), idx]
                                relation_trajectory.append(relations)
                                entity_trajectory.append(state['current_entities'])
                                is_pcnn_edge_ = [1 if idx[edge] > pcnn_edge_idx[edge] else 0 for edge in range(len(idx))]
                                is_pcnn_edge.append(is_pcnn_edge_)

                            if self.params['use_joint_model']:
                                state['current_entities'] = np.array(next_entities)[np.arange(self.batch_size * self.num_rollouts), idx]
                            else:
                                state = episode(idx)

                # =========================== get rewards from this inference =========================
                if not self.params['use_joint_model']:
                    # get the final reward from the environment
                    rewards = episode.get_reward()
                    # computed cumulative discounted reward
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]
                else:
                    # if use joint model, need rewrite reward function
                    reward = (state['current_entities'] == episode.end_entities)
                    condlist = [reward == True, reward == False]
                    choicelist = [episode.positive_reward, episode.negative_reward]
                    rewards = np.select(condlist, choicelist)  # [B,]
                    # computed cumulative discounted reward
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

                # =========================== store to replay memory =========================
                if self.params['use_replay_memory']:
                    entity_trajectory.append(state['current_entities'])     # store the end entities
                    all_query_relation.append(episode.get_query_relation())

                    # reshape
                    entity_trajectory = np.column_stack(entity_trajectory)      # shape = (batch_size*rollouts, path_length+1)
                    relation_trajectory = np.column_stack(relation_trajectory)  # shape = (batch_size*rollouts, path_length)
                    all_query_relation = np.column_stack(all_query_relation)    # shape = (batch_size*rollouts, )

                    # find if it's positive sentence or not
                    positive_sequences = [i for i in range(len(rewards)) if rewards[i] == 1]
                    negative_sequences = [i for i in range(len(rewards)) if rewards[i] == 0]
                    logger.info("positive sequences: {0:4d}, negative sequences: {1:4d}".
                                format(len(positive_sequences), len(negative_sequences)))

                    if self.params['use_joint_model']:
                        is_pcnn_edge = np.column_stack(is_pcnn_edge)    # shape = (batch_size*rollouts, path_length)
                        path_pcnn_sum = np.sum(is_pcnn_edge, axis=1)    # shape = (batch_size*rollouts, )
                        positive_pcnn_sequences = [index for index in positive_sequences if path_pcnn_sum[index] > 0]
                        positive_gfaw_sequences = [index for index in positive_sequences if path_pcnn_sum[index] == 0]
                        negative_pcnn_sequences = [index for index in negative_sequences if path_pcnn_sum[index] > 0]
                        negative_gfaw_sequences = [index for index in negative_sequences if path_pcnn_sum[index] == 0]
                        positive_sequences = positive_pcnn_sequences + positive_gfaw_sequences
                        negative_sequences = negative_pcnn_sequences + negative_gfaw_sequences
                        logger.info("positive gfaw sequences: {0:4d}, positive pcnn sequences: {1:4d}, negative gfaw sequences: {2:4d}, negative pcnn sequences: {3:4d}".
                                    format(len(positive_gfaw_sequences), len(positive_pcnn_sequences), len(negative_gfaw_sequences), len(negative_pcnn_sequences)))

                        # enlarge edges in KG, store positive pcnn edge into KG -> add edges to self.train_environment.grapher.array_store
                        if len(positive_pcnn_sequences):
                            entity_paths = entity_trajectory[positive_pcnn_sequences]
                            relation_paths = relation_trajectory[positive_pcnn_sequences]
                            paths_pcnn_egde = is_pcnn_edge[positive_pcnn_sequences]
                            for i in range(len(paths_pcnn_egde)):  # [0, 0, 1]
                                for j in range(len(paths_pcnn_egde[i])):
                                    if paths_pcnn_egde[i][j]:  # this is pcnn edge
                                        e1 = entity_paths[i][j]
                                        r = relation_paths[i][j]
                                        e2 = entity_paths[i][j + 1]
                                        edge_name = str(e1) + '#' + str(r) + '#' + str(e2)
                                        if edge_name not in pcnn_pos_edge_appearance:
                                            pcnn_pos_edge_appearance[edge_name] = 1
                                        else:
                                            pcnn_pos_edge_appearance[edge_name] += 1
                                        logger.info(
                                            "positive pcnn edge: {0:4d}".format(sum(pcnn_pos_edge_appearance.values())))

                                        if pcnn_pos_edge_appearance[edge_name] < 5:
                                            # at most add 5 repeated edges
                                            # store e1, e2, r in the PCNN Memory
                                            self.pcnn_experience.insert('triples', [
                                                [self.rev_entity_vocab[e1], self.rev_entity_vocab[e2],
                                                 self.rev_relation_vocab[r]]])
                                            # Store e1, e2, r in the graph
                                            for row in range(self.max_num_actions):
                                                if self.train_environment.grapher.array_store[e1][row].sum() == 0:
                                                    # change the first nozero edge into (e1, e2, r)
                                                    self.train_environment.grapher.array_store[e1][row] = [e2, r]
                                                    continue
                                                elif row == (self.max_num_actions - 1):
                                                    # e1's outgoint degree > max_num_action, random replace one adge with (e1, e2, r)
                                                    self.train_environment.grapher.array_store[e1][
                                                        np.random.randint(self.max_num_actions)] = [e2, r]

                    # add to replay memory
                    if len(positive_sequences):
                        self.pos_experience.insert('entity_path', entity_trajectory[positive_sequences])
                        self.pos_experience.insert('relation_path', relation_trajectory[positive_sequences])
                        self.pos_experience.insert('path_rewards', rewards[positive_sequences])
                        self.pos_experience.insert('state_rewards', cum_discounted_reward[positive_sequences])
                        self.pos_experience.insert('query_relation', all_query_relation[positive_sequences])

                    if len(negative_sequences):
                        self.neg_experience.insert('entity_path', entity_trajectory[negative_sequences])
                        self.neg_experience.insert('relation_path', relation_trajectory[negative_sequences])
                        self.neg_experience.insert('path_rewards', rewards[negative_sequences])
                        self.neg_experience.insert('state_rewards', cum_discounted_reward[negative_sequences])
                        self.neg_experience.insert('query_relation', all_query_relation[negative_sequences])
                else:
                    loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # =========================== iteration backprop =========================
            if self.params['train_pcnn']:
                if 'triples' not in self.pcnn_experience.memory or len(self.pcnn_experience.memory['triples']) < self.params['pcnn_batch_size']:
                    print("PCNN replay memory size < PCNN batch size, skip PCNN backward!")
                else:
                    print("PCNN backward...")
                    batch_pcnn_triples = self.sample_PCNN(framework)
                    batch_data = framework.test_data_loader.batch_gen_train(batch_pcnn_triples)
                    framework.backward(batch_data)

            print_flag = 0
            if not self.params['use_replay_memory']:
                # backprop as usual
                batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                       feed_dict={self.cum_discounted_reward: cum_discounted_reward})
                print_flag = 1
            else:
                if 'entity_path' not in self.neg_experience.memory:
                    if 'entity_path' not in self.pos_experience.memory:
                        print("Replay memory size < batch: skip")
                    elif (len(self.pos_experience.memory['entity_path']) < self.batch_size * self.num_rollouts):
                        print("Replay memory size < GFAW batch size, skip GFAW positive backward!")
                    else:
                        print("GFAW positive backward!")
                        with sess.as_default():
                            with sess.graph.as_default():
                                h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                                # sampling from positive MINERVA replay memory
                                feed_dict, rewards, cum_discounted_reward = self.sample_pos_GFAW()

                                for i in range(self.path_length):
                                    per_example_loss, per_example_logits, idx = sess.partial_run(h,
                                                                                                 [self.per_example_loss[i],
                                                                                                  self.per_example_logits[i],
                                                                                                  self.action_idx[i]],
                                                                                                 feed_dict=feed_dict[i])
                                    loss_before_regularization.append(per_example_loss)
                                    logits.append(per_example_logits)

                                loss_before_regularization = np.stack(loss_before_regularization, axis=1)

                                # backprop
                                batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                                       feed_dict={self.cum_discounted_reward: cum_discounted_reward})
                                print_flag = 1

                elif (len(self.neg_experience.memory['entity_path']) < self.params['sample_RM_neg_ratio']*self.batch_size*self.num_rollouts) or (len(self.pos_experience.memory['entity_path']) < (1-self.params['sample_RM_neg_ratio'])*self.batch_size*self.num_rollouts):
                    print("Replay memory size < GFAW batch sample size, skip GFAW backward!")
                else:
                    print("GFAW batch sample backward!")
                    with sess.as_default():
                        with sess.graph.as_default():
                            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                            # use a ratio of neg & pos to sample
                            feed_dict, rewards, cum_discounted_reward = self.sample_GFAW(self.params['sample_RM_neg_ratio'])
                            for i in range(self.path_length):
                                per_example_loss, per_example_logits, idx = sess.partial_run(h,
                                                                                             [self.per_example_loss[i],
                                                                                              self.per_example_logits[i],
                                                                                              self.action_idx[i]],
                                                                                             feed_dict=feed_dict[i])
                                loss_before_regularization.append(per_example_loss)
                                logits.append(per_example_logits)

                            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

                            # backprop
                            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                                   feed_dict={self.cum_discounted_reward: cum_discounted_reward})
                            print_flag = 1

            # =========================== print log =========================
            if print_flag:
                # print statistics
                train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
                avg_reward = np.mean(rewards)
                # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
                # entity pair, atleast one of the path get to the right answer
                reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
                reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
                reward_reshape = (reward_reshape > 0)
                num_ep_correct = np.sum(reward_reshape)
                if np.isnan(train_loss):
                    raise ArithmeticError("Error in computing loss")

                logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                            "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                            format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                   (num_ep_correct / self.batch_size),
                                   train_loss))

            if self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                if not os.path.exists(self.path_logger_file + "/" + str(self.batch_counter)):
                    os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess, beam=False, print_paths=False, save_model=True, auc = False):
        batch_counter = 0
        paths = defaultdict(list)
        self.test_environment.mode = "test"
        answers = []
        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_50 = 0
        all_final_reward_100 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            temp_batch_size = episode.no_examples

            self.qr = episode.get_query_relation()
            feed_dict[self.query_relation] = self.qr
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger rl_code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    feed_dict[self.first_state_of_test] = True
                feed_dict[self.next_relations] = state['next_relations']
                feed_dict[self.next_entities] = state['next_entities']
                feed_dict[self.current_entities] = state['current_entities']
                feed_dict[self.prev_state] = agent_mem
                feed_dict[self.prev_relation] = previous_relation

                loss, agent_mem, test_scores, test_action_idx, chosen_relation = sess.run(
                    [ self.test_loss, self.test_state, self.test_logits, self.test_action_idx, self.chosen_relation],
                    feed_dict=feed_dict)
                # test_action_idx, chosen_relation  shape=(batch_size*test_rollouts, )
                # test_scores  shape=(batch_size*test_rollouts, max_num_actions)

                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]
                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]

                    del idx, y, x

                previous_relation = chosen_relation

                ####logger rl_code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]

                del loss, test_scores, test_action_idx, chosen_relation

            if beam:
                self.log_probs = beam_probs

            ####Logger rl_code####

            if print_paths:
                self.entity_trajectory.append(
                    state['current_entities'])

            # ask environment for final reward
            rewards = episode.get_reward()  # [B*test_rollouts]
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_50 = 0
            final_reward_100 = 0
            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:   # find the highest rank in the test_rollouts path for each example
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                        # print("answer: ", answer)
                        # print("sorted_answers: ", sorted_answers)
                    else:
                        answer_pos = None

                if answer_pos != None:
                    if answer_pos < 100:
                        final_reward_100 += 1
                        if answer_pos < 50:
                            final_reward_50 += 1
                            if answer_pos < 10:
                                final_reward_10 += 1
                                if answer_pos < 5:
                                    final_reward_5 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0/((answer_pos+1))
                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    for r in sorted_indx[b]:
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')
                        paths[str(qr)].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')
                    paths[str(qr)].append("#####################\n")

                    del qr, start_e, end_e

            del beam_probs, state, agent_mem, previous_relation
            del rewards, reward_reshape, sorted_indx, ce, se

            all_final_reward_1 += final_reward_1
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_50 += final_reward_50
            all_final_reward_100 += final_reward_100
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_50 /= total_examples
        all_final_reward_100 /= total_examples
        auc /= total_examples
        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')
            np.save(file=os.path.join(self.model_dir, 'new_graph.npy'), arr=self.train_environment.grapher.array_store)

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@50: {0:7.4f}".format(all_final_reward_50))
            score_file.write("\n")
            score_file.write("Hits@100: {0:7.4f}".format(all_final_reward_100))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        with open('results.txt', 'a') as score_file:
            score_file.write(self.params['data_input_dir'])
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@50: {0:7.4f}".format(all_final_reward_50))
            score_file.write("\n")
            score_file.write("Hits@100: {0:7.4f}".format(all_final_reward_100))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@50: {0:7.4f}".format(all_final_reward_50))
        logger.info("Hits@100: {0:7.4f}".format(all_final_reward_100))
        logger.info("auc: {0:7.4f}".format(auc))

        del paths

        self.test_environment.mode = "train"
        gc.collect()


    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))

