


import sys
import os
import json
import nrekit
import numpy as np
import tensorflow as tf


dataset_name = "FB15K-237+NYT/text"
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
# dataset_name = 'pcnn_small'

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
dataset_dir = os.path.join('/data/base/Joint-Datasets/', dataset_name)
if not os.path.isdir(dataset_dir):
    raise Exception("[ERROR] Dataset dir %s doesn't exist!" % (dataset_dir))

# The first 3 parameters are train / test data file name, word embedding file name and relation-id mapping file name respectively.
train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train.json'),
                                                        os.path.join(dataset_dir, 'word_vec.json'),
                                                        os.path.join(dataset_dir, 'rel2id.json'),
                                                        mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                        shuffle=True, batch_size=60, case_sensitive=False,
                                                        reprocess=False)
test_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'test.json'),
                                                       os.path.join(dataset_dir, 'word_vec.json'),
                                                       os.path.join(dataset_dir, 'rel2id.json'),
                                                       mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                                       shuffle=False, batch_size=60, case_sensitive=False,
                                                       reprocess=False)

framework = nrekit.framework.re_framework(train_loader, test_loader)


class model(nrekit.framework.re_model):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")

        # Embedding
        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

        # Encoder
        if model.encoder == "pcnn":
            x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
        elif model.encoder == "cnn":
            x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
            x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
        elif model.encoder == "rnn":
            x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
        elif model.encoder == "birnn":
            x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
        else:
            raise NotImplementedError

        # Selector
        if model.selector == "att":
            self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label,
                                                                                   self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label,
                                                                                 self.rel_tot, False, keep_prob=1.0)
        elif model.selector == "ave":
            self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot,
                                                                                 keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot,
                                                                               keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        elif model.selector == "max":
            self._train_logit, train_repre = nrekit.network.selector.bag_maximum(x_train, self.scope, self.ins_label,
                                                                                 self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_maximum(x_test, self.scope, self.ins_label,
                                                                               self.rel_tot, False, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        else:
            raise NotImplementedError

        # Classifier
        self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot,
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


# # if len(sys.argv) > 2:
#     model.encoder = sys.argv[2]
# if len(sys.argv) > 3:
#     model.selector = sys.argv[3]

model.encoder = 'pcnn'
model.selector = 'att'

pred_result = framework.test(model, ckpt="/data/base/Joint-Checkpoint/FB15K-237+NYT/text_pcnn_att", return_result=True)

with open("/data/base/Joint-Datasets/FB15K-237+NYT/text/text_pcnn_att_pred.json", 'w') as outfile:
    json.dump(pred_result, outfile)

# sorted_pred_result = sorted(pred_result, key=lambda x: x['score'], reverse=True)
# with open('./test_result/' + dataset_name + "_" + model.encoder + "_" + model.selector + "_pred.txt", 'w') as outfile:
#     for score_dict in sorted_pred_result:
#         entpair = score_dict['entpair'].decode("utf-8").split('#')
#         entpair.append(str(score_dict['relation']))
#         entpair.append(str(score_dict['score']))
#         outfile.write('\t'.join(entpair))
#         outfile.write('\n')

for etypair in pred_result.keys():
    top1 = pred_result[etypair][:1]
    # top5 = pred_result[etypair][:5]
    # top10 = pred_result[etypair][:10]
    with open('/data/base/Joint-Datasets/FB15K-237+NYT/text/top1.txt', 'a') as outfile:
        for score_dict in top1:
            entpair = etypair.split('#')
            entpair.append(str(score_dict['relation']))
            entpair.append(str(score_dict['score']))
            outfile.write('\t'.join(entpair))
            outfile.write('\n')
    # with open('./test_result/top5.txt', 'a') as outfile:
    #     for score_dict in top5:
    #         entpair = etypair.split('#')
    #         entpair.append(str(score_dict['relation']))
    #         entpair.append(str(score_dict['score']))
    #         outfile.write('\t'.join(entpair))
    #         outfile.write('\n')
    # with open('./test_result/top10.txt', 'a') as outfile:
    #     for score_dict in top10:
    #         entpair = etypair.split('#')
    #         entpair.append(str(score_dict['relation']))
    #         entpair.append(str(score_dict['score']))
    #         outfile.write('\t'.join(entpair))
    #         outfile.write('\n')
