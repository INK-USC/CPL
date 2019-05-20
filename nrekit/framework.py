import tensorflow as tf
import os
import sklearn.metrics
import numpy as np
import sys
import time


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class re_model:
    def __init__(self, train_data_loader, batch_size, max_length=120):
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='pos2')
        self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='label')
        self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name='ins_label')
        self.length = tf.placeholder(dtype=tf.int32, shape=[None], name='length')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name='scope')
        self.train_data_loader = train_data_loader
        self.rel_tot = train_data_loader.rel_tot
        self.word_vec_mat = train_data_loader.word_vec_mat

    def loss(self):
        raise NotImplementedError
    
    def train_logit(self):
        raise NotImplementedError
    
    def test_logit(self):
        raise NotImplementedError

class re_framework:
    MODE_BAG = 0  # Train and test the model at bag level.
    MODE_INS = 1  # Train and test the model at instance level

    def __init__(self, train_data_loader, test_data_loader, max_length=120, batch_size=160):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.sess = None

    def one_step_multi_models(self, sess, models, batch_data_gen, run_array, return_label=True):
        """
        :param sess:
        :param models:
        :param batch_data_gen:
        :param run_array: for train: run_array = [loss, train_logit, train_op]
                          train_logit, shape = (batch_size, relation_total), train_op = Gradient Descent
        :param return_label:
        :return: iter_loss: float32
                 iter_logit: shape = (batch_size, relation_total), probability distribution of relations for one batch
                 _train_op:
                 iter_label: shape = (batch_size, )
        """
        feed_dict = {}
        batch_label = []
        for model in models:       # for each model on gpu
            batch_data = batch_data_gen.next_batch(batch_data_gen.batch_size // len(models))
            feed_dict.update({
                model.word: batch_data['word'],     # shape = (instance_nums, max_length), Use PAD to make sure the same length
                model.pos1: batch_data['pos1'],     # shape = (instance_nums, max_length), each word has a position
                model.pos2: batch_data['pos2'],     # shape = (instance_nums, max_length)
                model.label: batch_data['rel'],     # shape = (batch_size/gpu_nums, ), ?
                model.ins_label: batch_data['ins_rel'],     # shape = (instance_nums, ), label each instance with a relation
                model.scope: batch_data['scope'],       # shape = (batch_size/gpu_nums, 2)
                model.length: batch_data['length'],     # shape = (instance_nums, ), real length of each instance
            })
            if 'mask' in batch_data and hasattr(model, "mask"):     # shape = (instance_nums, max_length), 1, 2, 3, 0
                feed_dict.update({model.mask: batch_data['mask']})
            batch_label.append(batch_data['rel'])
        result = sess.run(run_array, feed_dict)
        batch_label = np.concatenate(batch_label)   # shape = (batch_size, )
        if return_label:
            result += [batch_label]
        ## new rl_code added
        # for debug why is loss becoming inf
        # if result[0] > 1000000:
        #     print(feed_dict)
        ## new rl_code ended
        return result

    def one_step(self, sess, model, batch_data, run_array):  # for test, only use one gpu
        """
        :param sess:
        :param model:
        :param batch_data:
                'word', shape = (instance_nums, max_length), Use PAD to make sure the same length
                'pos1'&'pos2', shape = (instance_nums, max_length), each word has a position
                'rel', shape = (batch_size, ), ?
                'ins_rel', shape = (instance_nums, ), label each instance with a relation
                'scope', shape = (batch_size, 2), etypair2scope = ety1#ety2 -> [index1, index2]
                         relfact2scope = ety1#ety2#rlt -> [index1, index2]
                'length', shape = (instance_nums, ), real length of each instance
                'multi_rel', shape = (batch_size, relation_total), in one row, if entpair have relation, then the scalar=1, or=0
                'entpair', list, len = batch_size
                'mask', shape = (instance_nums, max_length)
        :param run_array: [model.test_logit()]
        :return:
            result[0], iter_logit, shape = (batch_size, relation_total)
        """
        feed_dict = {
            model.word: batch_data['word'],
            model.pos1: batch_data['pos1'],
            model.pos2: batch_data['pos2'],
            model.label: batch_data['rel'],
            model.ins_label: batch_data['ins_rel'],
            model.scope: batch_data['scope'],
            model.length: batch_data['length'],
        }
        if 'mask' in batch_data and hasattr(model, "mask"):     # shape = (instance_nums, max_length), 1, 2, 3, 0
            feed_dict.update({model.mask: batch_data['mask']})
        result = sess.run(run_array, feed_dict)
        return result

    def train(self,
              model,
              model_name,
              ckpt_dir='./checkpoint',
              summary_dir='./summary',
              test_result_dir='./test_result',
              learning_rate=0.01,
              max_epoch=60,
              pretrain_model=None,
              test_epoch=1,
              optimizer=tf.train.GradientDescentOptimizer,
              gpu_nums=1):
        
        assert(self.train_data_loader.batch_size % gpu_nums == 0)
        print("Start training...")
        
        # Init
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        optimizer = optimizer(learning_rate)
        
        # Multi GPUs use
        tower_grads = []
        tower_models = []
        for gpu_id in range(gpu_nums):
            with tf.device("/gpu:%d" % gpu_id):
                with tf.name_scope("gpu_%d" % gpu_id):
                    cur_model = model(self.train_data_loader, self.train_data_loader.batch_size // gpu_nums, self.train_data_loader.max_length)
                    tower_grads.append(optimizer.compute_gradients(cur_model.loss()))
                    tower_models.append(cur_model)
                    tf.add_to_collection("loss", cur_model.loss())
                    tf.add_to_collection("train_logit", cur_model.train_logit())

        loss_collection = tf.get_collection("loss")
        loss = tf.add_n(loss_collection) / len(loss_collection)
        train_logit_collection = tf.get_collection("train_logit")
        train_logit = tf.concat(train_logit_collection, 0)

        grads = average_gradients(tower_grads)
        ## new rl_code added
        # use gradient clipping to avoid the loss becoming inf
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, 5)  # 5 is a hyper_para
        train_op = optimizer.apply_gradients(zip(capped_grads, variables))
        # train_op = optimizer.apply_gradients(grads)
        ## new rl_code ended
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        # Saver
        saver = tf.train.Saver(max_to_keep=None)
        if pretrain_model is None:
            print("Do not have pretrained models...")
            self.sess.run(tf.global_variables_initializer())
        else:
            print("Restoring from pretrained models...")
            saver.restore(self.sess, pretrain_model)

        # Training
        # best_metric = 0
        # best_prec = None
        # best_recall = None
        best_loss = float("inf")
        not_best_count = 0 # Stop training after several epochs without improvement.
        for epoch in range(max_epoch):
            print('###### Epoch ' + str(epoch) + ' ######')
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            while True:
                time_start = time.time()
                try:
                    iter_loss, iter_logit, _train_op, iter_label = self.one_step_multi_models(self.sess, tower_models, self.train_data_loader, [loss, train_logit, train_op])
                except StopIteration:
                    break
                time_end = time.time()
                t = time_end - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)     # shape = (batch_size, ), predict one relation for each row
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                    sys.stdout.flush()
                i += 1
            print("\nAverage iteration time: %f" % (time_sum / i))

            if iter_loss < best_loss:
                best_loss = iter_loss
                print("Best model, storing...")
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)
                path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
                print("Finish storing")
                not_best_count = 0
            else:
                not_best_count += 1

            if not_best_count >= 20:
                break

            # if (epoch + 1) % test_epoch == 0:
            #     metric = self.test(model)
            #     if metric > best_metric:
            #         best_metric = metric
            #         best_prec = self.cur_prec
            #         best_recall = self.cur_recall
            #         print("Best model, storing...")
            #         if not os.path.isdir(ckpt_dir):
            #             os.mkdir(ckpt_dir)
            #         path = saver.save(self.sess, os.path.join(ckpt_dir, model_name))
            #         print("Finish storing")
            #         not_best_count = 0
            #     else:
            #         not_best_count += 1
            #
            # if not_best_count >= 20:
            #     break
        
        print("######")
        print("Finish training " + model_name)
        print("Best loss = %f" % (best_loss))
        # print("Best epoch auc = %f" % (best_metric))
        # if (not best_prec is None) and (not best_recall is None):
        #     if not os.path.isdir(test_result_dir):
        #         os.mkdir(test_result_dir)
        #     np.save(os.path.join(test_result_dir, model_name + "_x.npy"), best_recall)
        #     np.save(os.path.join(test_result_dir, model_name + "_y.npy"), best_prec)


    # ================== start for Joint Model ===================== #

    def partial_run_setup(self,
                          model,
                          model_name,
                          ckpt_dir='./checkpoint',
                          summary_dir='./summary',
                          test_result_dir='./test_result',
                          learning_rate=0.01,
                          max_epoch=60,
                          pretrain_model=None,
                          test_epoch=1,
                          optimizer=tf.train.GradientDescentOptimizer,
                          gpu_nums=1):
        """ initialize model and setup the handler of tf.partial_run for joint model
        :return: hander of tf.partial_run
        """

        assert (self.test_data_loader.batch_size % gpu_nums == 0)
        print("Start training...")

        # optimizer setup
        optimizer = optimizer(learning_rate)

        # Single GPU use
        tower_grads = []
        tower_models = []
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device("/gpu:1" ):
                cur_model = model(self.test_data_loader, self.test_data_loader.batch_size,
                                  self.test_data_loader.max_length)
                tower_grads.append(optimizer.compute_gradients(cur_model.loss()))
                tower_models.append(cur_model)
                tf.add_to_collection("loss", cur_model.loss())
                tf.add_to_collection("test_logit", cur_model.test_logit())

                # some model operations setup
                self.loss_collection = tf.get_collection("loss")
                self.loss = tf.add_n(self.loss_collection) / len(self.loss_collection)
                self.test_logit_collection = tf.get_collection("test_logit")
                self.test_logit = tf.concat(self.test_logit_collection, 0)

                grads = average_gradients(tower_grads)
                ## new rl_code added
                # use gradient clipping to avoid the loss becoming inf
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(gradients, 5)  # 5 is a hyper_para
                self.train_op = optimizer.apply_gradients(zip(capped_grads, variables))
                # train_op = optimizer.apply_gradients(grads)
                ## new rl_code ended

                # Saver
                saver = tf.train.Saver(max_to_keep=None)

                # Init
                config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
                self.sess = tf.Session(config=config, graph=self.graph)
                self.optimizer = optimizer

                if pretrain_model is None:
                    print("Do not have pretrained models...")
                    self.sess.run(tf.global_variables_initializer())
                else:
                    print("Restoring from pretrained models...")
                    saver.restore(self.sess, pretrain_model)

                # summary_writer for TensorBoard
                self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

                # initialize loss recorder
                self.best_loss = float('inf')
                self.not_best_count = 0

                # setup the handler of tf.partial_run for joint model,
                # read document about tf.partial_run_setup for more details
                self.run_array = [self.loss, self.test_logit, self.train_op]

                handlers = []
                self.models = tower_models
                for model in self.models:
                    # Fetches and Feeds
                    handler = self.sess.partial_run_setup(self.run_array,
                             [model.word, model.pos1, model.pos2, model.label, model.ins_label, model.scope, model.length, model.mask])
                    handlers.append(handler)
                self.handlers = handlers

    def predict(self, current_entities_list, id_entities_dict, type="test"):
        """ get the most possible relations and next entities for this current_entities list

        :param current_entiries_list: the ids of current entities in GFAW, shape: (batch_size_GFAW, )
        :param id_entities_dict: the id dict for entites ids -> naive entities name, for data_loader.gen_batches
        :return: get the most possible (next_relations, next_entities), shape: (batch_size_GFAW, batch_size_PCNN)
        """

        results = []
        run_array = [self.test_logit] # get results, shape: (batch_size_PCNN, relation_total)
        current_nodes = [id_entities_dict[i] for i in current_entities_list]
        entpairs = []
        for current_node in current_nodes:
            if type == "test":
                batch_data = self.test_data_loader.batch_gen(current_node)

            if batch_data is not None:
                feed_dict = {}
                for model in self.models:  # for each model on gpu
                    feed_dict.update({
                        model.word: batch_data['word'],  # shape = (instance_nums, max_length), Use PAD to make sure the same length
                        model.pos1: batch_data['pos1'],  # shape = (instance_nums, max_length), each word has a position
                        model.pos2: batch_data['pos2'],  # shape = (instance_nums, max_length)
                        model.label: batch_data['rel'],  # shape = (batch_size/gpu_nums, ), ?
                        model.ins_label: batch_data['ins_rel'],  # shape = (instance_nums, ), label each instance with a relation
                        model.scope: batch_data['scope'],  # shape = (batch_size/gpu_nums, 2)
                        model.length: batch_data['length'],  # shape = (instance_nums, ), real length of each instance
                    })

                    if 'mask' in batch_data and hasattr(model, "mask"):  # shape = (instance_nums, max_length), 1, 2, 3, 0
                        feed_dict.update({model.mask: batch_data['mask']})

                entpairs.append(batch_data['entpair'])

                results.extend(self.get_results(run_array, feed_dict))
            else:
                results.append([])
                entpairs.append([])

        return entpairs, results


    def get_results(self, run_array=None, feed_dict=None):
        """ get results from PCNN without backward and parameters update.
        Before this function we should call `partial_run_setup` at first.

        :param run_array: list of operation in this model, eg. [model.train_op]
        :param feed_dict: dict of feeding data for this model, eg. {model.word: batch_data[word]}
        :return: the results calculated by run_array from feed_dict
        """

        # if feed_dict is None, we use the default operations and feeding data to test the function
        if feed_dict is None:
            feed_dict = {}
            batch_label = []
            for model in self.models:  # for each model on gpu
                batch_data = self.test_data_loader.next_batch(self.test_data_loader.batch_size // len(self.models))
                feed_dict.update({
                    model.word: batch_data['word'],  # shape = (instance_nums, max_length), Use PAD to make sure the same length
                    model.pos1: batch_data['pos1'],  # shape = (instance_nums, max_length), each word has a position
                    model.pos2: batch_data['pos2'],  # shape = (instance_nums, max_length)
                    model.label: batch_data['rel'],  # shape = (batch_size/gpu_nums, ), ?
                    model.ins_label: batch_data['ins_rel'],  # shape = (instance_nums, ), label each instance with a relation
                    model.scope: batch_data['scope'],  # shape = (batch_size/gpu_nums, 2)
                    model.length: batch_data['length'],  # shape = (instance_nums, ), real length of each instance
                })
                if 'mask' in batch_data and hasattr(model, "mask"):  # shape = (instance_nums, max_length), 1, 2, 3, 0
                    feed_dict.update({model.mask: batch_data['mask']})
                batch_label.append(batch_data['rel'])

        if run_array is None:
            run_array = [self.logit]

        # switch the graph and session from GFAW to PCNN
        with self.sess.as_default():
            with self.sess.graph.as_default():
                results = self.sess.run(run_array, feed_dict)
                return results

    def backward(self, batch_data=None, run_array=None):
        """ update the parameteres of this model through `self.train_op`

        :param run_array: list of operation in this model to calculate loss, eg. [model.train_loss]
        :param feed_dict: dict of feeding data for this model, eg. {model.word: batch_data['word]}
        :return: the results calculated by run_array from feed_dict, eg. the value of loss
        """

        # if feed_dict is None, we use the default operations and feeding data to test the function
        feed_dict = {}
        batch_label = []
        for model in self.models:  # for each model on gpu
            # batch_data = self.train_data_loader.next_batch(self.train_data_loader.batch_size // len(self.models))
            feed_dict.update({
                model.word: batch_data['word'],
                # shape = (instance_nums, max_length), Use PAD to make sure the same length
                model.pos1: batch_data['pos1'],  # shape = (instance_nums, max_length), each word has a position
                model.pos2: batch_data['pos2'],  # shape = (instance_nums, max_length)
                model.label: batch_data['rel'],  # shape = (batch_size/gpu_nums, ), ?
                model.ins_label: batch_data['ins_rel'],
                # shape = (instance_nums, ), label each instance with a relation
                model.scope: batch_data['scope'],  # shape = (batch_size/gpu_nums, 2)
                model.length: batch_data['length'],  # shape = (instance_nums, ), real length of each instance
            })
            if 'mask' in batch_data and hasattr(model, "mask"):  # shape = (instance_nums, max_length), 1, 2, 3, 0
                feed_dict.update({model.mask: batch_data['mask']})
            batch_label.append(batch_data['rel'])

        if run_array is None:
            run_array = [self.loss, self.train_op] # call bp operation

        # switch the graph and session from GFAW to PCNN
        with self.sess.as_default():
            with self.sess.graph.as_default():
                loss, _ = self.sess.run(run_array, feed_dict)

    # ================== end for Joint Model ===================== #

    def test(self,
             model,
             # current_node,
             ckpt=None,
             return_result=False,
             mode=MODE_BAG):
        if mode == re_framework.MODE_BAG:
            # return self.__test_bag__(model, current_node, ckpt=ckpt, return_result=return_result)
            return self.__test_bag__(model, ckpt=ckpt, return_result=return_result)
        elif mode == re_framework.MODE_INS:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
    def __test_bag__(self, model, ckpt=None, return_result=False):  # current_node,
        print("Testing...")
        if self.sess == None:
            self.sess = tf.Session()
        model = model(self.test_data_loader, self.test_data_loader.batch_size, self.test_data_loader.max_length)
        if not ckpt is None:
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt)
        # tot_correct = 0
        # tot_not_na_correct = 0
        # tot = 0
        # tot_not_na = 0
        entpair_tot = 0
        # test_result = []
        pred_result = {}
        sorted_pred_result = {}
         
        for i, batch_data in enumerate(self.test_data_loader):
        # for i, batch_data in enumerate(self.test_data_loader.batches_gen(batch_size=self.test_data_loader.batch_size, current_node='C0086418')):  # eg. batch_size = 1, current_node = 'C0086418'
            iter_logit = self.one_step(self.sess, model, batch_data, [model.test_logit()])[0]
            iter_output = iter_logit.argmax(-1)     # shape = (batch_size, )
            # iter_correct = (iter_output == batch_data['rel']).sum()
            # iter_not_na_correct = np.logical_and(iter_output == batch_data['rel'], batch_data['rel'] != 0).sum()
            # tot_correct += iter_correct
            # tot_not_na_correct += iter_not_na_correct
            # tot += batch_data['rel'].shape[0]
            # tot_not_na += (batch_data['rel'] != 0).sum()
            # if tot_not_na > 0:
            #     sys.stdout.write("[TEST] step %d | not NA accuracy: %f, accuracy: %f\r" % (i, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
            #     sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, self.test_data_loader.rel_tot):
                    # test_result.append({'score': iter_logit[idx][rel], 'flag': batch_data['multi_rel'][idx][rel]})
                    if batch_data['entpair'][idx] != "None#None":
                        entpair = batch_data['entpair'][idx]   # .encode('utf-8')
                        if entpair not in pred_result:
                            pred_result[entpair] = []
                            pred_result[entpair].append({'score': float(iter_logit[idx][rel]), 'relation': rel})
                        else:
                            pred_result[entpair].append({'score': float(iter_logit[idx][rel]), 'relation': rel})
                entpair_tot += 1
        # sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        for entpair in pred_result:
            sorted_pred_result[entpair] = sorted(pred_result[entpair], key=lambda x: x['score'], reverse=True)

        # prec = []
        # recall = []
        # correct = 0
        # for i, item in enumerate(sorted_test_result[::-1]):
        #     correct += item['flag']
        #     prec.append(float(correct) / (i + 1))
        #     recall.append(float(correct) / self.test_data_loader.relfact_tot)
        # auc = sklearn.metrics.auc(x=recall, y=prec)
        # print("\n[TEST] auc: {}".format(auc))
        # print("Finish testing")
        # self.cur_prec = prec
        # self.cur_recall = recall
        #
        # if not return_result:
        #     return auc
        # else:
        #     return (auc, pred_result)
        return sorted_pred_result
