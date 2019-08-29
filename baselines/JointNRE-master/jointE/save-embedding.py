import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys
import ctypes
import threading

export_path = "../data/" + sys.argv[1] + "/"
itera = int(sys.argv[2])

word_vec = np.load(export_path + 'vec.npy')
f = open(export_path + "config", 'r')
config = json.loads(f.read())
f.close()

ll = ctypes.cdll.LoadLibrary   
lib = ll("./init.so")
lib.setInPath(export_path)
lib.init()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nbatch_kg',100,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.001,'learning rate for kg')
tf.app.flags.DEFINE_integer('ent_total',lib.getEntityTotal(),'total of entities')
tf.app.flags.DEFINE_integer('rel_total',lib.getRelationTotal(),'total of relations')
tf.app.flags.DEFINE_integer('tri_total',lib.getTripleTotal(),'total of triples')
tf.app.flags.DEFINE_integer('katt_flag', 1, '1 for katt, 0 for att')

tf.app.flags.DEFINE_string('model', 'pcnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', config['textual_rel_total'],'maximum of relations')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch',20,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate for nn')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')


def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output
	else:
		print 'Make Shape Error!'

def main(_):

	print 'reading word embedding'
	word_vec = np.load(export_path + 'vec.npy')
	print 'reading training data'
	
	instance_triple = np.load(export_path + 'train_instance_triple.npy')
	instance_scope = np.load(export_path + 'train_instance_scope.npy')
	train_len = np.load(export_path + 'train_len.npy')
	train_label = np.load(export_path + 'train_label.npy')
	train_word = np.load(export_path + 'train_word.npy')
	train_pos1 = np.load(export_path + 'train_pos1.npy')
	train_pos2 = np.load(export_path + 'train_pos2.npy')
	train_mask = np.load(export_path + 'train_mask.npy')
	train_head = np.load(export_path + 'train_head.npy')
	train_tail = np.load(export_path + 'train_tail.npy')

	print 'reading finished'
	print 'mentions 		: %d' % (len(instance_triple))
	print 'sentences		: %d' % (len(train_len))
	print 'relations		: %d' % (FLAGS.num_classes)
	print 'word size		: %d' % (len(word_vec[0]))
	print 'position size 	: %d' % (FLAGS.pos_size)
	print 'hidden size		: %d' % (FLAGS.hidden_size)
	reltot = {}
	for index, i in enumerate(train_label):
		if not i in reltot:
			reltot[i] = 1.0
		else:
			reltot[i] += 1.0
	for i in reltot:
		reltot[i] = 1/(reltot[i] ** (0.05)) 
	print 'building network...'
	sess = tf.Session()
	if FLAGS.model.lower() == "cnn":
		model = network.CNN(is_training = True, word_embeddings = word_vec)
	elif FLAGS.model.lower() == "pcnn":
		model = network.PCNN(is_training = True, word_embeddings = word_vec)
	elif FLAGS.model.lower() == "lstm":
		model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "gru":
		model = network.RNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
		model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
		model = network.BiRNN(is_training = True, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	
	global_step = tf.Variable(0,name='global_step',trainable=False)
	global_step_kg = tf.Variable(0,name='global_step_kg',trainable=False)
	tf.summary.scalar('learning_rate', FLAGS.learning_rate)
	tf.summary.scalar('learning_rate_kg', FLAGS.learning_rate_kg)

	optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
	grads_and_vars = optimizer.compute_gradients(model.loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

	optimizer_kg = tf.train.GradientDescentOptimizer(FLAGS.learning_rate_kg)
	grads_and_vars_kg = optimizer_kg.compute_gradients(model.loss_kg)
	train_op_kg = optimizer_kg.apply_gradients(grads_and_vars_kg, global_step = global_step_kg)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)
	print ("================================================")
	saver.restore(sess, FLAGS.model_dir + sys.argv[1] + FLAGS.model + str(FLAGS.katt_flag)+"-"+str(itera))
	ent_embedding, rel_embedding = sess.run([model.word_embedding, model.rel_embeddings])
	ent_embedding = ent_embedding.tolist()
	rel_embedding = rel_embedding.tolist()
	f = open(export_path + "entity2vec", "w")
	f.write(json.dumps(ent_embedding))
	f.close()
	f = open(export_path + "relation2vec", "w")
	f.write(json.dumps(rel_embedding))
	f.close()

if __name__ == "__main__":
	tf.app.run() 
