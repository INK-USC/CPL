import config
import models
import tensorflow as tf
import numpy as np
import os
import sys
import json


#Input training files from benchmarks/FB15K/ folder.
con = config.Config()
#True: Input test files from the same folder.
con.set_in_path("./benchmarks/" + sys.argv[1] + "/")
con.set_test_link_prediction(True)
# con.set_test_triple_classification(True)
con.set_work_threads(1)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(50)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
# con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)

f = open("../JointNRE-master/entity2vec", "r")
entity2vec = json.loads(f.read())
entity2vec = np.array(entity2vec)
f.close()

print("====")
print len(entity2vec)
print len(entity2vec[0])
print (con.lib.getEntityTotal())
entity2vec = entity2vec[:con.lib.getEntityTotal(),:]

f = open("../JointNRE-master/relation2vec", "r")
relation2vec = json.loads(f.read())
relation2vec = np.array(relation2vec)
f.close()
print len(relation2vec)
print len(relation2vec[0])
print (con.lib.getRelationTotal())

relation2vec = relation2vec[:con.lib.getRelationTotal(), :]
#Train the model.
con.set_parameters_by_name("ent_embeddings", entity2vec)
con.set_parameters_by_name("rel_embeddings", relation2vec)

# con.run()
#To test models after training needs "set_test_flag(True)".
con.test()
# con.predict_head_entity(152, 9, 5)
# con.predict_tail_entity(151, 9, 5)
# con.predict_relation(151, 152, 5)
# con.predict_triple(151, 152, 9)
# con.predict_triple(151, 152, 8)
