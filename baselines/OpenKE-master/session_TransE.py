import config
import models
import tensorflow as tf
import numpy as np
import os
import sys
import time

if __name__ == "__main__":

    alpha = 0.001
    margin = 2.0
    train_times = 600
    nbatches = 100
    addr = sys.argv[1]

    stamp = time.strftime("%H%M%S%j")

    con = config.Config()
    con.set_in_path(addr + "/")

    print("Using TransE.")
    #Model Configurations
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(False)
    con.set_work_threads(4)
    con.set_train_times(train_times)
    con.set_nbatches(nbatches)
    con.set_alpha(alpha)
    con.set_margin(margin)
    con.set_bern(0)
    con.set_dimension(100)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")

    # Export position
    con.set_export_files("./output/CS1--model_TransE-" + stamp + ".vec.tf", 0)
    con.set_out_files("./res/CS1--embedding_TransE-" + stamp + ".vec.json")

    con.init()
    con.set_model(models.TransE)

    # Import
    # con.import_variables("./output/" + addr +"-model_TransE-" + stamp +".vec.tf")
    #con.import_variables("/home/toni/JointNRE-master/jointE/model/cnn5557.vec.tf")

    con.run()
    con.test()