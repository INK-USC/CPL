import config
import models
import tensorflow as tf
import numpy as np
import os
import sys
import time

if __name__ == "__main__":

    addr = sys.argv[1]

    stamp = time.strftime("%H%M%S%j")

    con = config.Config()
    con.set_in_path(addr + "/")

    print("Using RESCAL.")
    #Model Configurations
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(False)
    con.set_work_threads(4)
    con.set_train_times(500)
    con.set_nbatches(100)
    con.set_alpha(0.1)
    con.set_bern(0)
    con.set_margin(1)
    con.set_dimension(100)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adagrad")

    # Export position
    con.set_export_files("./output/CS1--model_RESCAL-" + stamp + ".vec.tf", 0)
    con.set_out_files("./res/CS1--embedding_RESCAL-" + stamp + ".vec.json")

    con.init()
    con.set_model(models.RESCAL)

    # Import
    # con.import_variables("./output/" + addr +"-model_TransE-" + stamp +".vec.tf")
    #con.import_variables("/home/toni/JointNRE-master/jointE/model/cnn5557.vec.tf")

    con.run()
    con.test()