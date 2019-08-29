import config
import models
import tensorflow as tf
import numpy as np
import os
import sys
import time

if __name__ == "__main__":
    
    stamp = time.strftime("%H%M%S%j")
    addr = sys.argv[1]

    con = config.Config()
    con.set_in_path(addr + "/")

    print("Using DistMult.")
    #Model Configurations
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(False)
    con.set_work_threads(8)
    con.set_train_times(600)
    con.set_nbatches(100)
    con.set_alpha(0.1)
    con.set_lmbda(0.0001)
    con.set_bern(0)
    con.set_dimension(100)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adagrad")

    # Export position
    con.set_export_files("./output/CS1--model_DistMult-" + stamp + ".vec.tf", 0)
    con.set_out_files("./res/CS1--embedding_DistMult-" + stamp + ".vec.json")

    con.init()
    con.set_model(models.DistMult)

    # Import
    # con.import_variables("./output/" + addr +"-model_DistMult-" + stamp +".vec.tf")

    con.run()
    con.test()