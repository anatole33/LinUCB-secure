#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.relpath("."))
from tools import *
import warnings
warnings.simplefilter("ignore")

#  Script of Figure 6(c)

nb_runs = 1
key_size_vals = [2048]
n_cores_vals = [1,2,3,4,5,6]
algo_name = "linucb_ds_parall"
K_vals = [[10,30,50,70]]
N = 200

for k in range(len(key_size_vals)):
        DIR = "experiment_n_cores/linucb_K_varies/"
        os.system("mkdir -p " + DIR)
        aggregates_time = dict()
        # If number of core is 1, don't use parallelized version
        for n_cores in n_cores_vals:
                if n_cores != 1:
                        algo = algo_name
                else:
                        algo = "linucb_ds_time"
                aggregates_time[str(n_cores)] = list()
                for i in range(len(K_vals[k])):
                        K = K_vals[k][i]
                        d = math.ceil(K/10)
                        print ("*" * 10 + "key size =", key_size_vals[k], "n =", n_cores, "K =", K)
                        output_file = DIR + "n=" + str(n_cores) + "_d=" + str(d) + "_K=" + str(K) + "_N=" + str(N) + "_" + algo + ".txt"
                        #os.system("python3 " + algo + ".py " + str(nb_runs) + " " + str(N)
                        #          + " " + str(K) + " " + str(d) + " " + output_file + " "
                        #          + str(key_size_vals[k]) + " " + str(n_cores) + " " + str(0) + " " + str(0))
                        aggregate_time, _ = parse_json_output(output_file)
                        aggregates_time[str(n_cores)].append(aggregate_time)
        
        # generate plot
        plot_lines(str(key_size_vals[k]), [str(n_cores_vals[i]) for i in range(len(n_cores_vals))], "Number of cores", "Number of arms", K_vals[k], aggregates_time, False, False, DIR)
