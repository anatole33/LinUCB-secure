#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.relpath("."))
from tools import parse_json_output, plot_lines
import warnings
warnings.simplefilter("ignore")

#  Script of Figure 7

nb_runs = 1
n_cores = 1
key_size = 2048
algos = ["spectralucb_ds", "linucb_ds"]
algos_names = ["SpectralUCB-DS", "LinUCB-DS"]
K_vals = [10,15,20,25,30]

DIR = "experiment_spectral/"
os.system("mkdir -p " + DIR)

aggregates_time = dict()
for algo in algos:
        aggregates_time[algo] = list()
for K in K_vals:
        d = K
        N = K
        for algo in algos:
                print ("*" * 10 + "K=", K, "algo=", algo)
                output_file = DIR + "N_K_d=" + str(K) + "_" + algo + ".txt"
                #os.system("python3 " + algo + ".py " + str(nb_runs) + " " + str(N)
                #          + " " + str(K) + " " + str(d) + " " + output_file + " "
                #          + str(key_size) + " " + str(n_cores))
                aggregate_time, _ = parse_json_output(output_file)
                aggregates_time[algo].append(aggregate_time)

# generate plot
plot_lines(str(key_size), algos_names, "", "N = K = d", K_vals, aggregates_time, False, False, DIR)
