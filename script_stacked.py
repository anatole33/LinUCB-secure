#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.relpath("."))
from tools import parse_json_output, plot_stack_lines
import warnings
warnings.simplefilter("ignore")

#  Script of Figures 6(a) 6(b)

nb_runs = 1
key_size = 2048
n_cores_vals = [1, 2, 3, 4, 5, 6]
algo = "linucb_ds_parall"
K_vals = [50, 6]
d_vals = [5, 18]
N = 200

DIR_exp = "experiment_n_cores/"
os.system("mkdir -p " + DIR_exp)

keys = ["theta", "Bi", "dec"]

for i in range(len(K_vals)):
        DIR = DIR_exp + "linucb_" + str(K_vals[i]) + "_" + str(d_vals[i]) + "/"
        os.system("mkdir -p " + DIR)
        aggregates_all = dict()
        for key in keys:
                aggregates_all[key] = list()
        for n_cores in n_cores_vals:
                print ("*" * 10 + "n =", n_cores, "K =", K_vals[i], "d =", d_vals[i])
                output_file = DIR + "n=" + str(n_cores) + "_d=" + str(d_vals[i]) + "_K=" + str(K_vals[i]) + "_N=" + str(N) + "_" + algo + ".txt"
                #os.system("python3 " + algo + ".py " + str(nb_runs) + " " + str(N) + " "
                #          + str(K_vals[i]) + " " +str(d_vals[i]) + " " + output_file + " "
                #          + str(key_size) + " " + str(n_cores) + " " + str(0) + " " + str(0))
                _, res = parse_json_output(output_file)
                for key in keys:
                        aggregates_all[key].append(res["time of " + str(key)])
                
        # generate plot
        plot_stack_lines("stack_lines_" + str(K_vals[i]) + "_" + str(d_vals[i]), "Number of cores", n_cores_vals, aggregates_all, DIR)

# Same figures for spectralucb_ds_parall
# Here d=K=N
algo = "spectralucb_ds_parall"
K_vals = [6, 50]
for K in K_vals:
        DIR = DIR_exp + "spectral_" + str(K) + "/"
        os.system("mkdir -p " + DIR)
        aggregates_all = dict()
        for key in keys:
                aggregates_all[key] = list()
        N = K
        d = K
        for n_cores in n_cores_vals:
                print ("*" * 10 + "n =", n_cores, "K =", K)
                output_file = DIR + "n=" + str(n_cores) + "_K=" + str(K) + "_" + algo + ".txt"
                #os.system("python3 " + algo + ".py " + str(nb_runs) + " " + str(N) + " "
                #          + str(K) + " " +str(d) + " " + output_file + " "
                #          + str(key_size) + " " + str(n_cores)  + " " + str(0) + " " + str(0))
                _, res = parse_json_output(output_file)
                for key in keys:
                        aggregates_all[key].append(res["time of " + str(key)])
                
        # generate plot
        plot_stack_lines("stack_lines_" + str(K), "Number of cores", n_cores_vals, aggregates_all, DIR)
