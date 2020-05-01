#!/usr/bin/env python3
import os
import sys
import math
import json
sys.path.append(os.path.relpath("."))
import warnings
warnings.simplefilter("ignore")

# Compare the output (sum of rewards) of different algos
# run with several MovieLens users and movies (they are
# contained in extract_movie_lens folder: the Users and Movies
# .txt files)
# If the algos output the same sum, 'All equals' is printed

def parse_json_file(file_name):
        with open(file_name, 'r') as f:
                result = json.load(f)
        nb_runs = len(result)
        list_sum = list()
        for run in range(nb_runs):
            list_sum.append(result[str(run)]["sum"])
        return list_sum

nb_runs = 3
key_size = 512
algos = ["linucb", "linucb_ds"]
K_vals = [10,20,30,40]

DIR = "experiment_correctness/"
os.system("mkdir -p " + DIR)

# res contains a dict for each algo. In each, the keys are
# the values of K and the objects are lists of nb_runs results
# We compare the lists from one dict/algo to another
res = dict()
for algo in algos:
        res[str(algo)] = dict()
for K in K_vals:
        d = math.floor(K/10)
        N = K * 10
        for algo in algos:
                print ("*" * 10 + "K=", K, "algo=", algo)
                output_file = DIR + "K=" + str(K) + "_" + algo + ".txt"
                #os.system("python3 " + algo + ".py " + str(nb_runs) + " " + str(N)
                #          + " " + str(K) + " " + str(d) + " " + output_file + " "
                #          + str(key_size) + " " + str(1) + " " + str(0) + " " + str(0))
                res[str(algo)][str(K)] = parse_json_file(output_file)

        # Compare the results
        for i in range(len(algos)-1):
                assert res[algos[i]][str(K)] == res[algos[i+1]][str(K)]
print("All equals")
