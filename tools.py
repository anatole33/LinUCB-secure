import numpy as np
import sys
import json
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing as mp
rng = np.random.RandomState(1)
random.seed(1)

# Return a reward for an arm x and the secret theta
def pull(x, theta):
        noise = [-1,1]
        index = int(random.uniform(0, 1) < 1/2)
        r = x.dot(theta)
        return r + noise[index]

# Return a list of a random permutation over [1,K]
def generate_permutation(K):
        sigma_dict = {}
        for element in range(1, K+1):
                sigma_dict[element] = rng.randint(0, sys.maxsize)
                # On Windows, sys.maxsize in out of bounds. Change it to pow(2,31)
        vals_sorted = sorted(sigma_dict.values())
        sigma = []
        for element in sigma_dict.keys():
                sigma.append(vals_sorted.index(sigma_dict[element]) + 1)
        return sigma

# Update b with an encrypted reward and an arm x
def update_b(b, r, x):
        for i in range(len(x)):
                b[i] += r * float(x[i])

# Use sigma, a permutation of range(1, K+1) to define the maximum element
# of a list in case there is an equality. If sigma is randomly chosen,
# then the returned element is also random (among all maximums)
def argmax(list_B, sigma):
        S_m = list_B[0]
        i_m = 0
        for i in range(1, len(list_B)):
                if list_B[i] > S_m or (list_B[i] == S_m and sigma[i] < sigma[i_m]):
                        S_m = list_B[i]
                        i_m = i
        return i_m


#  -------  Parallelization functions  -------

# Each of the three following functions are called multiple times in parallel
# when we want to compute O, or the list of Bi or decrypt the list of Bi.
# In each call, only a portion of the whole calculation is done. That portion
# is defined by the number of cores used for parallelizing.

# 'quotient' and 'remainder' refer to the euclidian division of d the size
# of the vectors by n the number of cores.
# In case the remainder is not zero, the first cores compute one more coordinate
# of O until remainder is consumed.
def compute_theta(i, inv, b, quotient, remainder):
        res = []
        if remainder > 0:
                if (remainder - i) > 0:
                        for x in range(i * quotient + i, (i+1) * quotient + i + 1):
                                res.append(inv[x].dot(b))
                else:
                        for x in range(i * quotient + remainder, (i+1) * quotient + remainder):
                                res.append(inv[x].dot(b))
        else:
                for x in range(i * quotient, (i+1) * quotient):
                        res.append(inv[x].dot(b))
        return res

# 'quotient' and 'remainder' refer to the euclidian division of K the number of
# arms by n the number of cores.
# In case the remainder is not zero, the first cores compute one more B
# until remainder is consumed.
def compute_B(i, list_K, O, inv, t, d, delta, quotient, remainder):
        res = []
        if remainder > 0:
                if (remainder - i) > 0:
                        for x in range(i * quotient + i, (i+1) * quotient + i + 1):
                                exploration_term = math.sqrt(d * list_K[x].dot(inv).dot(list_K[x])
                                        * math.log(t) * math.log((t**2)/delta))
                                res.append(list_K[x].dot(O) + exploration_term)
                else:
                        for x in range(i * quotient + remainder, (i+1) * quotient + remainder):
                                exploration_term = math.sqrt(d * list_K[x].dot(inv).dot(list_K[x])
                                        * math.log(t) * math.log((t**2)/delta))
                                res.append(list_K[x].dot(O) + exploration_term)
        else:
                for x in range(i * quotient, (i + 1) * quotient):
                        exploration_term = math.sqrt(d * list_K[x].dot(inv).dot(list_K[x])
                                        * math.log(t) * math.log((t**2)/delta))                        
                        res.append(list_K[x].dot(O) + exploration_term)
        return res
        
# 'quotient' and 'remainder' refer to the euclidian division of K the number of
# arms by n the number of cores.
# In case the remainder is not zero, the first cores decrypt one more B
# until remainder is consumed.
def decrypt_B(i, list_B, sk, quotient, remainder):
        dec = []
        if remainder > 0:
                if (remainder - i) > 0:
                        for x in range(i * quotient + i, (i+1) * quotient + i + 1):
                                dec.append(sk.decrypt(list_B[x]))
                else:
                        for x in range(i * quotient + remainder, (i+1) * quotient + remainder):
                                dec.append(sk.decrypt(list_B[x]))
        else:
                for x in range(i * quotient, (i + 1) * quotient):
                        dec.append(sk.decrypt(list_B[x]))
        return dec


#  -------  Plot functions  --------

# Run the experiment defined in the function for a given algorithm
# and write the results of nb_runs executions in a given file
def run_experiment(algo):
        nb_runs = int(sys.argv[1])
        N =  int(sys.argv[2])
        K = int(sys.argv[3])
        d = int(sys.argv[4])
        output_file = sys.argv[5]
        key_size = int(sys.argv[6])
        n_cores = int(sys.argv[7])
        gamma = 1
        delta = 1/N
        # For real data experiment using MovieLens, we use users and movies from a file.
        # d must be equal to the number of features in Users and Movies matrices
        if len(sys.argv) > 8:
                user_index = int(sys.argv[8])
                theta = get_data_from_file(user_index, user_index + 1,
                        "extract_movie_lens/Users" + str(d) + ".txt")
        else:
                theta = np.array(range(1, d+1))
        if len(sys.argv) > 9:
                arms_index = int(sys.argv[9])
                list_K = get_data_from_file(arms_index, arms_index + K,
                        "extract_movie_lens/Movies" + str(d) + ".txt")
        else:
                l = []
                if K > 0:
                        l.append([1.2]*d)
                for i in range(K-1):
                        l.append([1.1]*d)
                list_K = np.array(l)

        # Fill 'result' with nb_runs executions
        result = dict()
        for run in range(nb_runs):
                print ("run", run + 1)
                result[run] = algo(N, delta, gamma, d, theta, K, list_K, key_size, n_cores)

        # Write result in the given file
        with open(output_file, 'w') as fp:
                json.dump(result, fp)

# Read the file containing a matrix embedding the users or the movies,
# and return a np.array of the users (or movies) features from
# position k to l in the matrix
def get_data_from_file(k, l, file_name):
        res = []
        with open(file_name, 'r') as f:
                # Pass the first k elements
                for _ in range(k):
                        tmp = f.readline()
                for _ in range(k, l):
                        line = f.readline().split()
                        tmp = np.array([float(line[i]) for i in range(len(line))])
                        res.append(tmp)
        if len(res) == 1:
                return res[0]
        else:
                return np.array(res)                       

# Take as input the file containing 'nb_runs' results of an algorithm.
# Returns aggregate_time, the mean computation time, and aggregates,
# a dictionary of mean time of every participant
def parse_json_output(file_name):
        with open(file_name, 'r') as f:
                result = json.load(f)
        nb_runs = len(result)
        aggregate_time = 0
        aggregates = dict()
        for run in range(nb_runs):
                run = str(run)
                aggregate_time += result[run]["time"]
                if run == '0':  
                        for key, values in result[run].items():
                                if key != "sum" and key != "time":
                                        aggregates[key] = result[run][key]
                else:
                        for key, values in result[run].items():
                                if key != "sum" and key != "time":
                                        aggregates[key] += result[run][key]

        aggregate_time /= nb_runs
        for key in aggregates.keys():
                aggregates[key] /= nb_runs
        return (aggregate_time, aggregates)

# Used in script_n_cores and script_compare_spectral
def plot_lines(key_size, legend_vals, legend_title, xlabel, x_vals, data, xlog, ylog, DIR):
        plt.figure(figsize=(6, 5))
        plt.rcParams.update({'font.size':14})

        markers = ['.', 'v', '*', 'd', 'x', 'o']

        for key in data:
                plt.plot(x_vals, data[key], marker=markers.pop())
                
        plt.legend(legend_vals, loc = 'upper left', title = legend_title)

        if ylog:
                plt.yscale('log')
        if xlog:
                plt.xscale('log')
                
        plt.xlabel(xlabel)
        plt.xticks(x_vals)
        
        plt.ylabel('Time (seconds)')
        
        plt.savefig(DIR + "plot_compare_" + key_size + ".pdf")
        plt.clf()

# Used in script_stacked
def plot_stack_lines(scenario, xlabel, x_vals, data, DIR):
        plt.figure(figsize=(6,5))
        plt.rcParams.update({'font.size':16})
        fig, ax = plt.subplots()

        colors = ["beige", "pink", "lightblue"]
        hatches=["o", "x", "+"]
        labels = ["Step(3).i", "Step (3).ii", "Step (4).i"]

        # Plot stacked lines
        stacks = ax.stackplot(x_vals, data["theta"], data["Bi"], data["dec"], colors=colors, edgecolor=['black', 'black', 'black'])

        # Add hatch on the figure
        for stack, hatch in zip(stacks, hatches):
                stack.set_hatch(hatch)

        # Custom legend
        leg1 = mpatches.Patch(facecolor=colors[2], alpha=1, hatch=hatches[2],label=labels[2])
        leg2 = mpatches.Patch(facecolor=colors[1], alpha=1, hatch=hatches[1],label=labels[1])
        leg3 = mpatches.Patch(facecolor=colors[0], alpha=1, hatch=hatches[0],label=labels[0])
        ax.legend(handles=[leg1, leg2, leg3])
        
        plt.xlabel(xlabel)
        plt.xticks(x_vals)
        plt.ylabel('Time (seconds)')

        plt.savefig(DIR + "plot_" + scenario + ".pdf")
        plt.clf()

