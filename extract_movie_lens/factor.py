#!/usr/bin/env python3
from extract_from_data import *
import json
import sys

d = int(sys.argv[1])

# Build the CF model with default parameters
model = build_model(ratings, embedding_dim=d, init_stddev=1)
# Train the model to gain in precision
model.train(num_iterations=300, learning_rate=10.0)

# Select the matrices, their type is tensor
tensor_U = model._embedding_vars['user_id']
tensor_M = model._embedding_vars['movie_id']

# Convert tensor into numpy ndarray matrices
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    U = tensor_U.eval()
    M = tensor_M.eval()

# Fill a file with the matrices
file_U = "Users" + str(d) + ".txt"
file_M = "Movies" + str(d) + ".txt"
with open(file_U, 'w') as fu:
    for row in U:
        line = ""
        for i in range(len(row)):
            line += str(row[i])
            line += " "
        line += '\n'
        fu.write(line)

with open(file_M, 'w') as fm:
    for row in M:
        line = ""
        for i in range(len(row)):
            line += str(row[i])
            line += " "
        line += '\n'
        fm.write(line)

