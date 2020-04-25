from tools import *

#       Additional functions specific to SpectralUCB and secure versions

# The functions below generate the two input matrices of SpectralUCB.
# K is the number of nodes/arms = size of the matrices
def generate_W(K):
        W = np.zeros((K,K))
        for i in range(K):
                for j in range(K):
                        W[i][j] = random.randint(0,5)
        return W

def generate_D(W, K):
        D = np.zeros((K,K))
        for i in range(K):
                for k in range(K):
                        D[i][i] += W[i][k]
        return D

def generate_L(D, W):
        return np.add(D, -W)

def generate_eigen(L, K):
        values, vectors = np.linalg.eig(L)
        # Then sort the values, and the columns of the vectors in accordance
        tmp = [(values[i],i) for i in range(K)]
        tmp.sort()
        values_sorted, permutation = zip(*tmp)
        vectors_sorted = np.zeros((K,K))
        for i in range(K):
                vectors_sorted[:,i] = vectors[:,permutation[i]]
        return np.array(values_sorted), vectors_sorted

def eigendecomposition(eigen_values, eigen_vectors, K):
        Q = np.zeros((K,K))
        A = np.zeros((K,K))
        for i in range(K):
                A[i][i] = eigen_values[i]
                Q[:,i] = eigen_vectors[:,i]
        return A, Q

def generate_decomposition(L, K):
        eigen_values, eigen_vectors = generate_eigen(L, K)
        A, Q = eigendecomposition(eigen_values, eigen_vectors, K)
        return A, Q

# Use all above functions to return a diagonal matrix of eigenvalues
# A and a matrix of eigenvectors as columns Q
def generate_all(K):
        W = generate_W(K)
        D = generate_D(W, K)
        L = generate_L(D, W)
        return generate_decomposition(L, K)

def norm(v, M):
        return math.sqrt(v.dot(M).dot(v))


# Compute a portion of the list of upper bounds. This fucntion is called
# multiple times in parallel which allows to compute simultaneously different portions.
# Parallel functions for theta and decrypt the list of B are unchanged from linucb.

# 'quotient' and 'remainder' refer to the euclidian division of K the number of
# arms by n the number of cores used for parallelizing.
# In case the remainder is not zero, the first cores compute one more B
# until remainder is consumed.
def spectral_compute_B(i, Q, O, inv_V, E, quotient, remainder):
        res = []
        if remainder > 0:
                if (remainder - i) > 0:
                        for x in range(i * quotient + i, (i+1) * quotient + i + 1):
                                res.append(Q[x].dot(O) + E * norm(Q[x], inv_V))
                else:
                        for x in range(i * quotient + remainder, (i+1) * quotient + remainder):
                                res.append(Q[x].dot(O) + E * norm(Q[x], inv_V))
        else:
                for x in range(i * quotient, (i + 1) * quotient):
                        res.append(Q[x].dot(O) + E * norm(Q[x], inv_V))
        return res


# ------   Plot functions   -------

# Run the experiment defined in the function for a given algorithm
# and write the results of nb_runs executions in a given file
def spectral_run_experiment(algo):
        nb_runs = int(sys.argv[1])
        N =  int(sys.argv[2])
        K = int(sys.argv[3])
        d = int(sys.argv[4])
        output_file = sys.argv[5]
        key_size = int(sys.argv[6])
        n_cores = int(sys.argv[7])
        lamb = 0.01
        delta = 0.001
        B = 0.01
        C = math.log(N)
        A, Q = generate_all(K)
        theta = np.array(range(1, K+1))

        result = dict()
        for run in range(nb_runs):
                print ("run", run + 1)
                result[run] = algo(N, delta, lamb, theta, K, A, Q, B, C, key_size, n_cores)

        with open(output_file, 'w') as fp:
                json.dump(result, fp)                      
