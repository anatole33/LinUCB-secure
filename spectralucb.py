from linucb import *
from spectralucb_tools import *

#               SPECTRALUCB ALGORITHM

# Import DataClient and DataOWner from linucb

class Spectral_Cloud(Cloud):
        # delta, lamb and B are constant parameters, used for the exploration term
        # K is the number of arms/nodes, A is the diagonal matrix of eigen values and
        # Q is the matrix of eigen vectors as columns
        def __init__(self, delta, lamb, K, A, Q, B):
                self.time = 0
                t = time.time()
                self.delta = delta
                self.lamb = lamb
                self.K = K
                self.A = A
                self.Q = Q
                self.B = B
                self.time += time.time() - t
    
        def spectral_compute(self):
                ti = time.time()
                # list of the B of each arm. It's the upper bound term
                list_B = [0] * self.K

                T = np.add(self.A, self.lamb * np.identity(self.K))
                d = 0
                for i in range(self.K):
                        if (i-1) * self.A[i][i] <= self.N / math.log(1 + (self.N/self.lamb)):
                                d = i

                # Pull an arm at random and start updating the sum of rewards
                x = random.randint(0, self.K-1)
                s = pull(self.Q[x], self.theta)
                # Initialize list of rewards
                list_r = [s]
                # Initialize list of rows of Q
                list_x = [self.Q[x]]
                # Exploration / exploitation phase
                for t in range(1, self.N):
                        X = np.transpose(np.array(list_x))
                        R = np.transpose(np.array(list_r))
                        V = np.add(X.dot(np.transpose(X)), T)
                        inv_V = np.linalg.inv(V)
                        O = inv_V.dot(X).dot(R)

                        # Constant term of the exploration term
                        E = 2 * self.B * math.sqrt(d * math.log(1 + t/self.lamb) +
                                2 * math.log(1/self.delta)) + math.log(t)
                        for i in range(self.K):
                                list_B[i] = O.dot(self.Q[i]) + E * norm(self.Q[i], inv_V)
                        # Randomly choose one of the best arms if their are many equals 
                        o = generate_permutation(self.K)
                        max_B = argmax(list_B, o)
                        x = self.Q[max_B]
                        r = pull(x, self.theta)
                        s += r
                        list_r.append(r)
                        list_x.append(x)
                self.time += time.time() - ti
                return s

# Main function.
# delta, lamb, B and C are constant parameters, used for the exploration term
# K is the number of arms/nodes, A is the diagonal matrix of eigen values and
# Q is the matrix of eigen vectors as columns
# key_size is the length of Paillier keys
# n is the number of cores for parallelization
def spectral_ucb(N, delta, lamb, theta, K, A, Q, B, key_size=None, n=None):
        t_start = time.time()

        DC = DataClient(N)
        DO = DataOwner(theta)
        C = Spectral_Cloud(delta, lamb, K, A, Q, B)

        C.receive_theta(DO.outsource_theta())
        C.receive_budget(DC.send_budget())
        DC.receive_sum(C.spectral_compute())

        t_stop = time.time()
        result = dict()
        # Round the imprecision of float
        result["sum"] = float(f"{DC.s:.{5}f}")
        result["time"] = t_stop - t_start
        result["time DC"] = DC.time
        result["time DO"] = DO.time
        result["time C"] = C.time

        return result

if __name__ == "__main__":
        spectral_run_experiment(spectral_ucb)
