from linucb_ds_parall import *
from spectralucb_tools import *

#           DISTRIBUTED SECURE SPECTRALUCB ALGORITHM
#            Parallelize the 3 costliest operations

# Import DataClient, DataOwner and Comparator from linucb_ds_parall

class Spectral_Player(Player_p):
        def __init__(self, pk_comp, delta, lamb, K, A, Q, B, n):
                self.time = 0
                t = time.time()
                self.time_theta = 0
                self.time_Bi = 0
                self.time_dec = 0
                self.pk_comp = pk_comp
                self.delta = delta
                self.lamb = lamb
                self.K = K
                self.A = A
                self.Q = Q
                self.B = B
                self.n = n
                self.time += time.time() - t
                
        def spectral_compute(self):
                ti = time.time()
                # List of the B of each arm. It's the upper bound term
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
                # Create the object for parallelization
                quotient_K = self.K // self.n
                remainder_K = self.K % self.n
                p = mp.Pool(self.n)
                self.time += time.time() - ti
                # Exploration / exploitation phase
                for t in range(1, self.N):
                        ti = time.time()
                        X = np.transpose(np.array(list_x))
                        R = np.transpose(np.array(list_r))
                        V = np.add(X.dot(np.transpose(X)), T)
                        inv_V = np.linalg.inv(V)
                        inv_V_X = inv_V.dot(X)

                        # Time the computation of theta
                        t1 = time.time()
                        res = p.starmap(compute_theta, [(i, inv_V_X, R, quotient_K,
                                remainder_K) for i in range(self.n)])
                        O = np.array([])
                        for i in range(self.n):
                                O = np.append(O, res[i])
                        self.time_theta += time.time() - t1
                        # Constant term of the exploration term
                        E = 2 * self.B * math.sqrt(d * math.log(1 + t/self.lamb) +
                                2 * math.log(1/self.delta)) + math.log(t)
                        # Time the computation of the Bi
                        t2 = time.time()
                        list_B = []
                        res2 = p.starmap(spectral_compute_B, [(i, self.Q, O, inv_V, E,
                                quotient_K, remainder_K) for i in range(self.n)])
                        for i in range(self.n):
                                list_B += res2[i]
                        self.time_Bi += time.time() - t2
                        # Don't add to self the time of Decryption of Comp
                        self.time += time.time() - ti
                        max_B = self.get_max(list_B, p)
                        ti = time.time()
                        x = self.Q[max_B]
                        r = pull(x, self.theta)
                        s += r
                        list_r.append(r)
                        list_x.append(x)
                        self.time += time.time() - ti
                p.close()
                self.s = s                

# Main function.
# delta, lamb, B and C are constant parameters, used for the exploration term
# K is the number of arms/nodes, A is the diagonal matrix of eigen values and
# Q is the matrix of eigen vectors as columns
# key_size is the length of Paillier keys
# n is the number of cores for parallelization
def spectralucb_ds_p(N, delta, lamb, theta, K, A, Q, B, key_size=2048, n=1):
        t_start = time.time()

        DC = DataClient(N, key_size)
        pk_dc = DC.share_pk_dc()
        comparator = Comp_p(K, pk_dc, key_size, n)
        pk_comp = comparator.share_pk_comp()
        DO = DataOwner(pk_comp, theta)
        P = Spectral_Player(pk_comp, delta, lamb, K, A, Q, B, n)
        P.comparator = comparator

        P.receive_theta(DO.outsource_theta())
        P.receive_budget(DC.send_budget())
        P.spectral_compute()
        DC.receive_sum(P.get_re_encrypt())

        t_stop = time.time()
        result = dict()
        # Round the imprecision of float
        result["sum"] = float(f"{DC.s:.{5}f}")
        result["time"] = t_stop - t_start
        result["time of theta"] = P.time_theta
        result["time of Bi"] = P.time_Bi
        result["time of dec"] = P.time_dec
        result["time DC"] = DC.time
        result["time DO"] = DO.time
        result["time P"] = P.time
        result["time comparator"] = comparator.time

        return result

if __name__ == "__main__":
        spectral_run_experiment(spectralucb_ds_p)
