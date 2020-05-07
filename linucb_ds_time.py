from linucb_ds import *

#           DISTRIBUTED SECURE LINUCB ALGORITHM
#       Measures the computation time of the 3 costliest operations

# Import DataClient DataOwner and Comp from linucb_ds

class Player_t(Player):
        def __init__(self, pk_comp, delta, gamma, d, K, list_K):
                t = time.time()
                self.time_theta = 0
                self.time_Bi = 0
                self.time_dec = 0
                self.time = 0
                self.pk_comp = pk_comp
                self.delta = delta
                self.gamma = gamma
                self.R = 0.01
                self.d = d
                self.K = K
                self.list_K = list_K
                self.time += time.time() - t

        # Function called on all the upper bounds of the arms.
        # Permute the list and ask Comp for the index max
        #Time precisely the decryptions performed by Comp
        def get_max(self, list_B):
                t = time.time()
                o = generate_permutation(self.K)
                list_permu = [0] * self.K
                for i in range(self.K):
                        list_permu[o[i]-1] = list_B[i]
                self.time += time.time() - t
                # Don't add to self the time of the comparator
                
                # Time of the decryption and comparison
                t1 = time.time()
                max_permu = self.comparator.max_bound(list_permu)
                self.time_dec += time.time() - t1
                
                t = time.time()
                max_B = o.index(max_permu + 1)
                self.time += time.time() - t
                return max_B

        def compute(self):
                ti = time.time()
                # List of the B of each arm. It's the upper bound term
                list_B = [0] * self.K
                b = np.array([self.pk_comp.encrypt(0)] * self.d)
                # Randomly select an arm and initialize the variables
                a = random.randint(0, self.K - 1)
                x = self.list_K[a]
                r = pull(x, self.theta)
                s = r
                A = np.outer(x, x)
                update_b(b, r, x)
                self.time += time.time() - ti
                # Exploraton / exploitation phase
                for t in range(1, self.N):
                        ti = time.time()
                        inv = np.linalg.inv(A + self.gamma * np.identity(self.d))
                        
                        # Time the computation of theta
                        t1 = time.time()
                        O = inv.dot(b)
                        self.time_theta += time.time() - t1
                        
                        # Time the computation of the Bi
                        t2 = time.time()
                        #exploration_term = math.sqrt(2*self.d*(1+2*math.log(t**2/self.delta)))
                        for i in range(self.K):
                                exploration_term = 2 * self.R * math.sqrt(self.d *
                                        self.list_K[i].dot(inv).dot(self.list_K[i]) *
                                        math.log(t) * math.log ((t**2)/self.delta)) + math.log(t)
                                list_B[i] = self.list_K[i].dot(O) + exploration_term
                        self.time_Bi += time.time() - t2

                        # Don't add to self the time of decryption of Comp
                        self.time += time.time() - ti
                        max_B = self.get_max(list_B)
                        ti = time.time()
                        x = self.list_K[max_B]
                        r = pull(x, self.theta)
                        s += r
                        A += np.outer(x,x)
                        update_b(b, r, x)
                        self.time += time.time() - ti
                self.s = s
                
# Main function. N = the budget, delta = a constant for the exploration term.
# gamma = the regularizer for matrix inversion. K = the number of arms.
# theta = the common pull() parameter. list_K = the arms. d = the dimension
# of arms and theta. key_size = the length of Paillier keys. n = the number
# of cores for parallelization
def linucb_ds_t(N, delta, gamma, d, theta, K, list_K, key_size=2048, n=None):
        t_start = time.time()

        DC = DataClient(N, key_size)
        pk_dc = DC.share_pk_dc()
        comparator = Comp(K, pk_dc, key_size)
        pk_comp = comparator.share_pk_comp()
        DO = DataOwner(pk_comp, theta)
        P = Player_t(pk_comp, delta, gamma, d, K, list_K)
        P.comparator = comparator

        P.receive_theta(DO.outsource_theta())
        P.receive_budget(DC.send_budget())
        P.compute()
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
        run_experiment(linucb_ds_t)
