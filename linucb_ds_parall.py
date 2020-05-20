from linucb_ds import *

#           DISTRIBUTED SECURE LINUCB ALGORITHM
#          Parallelize the 3 costliest operations

# Import DataClient DataOwner, and unchanged functions for linucb_ds

# n is the number of cores used in parallelization
class Player_p(Player):
        def __init__(self, pk_comp, delta, gamma, d, K, list_K, n):
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
                self.n = n
                self.time += time.time() - t

        # Function called on all the upper bounds of the arms.
        # Permute the list and ask Comp for the index max, using
        # parallelizer p
        def get_max(self, list_B, p):
                t = time.time()
                o = generate_permutation(self.K)
                list_permu = [0] * self.K
                for i in range(self.K):
                        list_permu[o[i]-1] = list_B[i]
                self.time += time.time() - t
                # Don't add to self the time of the comparator
                
                # Time of the decryption and comparison
                t1 = time.time()
                max_permu = self.comparator.max_bound(list_permu, p)
                self.time_dec += time.time() - t1
                
                t = time.time()
                max_B = o.index(max_permu + 1)
                self.time += time.time() - t
                return max_B

        # Initialization + exploration-exploitation phase
        def compute(self):
                ti = time.time()
                # Compute the maximum norm among the arms
                norm_max = 0
                for arm in self.list_K:
                        norm = arm.dot(arm)
                        if norm > norm_max:
                                norm_max = norm
                b = np.array([self.pk_comp.encrypt(0)] * self.d)
                # Randomly select an arm and initialize the variables
                a = random.randint(0, self.K - 1)
                x = self.list_K[a]
                r = pull(x, self.theta)
                s = r
                A = np.outer(x, x)
                update_b(b, r, x)
                # Create the object for parallelization
                p = mp.Pool(self.n)
                quotient_d = self.d // self.n
                remainder_d = self.d % self.n
                quotient_K = self.K // self.n
                remainder_K = self.K % self.n
                self.time += time.time() - ti
                # Exploraton / exploitation phase
                for t in range(1, self.N):
                        ti = time.time()
                        inv = np.linalg.inv(A + self.gamma * np.identity(self.d))
                        
                        # Time the computation of theta
                        t1 = time.time()
                        res = p.starmap(compute_theta, [(i, inv, b, quotient_d,
                                remainder_d) for i in range(self.n)])
                        O = np.array([])
                        for i in range(self.n):
                                O = np.append(O, res[i])
                        self.time_theta += time.time() - t1
                        
                        # Time the computation of the Bi
                        t2 = time.time()
                        exploration_term = self.R * math.sqrt(self.d * math.log((1 + (t *
                                        norm_max)/self.gamma)/self.delta)) + math.sqrt(
                                        self.gamma) * math.log(t)
                        list_B = []
                        res2 = p.starmap(compute_B, [(i, self.list_K, O, exploration_term,
                                        quotient_K, remainder_K) for i in range(self.n)])
                        for i in range(self.n):
                                list_B += res2[i]
                        self.time_Bi += time.time() - t2

                        # Don't add to self the time of decryption of Comp
                        self.time += time.time() - ti
                        max_B = self.get_max(list_B, p)
                        ti = time.time()
                        x = self.list_K[max_B]
                        r = pull(x, self.theta)
                        s += r
                        A += np.outer(x,x)
                        update_b(b, r, x)
                        self.time += time.time() - ti
                p.close()
                self.s = s
                

# Theta is encrypted with his public key so in the end, also are all B values.
# At each round, Comp is sent a list of B in a permuted order. He decrypts them
# and returns the index of the maximal element
# n is the number of cores used in parallelization
class Comp_p(Comp):
        def __init__(self, K, pk_dc, key_size, n):
                t = time.time()
                self.time = 0
                self.K = K
                self.pk_dc = pk_dc
                self.n = n
                self.pk, self.sk = paillier.generate_paillier_keypair(n_length=key_size)
                self.time += time.time() - t

        # Decrypt the list of B using the parallelizer p
        # and return the index max
        def max_bound(self, encrypted_list_B, p):
                t = time.time()
                quotient_K = self.K // self.n
                remainder_K = self.K % self.n
                decrypted_list_B = []
                res = p.starmap(decrypt_B, [(i, encrypted_list_B,
                        self.sk,quotient_K, remainder_K) for i in range(self.n)])
                for i in range(self.n):
                                decrypted_list_B += res[i]
                max_permu = decrypted_list_B.index(max(decrypted_list_B))
                self.time += time.time() - t
                return max_permu


# Main function. N = the budget, delta = a constant for the exploration term.
# gamma = the regularizer for matrix inversion. K = the number of arms.
# theta = the common pull() parameter. list_K = the arms. d = the dimension
# of arms and theta. key_size = the length of Paillier keys. n = the number
# of cores for parallelization
def linucb_ds_p(N, delta, gamma, d, theta, K, list_K, key_size=2048, n_cores=1): 
        t_start = time.time()

        DC = DataClient(N, key_size)
        pk_dc = DC.share_pk_dc()
        comparator = Comp_p(K, pk_dc, key_size, n_cores)
        pk_comp = comparator.share_pk_comp()
        DO = DataOwner(pk_comp, theta)
        P = Player_p(pk_comp, delta, gamma, d, K, list_K, n_cores)
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
        run_experiment(linucb_ds_p)
