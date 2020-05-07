from phe import paillier
from tools import *

#           DISTRIBUTED SECURE LINUCB ALGORITHM

# Useful self made functions are in tools

# Fix the scale of the ciphertexts in order not to leak information.
precision = pow(10, -26)

# The data client generates paillier keys, shares his public key and
# the budget with the cloud
class DataClient():
        def __init__(self, N, key_size):
                t = time.time()
                self.time = 0
                self.N = N
                self.pk, self.sk = paillier.generate_paillier_keypair(n_length=key_size)
                self.time += time.time() - t

        def send_budget(self):
                return self.N

        def share_pk_dc(self):
                return self.pk

        # At the end, receive the total reward and decrypt it using the private key
        def receive_sum(self, encrypted_s):
                t = time.time()
                self.s = self.sk.decrypt(encrypted_s)
                self.time += time.time() - t

# The data owner knows the value of the pull() parameter theta.
# Encrypts it with the public key of Comp before giving it to the cloud
class DataOwner():
        def __init__(self, pk_comp, theta):
                t = time.time()
                self.time = 0
                self.theta = theta
                self.pk_comp = pk_comp
                self.time += time.time() - t

        def outsource_theta(self):
                t = time.time()
                encrypted_theta = []
                for theta_i in self.theta:
                        encrypted_theta.append(self.pk_comp.encrypt(theta_i, precision))
                self.time += time.time() - t
                return encrypted_theta

# Node of the cloud who receives data from DO and DC, he pulls the arms
# and updates the variables at each step, until the budget is consumed
class Player():
        def __init__(self, pk_comp, delta, gamma, d, K, list_K):
                t = time.time()
                self.time = 0
                self.pk_comp = pk_comp
                self.delta = delta
                self.gamma = gamma
                self.R = 0.01
                self.d = d
                self.K = K
                self.list_K = list_K
                self.time += time.time() - t

        def receive_theta(self, theta):
                t = time.time()
                self.theta = np.array(theta)
                self.time += time.time() - t

        def receive_budget(self, N):
                t = time.time()
                self.N = N
                self.time += time.time() - t

        # Function called on all the upper bounds of the arms.
        # Permute the list and ask Comp for the index max
        def get_max(self, list_B):
                t = time.time()
                o = generate_permutation(self.K)
                list_permu = [0] * self.K
                for i in range(self.K):
                        list_permu[o[i]-1] = list_B[i]
                self.time += time.time() - t
                # Don't add to self the time of the comparator
                max_permu = self.comparator.max_bound(list_permu)
                t = time.time()
                max_B = o.index(max_permu + 1)
                self.time += time.time() - t
                return max_B

        # Initialization + exploration-exploitation phase
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
                        O = inv.dot(b)
                        for i in range(self.K):
                                exploration_term = 2 * self.R * math.sqrt(self.d *
                                        self.list_K[i].dot(inv).dot(self.list_K[i]) *
                                        math.log(t) * math.log ((t**2)/self.delta)) + math.log(t)
                                list_B[i] = self.list_K[i].dot(O) + exploration_term

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

        # Key switching
        # Add a random value to the encrypted sum, then
        # let the comparator decrypt it and encrypt it
        # again with the DataClient public key
        def get_re_encrypt(self):
                t = time.time()
                # Setting rand to a value too high will
                # erase s: after the key switching the result will
                # be 0. Rand is set to 10^10. If s gets too high then
                # in turn increase rand.
                rand = random.uniform(0, pow(10,10))
                randomized_s = self.s + rand
                self.time += time.time() - t
                randomized_re_encrypted_s = self.comparator.re_encrypt(randomized_s)
                t = time.time()
                re_encrypted_s = randomized_re_encrypted_s - rand
                self.time += time.time() - t
                return re_encrypted_s
                

# Theta is encrypted with his public key so in the end, also are all B values.
# At each round, Comp is sent a list of B in a permuted order. He decrypts them
# and returns the index of the maximal element
class Comp():
        def __init__(self, K, pk_dc, key_size):
                t = time.time()
                self.time = 0
                self.K = K
                self.pk_dc = pk_dc
                self.pk, self.sk = paillier.generate_paillier_keypair(n_length=key_size)
                self.time += time.time() - t

        def share_pk_comp(self):
                return self.pk

        # Decrypts and returns index max
        def max_bound(self, encrypted_list_B):
                t = time.time()
                decrypted_list_B = []
                for i in range(self.K):
                        decrypted_list_B.append(self.sk.decrypt(encrypted_list_B[i]))
                max_permu = decrypted_list_B.index(max(decrypted_list_B))
                self.time += time.time() - t
                return max_permu

        # Key switching
        def re_encrypt(self, encrypted_s):
                t = time.time()
                randomized_s = self.sk.decrypt(encrypted_s)
                re_encrypted_s = self.pk_dc.encrypt(randomized_s, precision)
                self.time += time.time() - t
                return re_encrypted_s
                

# Main function. N = the budget, delta = a constant for the exploration term.
# gamma = the regularizer for matrix inversion. K = the number of arms.
# theta = the common pull() parameter. list_K = the arms. d = the dimension
# of arms and theta. key_size = the length of Paillier keys. n = the number
# of cores for parallelization
def linucb_ds(N, delta, gamma, d, theta, K, list_K, key_size=2048, n=None): 
        t_start = time.time()

        DC = DataClient(N, key_size)
        pk_dc = DC.share_pk_dc()
        comparator = Comp(K, pk_dc, key_size)
        pk_comp = comparator.share_pk_comp()
        DO = DataOwner(pk_comp, theta)
        P = Player(pk_comp, delta, gamma, d, K, list_K)
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
        result["time DC"] = DC.time
        result["time DO"] = DO.time
        result["time P"] = P.time
        result["time comparator"] = comparator.time

        return result

if __name__ == "__main__":
        run_experiment(linucb_ds)
