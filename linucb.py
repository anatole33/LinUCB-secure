from tools import *

#               LINUCB ALGORITHM

# Useful self made functions are in tools

# The data client gives the budget N to the cloud, and receive
# s the sum of rewards with LinUCB
class DataClient():
        def __init__(self, N):
                self.time = 0
                t = time.time()
                self.N = N
                self.time += time.time() - t

        def send_budget(self):
                return self.N

        def receive_sum(self, s):
                t = time.time()
                self.s = s
                self.time += time.time() - t

# The data owner provide the pull() paramter theta to the cloud
class DataOwner():
        def __init__(self, theta):
                self.time = 0
                t = time.time()
                self.theta = theta
                self.time += time.time() - t

        def outsource_theta(self):
                return self.theta

# The cloud executes LinUCB algorithm with budget N,
# and send the output s to the client
class Cloud():
        def __init__(self, delta, gamma, d, K, list_K):
                self.time = 0
                t = time.time()
                self.delta = delta
                self.gamma = gamma
                self.d = d
                self.K = K
                self.list_K = list_K
                self.time += time.time() - t

        def receive_theta(self, theta):
                t = time.time()
                self.theta = theta
                self.time += time.time() - t

        def receive_budget(self, N):
                t = time.time()
                self.N = N
                self.time += time.time() - t
    
        def compute(self):
                ti = time.time()
                # List of the B of each arm. It's the upper bound term
                list_B = [0] * self.K
                # Total reward
                s = 0        
                # Randomly select an arm and initialize the variables
                a = random.randint(0, self.K - 1)
                x = self.list_K[a]
                r = pull(x, self.theta)
                s += r
                A = np.outer(x,x)
                b = r*x
                # Exploration / exploitation phase
                for t in range(1, self.N):
                        inv = np.linalg.inv(A + self.gamma * np.identity(self.d))
                        O = inv.dot(b)
                        for i in range(self.K):
                                exploration_term = math.sqrt(self.d * self.list_K[i].dot(inv).dot(
                                        self.list_K[i])* math.log(t) * math.log ((t**2)/self.delta))
                                list_B[i] = self.list_K[i].dot(O) + exploration_term
                        o = generate_permutation(self.K)
                        # Choose one arm among all equal maximums using the random permutation
                        max_B = argmax(list_B, o)
                        x = self.list_K[max_B]
                        r = pull(x, self.theta)
                        s += r
                        A += np.outer(x,x)
                        b += r*x
                self.time += time.time() - ti
                return s

# Main function. N = the budget, delta = a constant for the exploration term.
# gamma = the regularizer for matrix inversion. K = the number of arms.
# theta = the common pull() parameter. list_K = the arms. d = the dimension
# of arms and theta. key_size = the length of Paillier keys. n = the number
# of cores for parallelization
def linucb(N, delta, gamma, d, theta, K, list_K, key_size=None, n=None):
        t_start = time.time()

        DC = DataClient(N)
        DO = DataOwner(theta)
        C = Cloud(delta, gamma, d, K, list_K)

        C.receive_theta(DO.outsource_theta())
        C.receive_budget(DC.send_budget())
        DC.receive_sum(C.compute())

        t_stop = time.time()
        result = dict()
        result["sum"] = DC.s
        result["time"] = t_stop - t_start
        result["time DC"] = DC.time
        result["time DO"] = DO.time
        result["time C"] = C.time

        return result

if __name__ == "__main__":
        run_experiment(linucb)
