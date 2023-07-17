import numpy as np

class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def forward_log(self, O: list):
        """
        :param O: is the sequence (an array of) discrete (integer) observations, i.e. [0, 2, 1 ,3, 4]
        :return: ln P(O|λ) score for the given observation, ln: natural logarithm
        """

        # T is the length of the observation sequence which we will use for the size of the alpha array
        T = len(O)

        # N is the number of states in the model
        N = self.A.shape[0]

        # 2D Array of T by N Zeros, will be used to store forward variable
        alpha = np.zeros((T, N))

        # Initializing first column of alpha using initial state probabilities and first observation
        alpha[0, :] = np.log(self.Pi) + np.log(self.B[:, O[0]])

        # Updating alpha using probability of being in each state and probability of transitioning from
        # prev state to current one
        # Also we are using log to stabilize the data and avoid cases such as underflows
        for t in range(1, T):
            alpha[t, :] = np.log(self.B[:, O[t]]) + np.log(np.dot(np.exp(alpha[t - 1, :]), self.A))


        # In the end we return final probability of the sequence using the probabilities of being in each state
        return np.log(np.sum(np.exp(alpha[-1, :])))



    def viterbi_log(self, O: list):
        """
        :param O: is an array of discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: the tuple (Q*, ln P(Q*|O,λ)), Q* is the most probable state sequence for the given O
        """

        # T and N same as before
        T = len(O)
        N = self.A.shape[0]

        # 2D Arrag of Zeros to store delta which is the likelihood of being in a state
        delta = np.zeros((T, N))

        # Array used to store the most likely state at each time step
        psi = np.zeros((T, N))

        # Initializing first column of delta
        delta[0, :] = np.log(self.Pi) + np.log(self.B[:, O[0]])


        # Updating delta using the likelihood of observation at t given current state and max likelihood
        # of trannsitioning from any prev state to state j

        # psi used to keep track of most likely sequence
        for t in range(1, T):
            for j in range(N):
                delta[t, j] = np.max(delta[t - 1, :] + np.log(self.A[:, j])) + np.log(self.B[j, O[t]])
                psi[t, j] = np.argmax(delta[t - 1, :] + np.log(self.A[:, j]))

        # Initialize Q wih most likely state
        Q = [np.argmax(delta[-1, :])]

        # Append to Q most likely state at time t using psi that we made previously
        for t in range(T - 1, 0, -1):
            Q.append(int(psi[t, Q[-1]]))

        # Correcting the order
        Q.reverse()

        # Return most likely sequence
        return Q, np.max(delta[-1, :])

