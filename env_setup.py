import numpy as np

class Environment:
    """
    Environment of Multi-Armed Bandit
    1. The multi-armed
    2. Each arm has its success probability (static)
    3. How many times armed action are taken 
    """
    def __init__(self, probs) -> None:
        self.armed_probs = probs # each arm success prob
        self.armed_count = [] # how many times each armed action is taken

    def step(self, action):
        """
        get the reward if action is taken. If success return 1 else failure return 0
        """
        return 1 if np.random.random() < self.armed_probs[action] else 0

class Agent:
    """
    The Agent would like to get the optimal rewards.
    So the agent takes the action to interact with the environment and get the reward from it.
    The Agent has:
    1. select which armed
    2. the decision policy, here we will use Q-value
    3. take the action by exploration or exploitation, here to use epsilon
    4. the count of actions (for updating Q-value) 
    """
    def __init__(self, n_actions, eps) -> None:
        self.n_actions = n_actions # how many actions we could choose from
        self.eps = eps # epsilon
        self.count_actions = np.zeros(n_actions, dtype=np.int16) # the number of times for taken action
        self.Q = np.zeros(n_actions, dtype=np.float16) # the Q-value for each action

    def update_Q(self, action, reward) -> None:
        """ update Q value of each action """
        self.count_actions[action] += 1 # increment the action count
        # update Q action: Q_{k+1}(a) = Q_{k+1}(a) + 1.0/(k+1)*(r_{k+1}-Q_k(a))
        self.Q[action] += (1.0/self.count_actions[action])*(reward-self.Q[action])

    def take_action(self):
        """
        take the action by epsilon-greedy algorithm
        """
        if np.random.random() < self.eps:
            # explore
            return np.random.randint(self.n_actions)
        else:
            # exploit
            return np.random.choice(np.flatnonzero([q == max(self.Q) for q in self.Q]))
