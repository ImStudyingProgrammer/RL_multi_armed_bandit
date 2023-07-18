"""
The reinforcement learning practice for multi-armed bandit problem.

Author: Hao-Ying Cheng
Alias: Maskertim
"""

import numpy as np
from env_setup import Environment, Agent
import matplotlib.pyplot as plt
import os

np.random.seed(0) # fixed the random sample

def experiment(probs, eps, episode):
    """
    Running the simulation
    """
    env = Environment(probs)
    agent = Agent(len(probs), eps)
    actions, rewards = [], []
    # start the simulation
    for _ in range(episode):
        action = agent.take_action()
        reward = env.step(action)
        agent.update_Q(action, reward)
        # store the record for action and reward per each step
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)

def output_fig(filename, SAVE=True):
    """ save the file of plot figure"""
    if SAVE:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        plt.savefig(os.path.join(output_dir, "{}.png".format(filename)), bbox_inches= "tight")
    else:
        plt.show()
    plt.close()

if __name__=="__main__":
    # Environment Configuration
    probs = [0.1, 0.2, 0.6, 0.4, 0.8, 0.25, 0.3, 0.75, 0.35, 0.55]
    times = 1000
    episode = 500 # the number of steps
    epsillon = 0.2

    # show the plot
    avg_rewards = np.zeros((episode,))
    count_percent_actions = np.zeros((episode, len(probs)))

    # main entry: run the multi-armed bandit
    print("Runnning multi-armed bandit with actions:{}, total steps:{}".format(len(probs), episode))
    for i in range(times):
       actions, rewards = experiment(probs, epsillon, episode)
       # print out the experiment running state
       if (i+1) % (times / 100) == 0:
           print("Experiment {}/{}\t".format(i+1, times)+
                 "Avg_rewards: {}".format(np.sum(rewards)/len(rewards)))
       avg_rewards += rewards # add two numpy array
       for i,action in enumerate(actions):
           count_percent_actions[i][action] += 1

# path configuration for save files
output_dir = os.path.join(os.getcwd(), "figs")

# Plot the average reward for each steps
# X: Step; Y: Average Reward
avg_rewards = avg_rewards/np.float16(times)
plt.plot(avg_rewards)
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.grid()
plt.xlim([1, episode])
output_fig("avg_rewards")

# Plot the count percentage of action for each steps
for i in range(len(probs)):
    percent_action = 100*count_percent_actions[:,i]/times
    steps = list(np.array(range(len(percent_action)))+1)
    plt.plot(steps, percent_action, linewidth=2, label="Arm {}, success rate:({:.0f}%)".format(i+1, 100*probs[i]))

plt.xlabel("Steps")
plt.ylabel("Trigger Percentage (%)")
ax = plt.gca()
plt.legend(loc="upper left", shadow=True)
plt.xlim([1, episode])
plt.ylim([0,100])
handlers, labels = ax.get_legend_handles_labels()
for handler in handlers:
    handler.set_linewidth(2.0)
output_fig("actions")



    