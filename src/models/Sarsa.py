import numpy as np

from collections import defaultdict
from tqdm import tqdm
from src.features.blackjackutility import  create_epsilon_greedy_action_policy
from IPython.display import clear_output

from src.features.blackjackutility import reward_function

class SarsaAgent:
    def __init__(self, environment, agent_type = None, learning_rate = 0.001, initial_epsilon = 0.1, epsilon_decay = 0.05, final_epsilon = 0.01, discount_factor = 0.95):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(environment.action_space.n))
        self.env = environment
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.agent_type = agent_type

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs: tuple[int, int, bool], action: int, next_action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool]):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]

        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def train(self, env, n_episodes):

        for episode in tqdm(range(n_episodes)):
            obs = env.reset()
            done = False
            rounds = 0
            # play one episode
            while not done:
                if rounds == 0:
                    action = self.get_action(obs)
                    next_obs, reward, terminated, truncated = env.step(action)
                    next_action = self.get_action(next_obs)

                    if self.agent_type is not None:        
                        custom_reward = reward_function(obs, action, agent_type = self.agent_type, player_sum = obs[0])
                        reward += custom_reward

                    # update the agent
                    self.update(obs, action, next_action, reward, terminated, next_obs)

                    # update if the environment is done and the current obs
                    done = terminated or truncated

                    action = next_action
                    obs = next_obs

                    rounds += 1

                else:
                    next_obs, reward, terminated, truncated = env.step(action)
                    next_action = self.get_action(next_obs)

                    if self.agent_type is not None:        
                        custom_reward = reward_function(obs, action, agent_type = self.agent_type, player_sum = obs[0])
                        reward += custom_reward

                    # update the agent
                    self.update(obs, action, next_action, reward, terminated, next_obs)

                    # update if the environment is done and the current obs
                    done = terminated or truncated

                    action = next_action
                    obs = next_obs

            self.decay_epsilon()

def SARSA(env, episodes, epsilon, alpha, gamma):
    """
    SARSA Learning Method
    
    Args:
        env: OpenAI gym environment.
        episodes: Number of episodes to sample.
        epsilon: Probability of selecting random action instead of the 'optimal' action
        alpha: Learning Rate
        gamma: Gamma discount factor
        
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. 
    """
    
    # Initialise a dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The policy we're following
    pol = create_epsilon_greedy_action_policy(env,Q,epsilon)
    for i in range(1, episodes + 1):
        # Print out which episode we're on
        if i% 1000 == 0:
            print("\rEpisode {}/{}.".format(i, episodes), end="")
            clear_output(wait=True)
        curr_state = env.reset()
        probs = pol(curr_state)   #get epsilon greedy policy
        curr_act = np.random.choice(np.arange(len(probs)), p=probs)
        while True:
            next_state,reward,done,_ = env.step(curr_act)
            next_probs = create_epsilon_greedy_action_policy(env,Q,epsilon)(next_state)
            next_act = np.random.choice(np.arange(len(next_probs)),p=next_probs)
            td_target = reward + gamma * Q[next_state][curr_act]
            td_error = td_target - Q[curr_state][curr_act]
            Q[curr_state][curr_act] = Q[curr_state][curr_act] + alpha * td_error
            if done:
                break
            curr_state = next_state
            curr_act = next_act
    return Q, pol