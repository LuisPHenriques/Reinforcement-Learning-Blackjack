import numpy as np
from collections import defaultdict
from src.features.blackjackutility import random_action

def monte_carlo(environment, N_episodes=100000, discount_factor=1, epsilon=0.1, theta=0.0001):
    """
    Simple Monte Carlo for Blackjack to estimate state values.
    Returns:
    V: a dictionary of estimated values for blackjack
    DELTA: list of deltas for each episode
    Args:
    environment: AI gym blackjack environment object
    N_episodes: number of episodes
    discount_factor: discount factor
    epsilon: epsilon value
    theta: stopping threshold
    """
    # Dictionary of estimated values for blackjack, initialized to 0
    V = defaultdict(float)
    # Dictionary to track the number of visits to each state
    N = defaultdict(int)
    # Number of actions
    number_actions = environment.action_space.n
    # List of max difference between value functions per iteration
    DELTA = []

    for i in range(N_episodes):
        # Max difference between value functions
        delta = 0
        # List that stores each state and reward for each episode
        episode = []

        # Reset the environment for the next episode and find the first state
        state = environment.reset()
        done = False

        while not done:
            # Select action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.randint(number_actions)
            else:
                action = np.argmax([V[state, a] for a in range(number_actions)])

            # Take action and observe next state and reward
            next_state, reward, done, _ = environment.step(action)
            
            # Store the state and reward in the episode
            episode.append((state, reward))
            
            # Update state for the next iteration
            state = next_state
        
        # Calculate returns and update value function
        G = 0
        for state, reward in reversed(episode):
            G = discount_factor * G + reward
            N[state] += 1  # Track the number of visits to each state
            V[state] += (G - V[state]) / N[state]  # Update using running average
            delta = max(delta, abs(G - V[state]))

        DELTA.append(delta)
        if delta < theta:
            break

    return V, DELTA



