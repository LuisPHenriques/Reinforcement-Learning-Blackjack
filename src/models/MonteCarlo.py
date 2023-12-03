import numpy as np
from collections import defaultdict
from src.features.blackjackutility import random_action, create_greedy_action_policy
from IPython.display import clear_output
from src.features.blackjackutility import reward_function

def monte_carlo_on_policy(environment, N_episodes = 100000, discount_factor = 1, first_visit = True, epsilon = 0.1, theta = 0.0001, agent_type=None):
    """
    plot the policy for blackjack 
    Returns:   
    policy: a dictionary of estimated policy for blackjack 
    V: a dictionary of estimated values for blackjack 
    Q: a dictionary of estimated action function
    DELTA: list of deltas for each episode 
    Args:
    environment:AI gym balckjack envorment object 
    N_episodes:number of episodes 
    discount_factor:discount factor
    first_visit: select first-visit MC (True) and every-visit MC (False)
    epsilon: epsilon value 
    theta:stoping threshold
    """  
    #a dictionary of estimated values for blackjack 
    V = defaultdict(float)
    #a dictionary of estimated action function for blackjack
    Q = defaultdict(float)
    # number of visits to the action function 
    NumberVisitsValue = defaultdict(float)
    # visits to action function
    NumberVisits = defaultdict(float)
    #dictionary  for policy 
    policy = defaultdict(float) 
    #number  of actions 
    number_actions = environment.action_space.n
    #list of max difference between  value functions per  iteration 
    DELTA = []

    for i in range(N_episodes):
        #max difference between  value functions
        delta = 0
        #list that stores each state and reward for each episode     
        episode =[ ]
        # reset the  environment for the next episode and find first state  
        state = environment.reset()   
        #reward for the first state
        reward = 0.0
        #flag for end of episodes  
        done = False
        #action for the first state 
        action = np.random.randint(number_actions)
        #append firt state, reward and action
        episode.append({'state':state, 'reward':reward, 'action':action})
        #Past states for signal visit  Monte Carlo 
        state_action = [(state,action)]
        #enumerate for each episode 
        while not done:

                #take action and find next state, reward and check if the episode is  done (True)
                (state, reward, done, prob) = environment.step(action)

                if agent_type is not None:        
                    custom_reward = reward_function(state, action, agent_type=agent_type, player_sum=state[0])
                    reward += custom_reward

                #check if a policy for the state exists  
                if isinstance(policy[state],np.int64):
                    #obtain action from policy
                    action = int(policy[state])
                    random_action(action, epsilon, number_actions)
                else:
                     #if no policy for the state exists  select a random  action  
                    action = np.random.randint(number_actions)
                #add state reward and action to list 
                episode.append({'state':state, 'reward':reward, 'action':action})
                #add  states action this is for fist visit only 
                state_action.append((state,action))
         #reverse list as the return is calculated from the last state
        episode.reverse()
        #append the state-action pairs to a list 
        state_action.reverse()


        #determine the return
        G = 0

        for t,step in enumerate(episode):

                #check flag for first visit
                G = discount_factor * G + step['reward']
                #check flag for first visit
                if first_visit:
                    #check if the state has been visited before 
                    if (step['state'],step['action']) not in set(state_action[t + 1:]): 

                        #increment counter for action 
                        NumberVisits[step['state'],step['action']] += 1
                        #increment counter for value function 
                        NumberVisitsValue[step['state']] += 1
                        #if the action function value  does not exist, create an array  to store them 
                        if not isinstance(Q[step['state']],np.ndarray):
                            Q[step['state']] = np.zeros((number_actions))

                        #calculate mean of action function Q Value functions V using the  recursive definition of mean 
                        Q[step['state']][step['action']] = Q[step['state']][step['action']] + (NumberVisits[step['state'], step['action']] ** -1) * (G - Q[step['state']][step['action']])
                        
                        # record the old value of the value function 

                        v = V[step['state']]
                        
                        V[step['state']] = V[step['state']] + (NumberVisitsValue[step['state']] ** -1) * (G - V[step['state']])
                        #update the policy to select the action fuciton argment with the largest value 
                        policy[step['state']] = np.random.choice(np.where(Q[step['state']] == Q[step['state']].max())[0])
                        #find max difference between all value functions per  iteration 
                        delta = max(delta,abs(v - V[step['state']]))

                else:
                         #increment counter for action 
                        NumberVisits[step['state'],step['action']] += 1
                        #increment counter for value function 
                        NumberVisitsValue[step['state']] += 1
                        #if the action function value  does not exist, create an array  to store them 
                        if not isinstance(Q[step['state']], np.ndarray):
                            Q[step['state']] = np.zeros((number_actions))

                        #calculate mean of action function Q Value functions V using the  recursive definition of mean 
                        Q[step['state']][step['action']] = Q[step['state']][step['action']] + (NumberVisits[step['state'], step['action']] ** -1) * (G-Q[step['state']][step['action']])
                        v = V[step['state']]
                        V[step['state']] = V[step['state']] + (NumberVisitsValue[step['state']] ** -1) * (G - V[step['state']])
                        ##update the policy to select the action functioon argument with the largest value 
                        policy[step['state']] = np.random.choice(np.where(Q[step['state']] == Q[step['state']].max())[0])
                        #find max difference between all value functions per iteration 
                        delta = max(delta, abs(v - V[step['state']]))
            
        DELTA.append(delta)
        if delta < theta:
            break

    return policy, V, Q, DELTA

def monte_carlo_off_policy(environment, N_episodes=200000, discount_factor=1, epsilon=0.1, theta=0, agent_type=None):
    # Initializes Q as a dictionary that returns a default dictionary of zeros.
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))
    C = defaultdict(float)  # State-action pair frequency counter
    policy = {}  # The policy that is being improved
    number_actions = environment.action_space.n  # Number of possible actions
    DELTA = [] # To store the maximum Q differences for each episode

    for i in range(N_episodes):
        episode = []
        state = environment.reset()
        done = False
        while not done:
            # The behavior policy can be different here, like a random policy.
            action = np.random.choice(number_actions, p=behavior_policy(state))
            #next_state, reward, done, _ = environment.step(action)
            next_state, reward, done, info, extra = environment.step(action)

            # Add the personalized reward to the environment reward
            if agent_type is not None:        
                custom_reward = reward_function(state, action, agent_type=agent_type, player_sum=state[0])
                reward += custom_reward

            episode.append((state, action, reward))
            state = next_state

        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]

            G = discount_factor * G + reward
            C[state, action] += W
            Q[state][action] += (W / C[state, action]) * (G - Q[state][action])
            
           # Updates the policy for the action with the highest Q-value in the state.
            policy[state] = np.argmax(Q[state])
            
            if action != policy[state]:
                break  # If the action taken is not optimal, the episode no longer contributes.
            
            W *= 1. / behavior_policy(state)[action]  # Atualiza o peso de amostragem.

        # If necessary, calculate the maximum difference in Q to check for convergence.
        # This code assumes that you want to check convergence.
        delta = max(abs(Q[s][a] - old_Q) for s, a_values in Q.items() for a, old_Q in enumerate(a_values))
        DELTA.append(delta)
        if delta < theta:
            break

    return policy, Q, DELTA

def behavior_policy(state):
    # Retorna a distribuição de probabilidade de ações para a política de comportamento.
    # Isso é apenas um exemplo e deve ser substituído pela sua política de comportamento real.
    return np.array([0.5, 0.5])  # Por exemplo, uma política aleatória em um espaço de ação com 2 ações.

# Suponha que `environment` seja um objeto de ambiente do Gym que você tenha definido em outro lugar.
# policy, Q, DELTA = monte_carlo_off_policy(environment)

    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: Environment.
        num_episodes: Number of episodes to sample.
        policy: The policy to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedy policy 
    target_policy = create_greedy_action_policy(env,Q)
        
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            clear_output(wait=True)

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            # Sample an action from our policy
            probs = target_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Update the total reward since step t
            G = discount_factor * G + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula 
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            # If the action taken by the policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action !=  np.argmax(target_policy(state)):
                break
            W = W * 1./policy(state)[action]
        
    return Q, target_policy