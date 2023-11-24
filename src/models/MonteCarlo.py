import numpy as np
from collections import defaultdict
from src.features.blackjackutility import random_action, create_epsilon_greedy_action_policy, create_greedy_action_policy
from IPython.display import clear_output

def monte_carlo_ES(environment, N_episodes = 100000, discount_factor = 1, first_visit = True, epsilon = 0.1, theta = 0.0001):
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

def On_pol_mc_control_learn(env, episodes, discount_factor, epsilon):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: Environment.
        episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state to action values.
        Policy is the trained policy that returns action probabilities
    """
    # Keeps track of sum and count of returns for each state
    # An array could be used to save all returns but that's memory inefficient.
    # defaultdict used so that the default value is stated if the observation(key) is not found
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    pol = create_epsilon_greedy_action_policy(env,Q,epsilon)
    
    for i in range(1, episodes + 1):
        # Print out which episode we're on
        if i% 1000 == 0:
            print("\rEpisode {}/{}.".format(i, episodes), end="")
            clear_output(wait=True)

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = pol(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            #First Visit MC:
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
    
    return Q, pol

def Off_pol_mc_control_learn(env, num_episodes, policy, discount_factor):
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