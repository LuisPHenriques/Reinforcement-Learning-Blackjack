import numpy as np

from collections import defaultdict
from src.features.blackjackutility import random_action


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
