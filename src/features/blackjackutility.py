import numpy as np
import matplotlib.pyplot as plt

def get_total(Input):
    """
    This function will calculate the maximum total of a hand in 21. 
    It will also take into consideration if the Deck has aces.  

    Returns: 
    Maxmum value of a deck of card 
    Args:
    Input:List of numbers representing  a hand 

    """
    Input=np.array(Input)
    #maximum player hand under or equal to 21 
    Max=0
    #check if the hand has an ace
    if 1 in set(Input):
        #put all aces in one array 
        aces=Input[Input==1]
        #all  other cards  in a second array 
        not_aces=Input[Input!=1]
        #try different posable combinations of aces as 1 or 11 
        for ace in range(len(aces)+1):
            #convert each  ace to an 11 
            aces[0:ace]=11
            #find the total of a particular combination 
            total=aces.sum()+not_aces.sum()
            # check if the total is 21 
            if total==21:
                Max=21
                break
            #check if the total is larger than Max value but less than 22 

            elif total>Max and total<22:
                #if so total is new max
                Max=total

    else:
        #determine  sum if no aces in the deck 
        Max=sum(Input)

    return Max  

def game_result (environment,state,show=True):
    '''
    this function  will determine the results of  a game  of Black Jack after an episode only  tested for open AI  gym 
    Returns: 
    result:   used to debug result of a game like open AI  gym +1,drawing is 0, and losing is -1, None for error 
    Args:
    environment: open ai gym black jack environment
    state: state of open ai gym black jack environment
    '''
    if show:
        print("state:",state)
        print("player has", environment.player)
        print("the players current sum:{},dealer's one showing card:{}, usable ace:{}".format(state[0],state[1],state[2]))
    dealer_sum=get_total(environment.dealer)
    result=None
    if show:
        print("dealer cards: {} and score: {} your score i: {} ".format(environment.dealer,dealer_sum,state[0]))
    if state[0]>21:
        if show:
            print("Bust")
        result=-1
    elif dealer_sum>21:
        if show:
            print("agent wins")
        result=1  
        
    elif state[0]>dealer_sum and state[0]<22:
        if show:
            print("agent  wins")
        result=1
        
    elif  state[0]<dealer_sum and dealer_sum<22 : 
        if show:
            print("agent  loses")
        result=-1
    return result 

#This function calculates the average number of wins, losses and draws for a game of blackjack given a policy:
def average_results_with_plot(environment, policy=None, episodes=10, plot = False):
    """
    This function calculates and plots the evolution of the average win, loss and draw rates 
    for a blackjack game given a policy over several episodes.
    If no policy is provided, a random policy is selected.
    
    Args:
    - environment: gym AI blackjack environment object.
    - policy: Policy for blackjack; if none, a random action is selected.
    - episodes: Number of episodes to be simulated.

    Returns: 
    - average_wins: The average number of wins in the last episode.
    - average_losses: The average number of losses in the last episode.
    - average_draws: The average number of draws in the last episode.
    """

    # Initialize win, loss and draw counters
    wins, losses, draws = 0, 0, 0

    # Lists to store the average rates after each episode
    win_rate, loss_rate, draw_rate = [], [], []

    for episode in range(1, episodes + 1):
        # Initializes the episode's state and completion flag
        state = environment.reset()
        done = False

        while not done:
            # Choose an action based on the provided policy or randomly
            action = policy[state] if policy and isinstance(policy[state], np.int64) else environment.action_space.sample()
            state, reward, done, info = environment.step(action)

        # Update counters based on the episode result
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else: # reward == 0, which represents a draw
            draws += 1

        # Calculates the averages after each episode
        win_rate.append(wins / episode)
        loss_rate.append(losses / episode)
        draw_rate.append(draws / episode)

    if plot:

        # Plotting the results
        plt.plot(win_rate, label='Average Wins')
        plt.plot(loss_rate, label='Average Losses')
        plt.plot(draw_rate, label='Average Draws')
        plt.xlabel('Episodes')
        plt.ylabel('Average Rate')
        plt.title('Average Win, Loss, and Draw Rates over Episodes')
        plt.legend()
        plt.show()

    # Returns the final averages
    return win_rate[-1], loss_rate[-1], draw_rate[-1]

# define a function that plays n games and computes the percentage of wins, drwas and losses
def play_n_games(agent, environment, n_games):
    wins = 0
    draws = 0
    losses = 0
    agent.epsilon = 0
    
    for _ in range(n_games):
        state = environment.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, done, info = environment.step(action)
        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1
    return wins / n_games, losses / n_games, draws / n_games


def games_with_policy(environment, policy = None, episodes = 10):
    sum_ = 0

    for episode in range(episodes):
        state = environment.reset()
        done = False
        print("_________________________________________")
        print("Episode {}".format(episode))
    

        print("State: {}".format(state))
        print("The Agent's current sum: {}, The Dealer's one showing card: {}, Agent's usable ace: {}".format(state[0],state[1],state[2]))
        print("The Agent has the following cards: {}".format(environment.player))
        print('The Dealer has the following cards: {}'.format(environment.dealer))
        while not done:
        
            if policy and isinstance(policy[state], np.int64):
                    
                action = policy[state]
                    
            else:
                action = environment.action_space.sample()
        
            if action:
                print("Hit")
                
            else:
                print("Stand")

            state, reward, done, info = environment.step(action)

            print("State: {}".format(state))
            print("The Agent's current sum: {}, The Dealer's one showing card: {}, Agent's usable ace: {}".format(state[0],state[1],state[2]))
            print("The Agent has the following cards: {}".format(environment.player))
            print('The Dealer has the following cards: {}'.format(environment.dealer))

        print("Done: {}".format(done))
        result = game_result(environment, state)
        sum_ += reward
    print('Total reward: {}'.format(sum_))

def random_action(action, epsilon = 0.1, n_actions = 2):
    ''' 
    This function takes the best estimated action, eplsilon, and action space 
    and returns some action. 
    '''
    # generate a random number from 0 to 1.
    number = np.random.rand(1)
    
    # if number is smaller than 1-epsilon then return best estimated action
    if number < 1 - epsilon:
        return action
    # if number is bigger or equals to 1-epsilon then return some random action from the action space
    else:
        action = np.random.randint(n_actions)  
        return action 

#Function for the custom reward 
def reward_function(state, action, agent_type, player_sum):
    """
    Função de recompensa modificada para diferentes tipos de jogadores.

    Args:
    state: estado atual do jogo
    action: ação tomada pelo jogador
    agent_type: tipo do jogador (Conservador, Neutro, Agressivo)
    player_sum: soma das cartas do jogador

    Returns:
    reward: recompensa modificada
    """

    if agent_type == "Conservador":
        if player_sum < 12 and action == 0:  # Ação de Ficar
            reward = 0.3  # Recompensa por ficar
        elif action == 0:
            reward = 0.1  # Recompensa por não pedir carta
        else:
            reward = -0.2  # Penalização por pedir carta

    elif agent_type == "Neutro":
        if player_sum <= 15 and action == 1:  # Ação de Pegar carta
            reward = 0.3  # Recompensa por pedir carta
        else:
            reward = -0.1  # Penalização por ficar

    elif agent_type == "Agressivo":
        if player_sum > 15 and action == 1:  # Ação de Pegar carta
            reward = 0.3  # Recompensa por pedir carta
        elif player_sum > 21:
            reward = -0.3  # Penalização por ultrapassar 21
        elif action == 1:
           reward = 0.1  # Recompensa por pedir carta
        else:
            reward = -0.1  # Penalização por ficar

    else:
        reward = 0.0

    return reward

def draw_till_17_pol(obs):
    return [1,0] if obs[0]<17 else [0,1]

def create_epsilon_greedy_action_policy(env,Q,epsilon):
    """ Create epsilon greedy action policy
    Args:
        env: Environment
        Q: Q table
        epsilon: Probability of selecting random action instead of the 'optimal' action
    
    Returns:
        Epsilon-greedy-action Policy function with Probabilities of each action for each state
    """
    def policy(obs):
        P = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n  #initiate with same prob for all actions
        best_action = np.argmax(Q[obs])  #get best action
        P[best_action] += (1.0 - epsilon)
        return P
    return policy

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation state as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(obs):
        return A
    return policy_fn

def create_greedy_action_policy(env,Q):
    """ Create greedy action policy
    Args:
        env: Environment
        Q: Q table
    
    Returns:
        Greedy-action Policy function 
    """
    def policy(obs):
        P = np.zeros_like(Q[obs], dtype=float)
        best_action = np.argmax(Q[obs])  #get best action
        P[best_action] = 1
        return P
    return policy