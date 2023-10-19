import numpy as np

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



#This function calculates the average number of wins for a game of blackjack given a policy:

def average_wins(environment, policy = None, episodes = 10):
    """
    This function calculates the average number of wins for a game of blackjack given a policy.
    If no policy is provided a random policy is selected.
    Returns: average_wins: the average number of wins 
    std_wins: the average number of wins 
    Args:
    environment:AI gym balckjack envorment object 
    policy:policy for blackjack if none a random  action will be selected 
    episodes: number of episodes 
    """

    win_loss = np.zeros(episodes)

    for episode in range(episodes):
        state = environment.reset()
        done = False

        while not done:
            if policy and isinstance(policy[state],np.int64):
                 
                action = policy[state]
                
            else:
                action = environment.action_space.sample()

            state, reward, done, info = environment.step(action)
        result = game_result(environment, state, show = False)
        if reward == 1:
            win_loss[episode] = 1
        else:
            win_loss[episode] = 0  

        
    average_wins = win_loss.mean()
    std_win = win_loss.std() / np.sqrt(episodes)

    return average_wins, std_win


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
    