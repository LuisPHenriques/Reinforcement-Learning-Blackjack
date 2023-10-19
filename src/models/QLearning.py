import numpy as np

from src.features.blackjackutility import random_action, game_result


def QLearning(env, episodes = 1000000, alpha = 0.1, gamma = 0.6, epsilon = 0.1):

    unique_states = []
    for player_hand in range(1,33):
        for dealer_hand in range(1,12):
            for ace_present in range(0,2):
                identifier = str(player_hand) + '_' + str(dealer_hand) + '_' + str(ace_present)
                unique_states.append(identifier)

    q_table = {}
    for unique_state in unique_states:
        q_table[unique_state] = {}
        for hit_or_stay in range(0,2):
            q_table[unique_state][hit_or_stay] = 0

    #number  of actions 
    number_actions = env.action_space.n

    for i in range(1, episodes + 1):

        # Initialize environment, assign state code
        done = False
        state = env.reset()
        if state[2] == False:
            is_ace = 0
        else :
            is_ace = 1
        state_code = str(state[0]) + '_' + str(state[1]) + '_' + str(is_ace)

        # While game still active, select action and get next state, reward, new environment
        while not done:
                
            to_consider = [q_table[state_code][0], q_table[state_code][1]]
            action = np.argmax(to_consider) # Exploit learned values
            random_action(action, epsilon, number_actions)

            next_state, reward, done, info = env.step(action)

            if next_state[0] < 21:
                if next_state[2] == False:
                    next_is_ace = 0
                else :
                    next_is_ace = 1
                next_state_code = str(next_state[0]) + '_' + str(next_state[1]) + '_' + str(next_is_ace)

                old_value = q_table[state_code][action]

                next_to_consider = [q_table[next_state_code][0], q_table[next_state_code][1]]
                next_max = np.max(next_to_consider)

                # correct_new_value considering the formula in the slides
                # correct_new_value = old_value + alpha * (reward + (gamma * next_max) - old_value)
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state_code][action] = new_value

            else:

                # correct_new_value considering the formula in the slides
                # correct_new_value = old_value + alpha * (reward + (gamma * next_max) - old_value)
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state_code][action] = new_value

    print("Training finished.\n")

    # Create a 4D NumPy array filled with zeros
    shape = (32, 11, 2, 2)
    q_table_new = np.zeros(shape)

    # Fill the 4D array with values from the dictionary
    for key, value in q_table.items():
        indices = [int(i) for i in key.split('_')]
        q_table_new[indices[0] - 1][indices[1] - 1][indices[2]][0] = value[0]
        q_table_new[indices[0] - 1][indices[1] - 1][indices[2]][1] = value[1]

    return q_table_new


def games_with_policy_QLearning(environment, q_table = None, episodes = 10):   
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
                        
            action = np.argmax(q_table[state[0]][state[1]][int(state[2])])
            
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


def average_wins_QLearning(environment, q_table = None, episodes = 200000):
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

            action = np.argmax(q_table[state[0]][state[1]][int(state[2])])

            state, reward, done, info = environment.step(action)

        if reward == 1:
            win_loss[episode] = 1
        else:
            win_loss[episode] = 0  

        
    average_wins = win_loss.mean()
    std_win = win_loss.std() / np.sqrt(episodes)

    return average_wins, std_win