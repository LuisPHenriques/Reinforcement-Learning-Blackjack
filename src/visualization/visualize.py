import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from matplotlib.patches import Patch


# We will define some functions to help us train the algorithm and visualize the game results. 

# This function is used to plot the value function:
def plot_value_function(V):
    """
    plot the estimated value function for blackjack 
    Returns:  void plots value function 
    Args:
    V: a dictionary of estimated values for blackjack 
    """
    #range of player score  
    player = [state[0] for state in V.keys()]
    max_player = max(player)
    min_player = min(player)
    player_range = np.arange(min_player, 22, 1)
    #range of dealer score      
    dealer = [state[1] for state in V.keys()]
    max_dealer = max(dealer)
    min_dealer = min(dealer)
    dealer_range = np.arange(min_dealer, 11, 1)
    #empty array for the value function, first access in the players score second  is the dealer, third is if  there  is an ace    
    V_plot = np.zeros((21 - min_player + 1, max_dealer - min_dealer + 1, 2))
    #create a mesh grid for plotting 
    X, Y = np.meshgrid(dealer_range, player_range)

    #populate an array  of values for different  scores not including losing scores 
    for (player,dealer,ace), v in V.items():
        if player <= 21 and dealer <= 21:
            V_plot[player - min_player, dealer - min_dealer, (1 * ace)] = V[(player, dealer, ace)]

    #plot surface        
    fig, ax = plt.subplots(nrows = 1, ncols = 2, subplot_kw = {'projection': '3d'})
    ax[0].plot_wireframe(X, Y, V_plot[:,:,0])
    ax[0].set_title('no ace')
    ax[0].set_xlabel('dealer')
    ax[0].set_ylabel('player ')
    ax[0].set_zlabel('value function')
    ax[1].plot_wireframe(X, Y, V_plot[:,:,1])
    ax[1].set_title('no ace')
    ax[1].set_xlabel('dealer')
    ax[1].set_ylabel('player ')
    ax[1].set_zlabel('value function')
    ax[1].set_title(' ace')
    fig.tight_layout()
    plt.show()

    #plot top view of the surface     
    fig, ax = plt.subplots(nrows = 1, ncols = 2)   
    ax[0].imshow((V_plot[:, :, 0]), extent =[1, 10, 21, 4])
    ax[0].set_title('no ace')
    ax[0].set_xlabel('dealer')
    ax[0].set_ylabel('player ')   
    im=ax[1].imshow(V_plot[:, :, 1],extent =[1, 10, 21, 4])
    ax[1].set_title('ace')
    ax[1].set_xlabel('dealer')
    fig.colorbar(im, ax = ax[1])


# This function will plot blackjack policy:
def plot_policy_blackjack(policy):
    """
    plot the policy for blackjack 
    Returns:  void plots policy function 
    Args:
    policy: a dictionary of estimated values for blackjack 
    """    
    #range of player score 
    player = [state[0] for state in  policy.keys()]
    max_player = max(player)
    min_player = min(player)
    #this vale is use in RL book 
    #min_player=12
    player_range = np.arange(min_player, 22, 1)
    #range of dealer score      
    dealer = [state[1] for state in policy.keys()]
    max_dealer = max(dealer)
    min_dealer = min(dealer)
    dealer_range = np.arange(min_dealer, 11, 1)
    #empty array for the value function, first access in the players score second  is the dealer, third is if  there  is an ace    
    policy_plot = np.zeros((21 - min_player + 1, max_dealer - min_dealer + 1, 2))
    #create a mesh grid for plotting 
    X, Y = np.meshgrid(dealer_range, player_range)
    
    
    #populate an array  of values for different  policy not including losing states above 21 
    for (player, dealer, ace), v in policy.items():
        if player <= 21 and dealer <= 10 and player >= min_player:
            policy_plot[player - min_player, dealer - min_dealer, (1 * ace)] = policy[(player, dealer, ace)]

    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)   
    ax[0].imshow((policy_plot[:, :, 0]),cmap = plt.get_cmap('GnBu', 2),extent = [1,10,21,4])
    ax[0].set_title('no ace')
    ax[0].set_xlabel('dealer')
    ax[0].set_ylabel('player ') 
    

    ax[1].set_title('ace')
    ax[1].set_xlabel('dealer')
    im=ax[1].imshow(policy_plot[:, :, 1],extent = [1, 10, 21, 4], cmap = plt.get_cmap('GnBu', 2))
    fig.colorbar(im, ax = ax[1], ticks=[0, 1])


def training_results(agent, env, rolling_length = 500):

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig
