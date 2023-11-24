import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import seaborn as sns

from collections import defaultdict
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


# We will define some functions to help us train the algorithm and visualize the game results. 

# This function is used to plot the value function:
def plot_value_function_old(V):
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


def calc_payoffs(env,rounds,players,pol):
    """
    Calculate Payoffs.
    
    Args:
        env: environment
        rounds: Number of rounds a player would play
        players: Number of players 
        pol: Policy used
        
    Returns:
        Average payoff
    """
    average_payouts = []
    for player in range(players):
        rd = 1
        total_payout = 0 # to store total payout over 'num_rounds'

        while rd <= rounds:
            obs, _, _, _ = env.step(env.action_space.sample()) # Get initial observation
            action = np.argmax(pol(obs))
            obs, payout, is_done, _ = env.step(action)
            if is_done:
                total_payout += payout
                env.reset()
                rd += 1
        average_payouts.append(total_payout / rounds) # Store the average payout

    plt.plot(average_payouts)
    plt.xlabel('Player')
    plt.ylabel('Average payout after ' + str(rounds) + ' rounds')
    plt.show()
    average_total_payout = sum(average_payouts) / players
    print("Average payout of a player after {} rounds is {}".format(rounds, average_total_payout))
    return average_total_payout


def plot_policy(policy):

    """ This function visualizes a given policy for a blackjack game.
    # The policy is represented as a mapping from states (player's hand, dealer's showing card, and whether there's a usable ace) to actions.
    # The plot displays whether to hit or stick (0 or 1) across different game states.     """

    # Helper function to determine the action for a given state.
    # If the state is not explicitly in the policy dictionary, it defaults to 1 (hit).
    def get_Z(player_hand, dealer_showing, usable_ace):
        return policy.get((player_hand, dealer_showing, usable_ace), 1)

    # Helper function to prepare the plot for a given ace condition (usable or not).
    # It sets up the axes, labels, and color map for the plot.
    def get_figure(usable_ace, ax):
        x_range = np.arange(1, 11)  # Dealer showing card range
        y_range = np.arange(11, 22)  # Player hand range
        X, Y = np.meshgrid(x_range, y_range)  # Create a grid over the ranges
        Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace) for dealer_showing in x_range] for player_hand in range(21, 10, -1)])  # Determine actions for the grid
        surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 2), vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])  # Plot the actions
        plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))  # Set x-ticks to show card values
        plt.yticks(y_range)  # Set y-ticks to show player's hand values
        ax.set_xlabel('Dealer Showing')  # X-axis label
        ax.set_ylabel('Player Hand')  # Y-axis label
        ax.grid(color='black', linestyle='-', linewidth=1)  # Add a grid for clarity
        # Setup the colorbar to show action labels
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])  # Set labels for actions
        cbar.ax.invert_yaxis() 
            
    # Setup the figure and axes for the two ace conditions
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace', fontsize=16)  # Title for the usable ace plot
    get_figure(True, ax)  # Plot for usable ace
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace', fontsize=16)  # Title for the no usable ace plot
    get_figure(False, ax)  # Plot for no usable ace
    plt.show()

def plot_value_function(V, title="Value Function"):
    """ This function plots the value function of a policy in a 3D surface plot.
    The value function maps each state to the expected return from that state under the policy.
    The plot shows the values across different states, for both conditions of having a usable ace or not. """
    min_x = min(k[0] for k in V.keys())  # Minimum player hand value
    max_x = max(k[0] for k in V.keys())  # Maximum player hand value
    min_y = min(k[1] for k in V.keys())  # Minimum dealer showing card value
    max_y = max(k[1] for k in V.keys())  # Maximum dealer showing card value

    x_range = np.arange(min_x, max_x + 1)  # Player hand range
    y_range = np.arange(min_y, max_y + 1)  # Dealer showing card range
    X, Y = np.meshgrid(x_range, y_range)  # Create a grid over the ranges

    # Find the value function for all states with and without a usable ace
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    # Helper function to plot the surface for given data
    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')  # X-axis label
        ax.set_ylabel('Dealer Showing')  # Y-axis label
        ax.set_zlabel('Value')  # Z-axis (value) label
        ax.set_title(title)  # Title for the plot
        ax.view_init(ax.elev, -120)  # Set the view angle
        fig.colorbar(surf)  # Add a colorbar to show the value scale
        plt.show()

    # Plot the value function for both ace conditions
    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))  # Plot for no usable ace
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))  # Plot for usable ace