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