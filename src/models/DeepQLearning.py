import torch
import itertools
from collections import namedtuple, deque

class DeepQLearningAgent(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DeepQLearningAgent, self).__init__()
        # Fully connected layers
        self.layer_obs_4096 = torch.nn.Linear(n_observations, 4096)
        self.layer_4096_2048 = torch.nn.Linear(4096, 2048)
        self.layer_2048_1024 = torch.nn.Linear(2048, 1024)
        self.layer_1024_act = torch.nn.Linear(1024, n_actions)
        # Activation function
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU()
        # Batch normalization
        self.batch_norm4096 = torch.nn.BatchNorm1d(4096)
        self.batch_norm2048 = torch.nn.BatchNorm1d(2048)
        self.batch_norm1024 = torch.nn.BatchNorm1d(1024)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.relu(self.batch_norm4096(self.layer_obs_4096(x)))
        x = self.relu(self.batch_norm2048(self.layer_4096_2048(x)))
        x = self.relu(self.batch_norm1024(self.layer_2048_1024(x)))
        x = self.layer_1024_act(x)

        return x

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')

    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device


def select_action(state, policy_network, device, eps_start = 0.9, eps_end = 0.05, eps_decay = 1000):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * torch.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_network(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)
    

def optimize_model(policy_network, target_network, memory, optimizer, device, gamma = 0.9, mini_batch_size = 128):

    if len(memory) < mini_batch_size:
        return
    
    transitions = memory.sample(mini_batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype = torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_network(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(mini_batch_size, device = device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)

    optimizer.step()


def train_deepqlearning(env, policy_network, target_network, memory, optimizer, device, num_episodes = 1000, TAU = 0.01):

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in itertools.count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_network = policy_network, 
                           target_network = target_network, 
                           memory = memory, 
                           optimizer = optimizer, 
                           device = device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_network.state_dict()
            policy_net_state_dict = policy_network.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)

            target_network.load_state_dict(target_net_state_dict)

            if done:
                break