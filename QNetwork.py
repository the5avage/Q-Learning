import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from collections import deque
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, ExponentialLR
from truncated_normal_distribution import *
from ReplayBuffer import ReplayBuffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def warmup_lr_lambda(current_step, warmup_steps):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0

class QLearningAgent:
    def __init__(self, action_size, n, learning_rate, gamma,
                        epsilon, epsilon_decay, epsilon_min,
                        warmup_steps, learning_rate_decay,
                        stored_episodes, samples_per_episode, alpha=0.1):
        self.action_size = action_size
        self.n = n  # Number of previous states and actions to consider
        self.input_size = n * 2 + 2  # Last n actions and states + current state and setpoint
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.alpha = alpha
        #self.stored_episodes = stored_episodes
        #self.samples_pre_episode = samples_per_episode

        self.replay_buffer = ReplayBuffer(stored_episodes, samples_per_episode, n, device=device)

        self.qnetwork = QNetwork(self.input_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Define Warmup and Decay
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps))
        self.exponential_scheduler = ExponentialLR(self.optimizer, learning_rate_decay)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[self.warmup_scheduler, self.exponential_scheduler], milestones=[warmup_steps])

    def remember(self, state, action, reward, target):
        self.replay_buffer.add(state, action, reward, target)

    def act(self, state, setpoint):
        with torch.no_grad():
            past_states, past_actions = self.replay_buffer.getInput()

            input_vector = torch.cat([
                past_states,
                state,
                past_actions,
                setpoint], dim=0)

            act_values = self.qnetwork(input_vector)

            result = 0
            if np.random.rand() <= self.epsilon:
                result = torch.randint(self.action_size, (1,), device=device)
            else:
                result = torch.argmax(act_values)

            return result

    def replay(self, batch_size):
        with torch.no_grad():
            if self.replay_buffer.size < self.replay_buffer.stored_epsiodes:
                return

            states, actions, rewards, targets = self.replay_buffer.random_sample(batch_size)

            current_inputs = torch.cat([
                states[:, : self.n +1],
                actions[:, : self.n],
                targets[:]], dim=1)

            predicted = self.qnetwork(current_inputs)

            next_inputs = torch.cat([
                states[:, 1 : self.n +2],
                actions[:, 1: self.n + 1],
                targets[:]], dim=1)

            next_predicted = self.qnetwork(next_inputs)
            target_f = predicted.clone()

            for k in range(batch_size):
                action = actions[k, self.n:self.n+1]
                target = (1-self.alpha) * predicted[k, action] + self.alpha * (rewards[k] + self.gamma * torch.max(next_predicted[k, :]))
                target_f[k, action] = target

        self.optimizer.zero_grad()
        loss = self.criterion(target_f, self.qnetwork(current_inputs))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.qnetwork.load_state_dict(torch.load(name, weights_only=True))

    def save(self, name):
        torch.save(self.qnetwork.state_dict(), name)

def entropy(probs):
    # Ensure the probabilities are not 0.0
    probs = torch.clamp(probs, min=1e-20)
    return -torch.sum(probs * torch.log(probs))

# Here the discrete version is modified into a Soft Q-Learning algorithm.
# A probability for each action is calculated by softmax(q_values / temperature).
# There is an analogy to the boltzmann gibbs distribution. The Q-Values can be seen as as energy levels
# of the states and the probabilies of a particle beeing in a certain state, corresponds to the probability
# of a certain action.
class QLearningAgentBoltzmann(QLearningAgent):
    def __init__(self, action_size, gamma,
                        temperature, temperature_decay, temperature_min,
                        warmup_steps, learning_rate, learning_rate_decay,
                        stored_episodes, samples_per_episode,
                        n=100, alpha=0.1):
        super().__init__(action_size=action_size, n=n, alpha=alpha, gamma=gamma,
                        epsilon=0.0, epsilon_decay=0.0, epsilon_min=0.0,
                        learning_rate=learning_rate, warmup_steps=warmup_steps, learning_rate_decay=learning_rate_decay,
                        samples_per_episode=samples_per_episode, stored_episodes=stored_episodes)
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

    def act(self, state, setpoint):
        with torch.no_grad():
            past_states, past_actions = self.replay_buffer.getInput()

            input_vector = torch.cat([
                past_states,
                state,
                past_actions,
                setpoint], dim=0)

            act_values = self.qnetwork(input_vector)
            act_probabilities = torch.softmax(act_values / self.temperature, dim=0)
            result = torch.multinomial(act_probabilities, 1)
            self.entropy_reward = entropy(act_probabilities) * self.temperature
            return result

    def remember(self, state, action, reward, target):
        with torch.no_grad():
            self.replay_buffer.add(state, action, reward + self.entropy_reward, target)

    def replay(self, batch_size):
        if self.replay_buffer.size < self.replay_buffer.stored_epsiodes:
            return

        with torch.no_grad():
            states, actions, rewards, targets = self.replay_buffer.random_sample(batch_size)

            current_inputs = torch.cat([
                states[:, : self.n +1],
                actions[:, : self.n],
                targets[:]], dim=1)

            predicted = self.qnetwork(current_inputs)

            next_inputs = torch.cat([
                states[:, 1 : self.n +2],
                actions[:, 1: self.n + 1],
                targets[:]], dim=1)

            next_predicted = self.qnetwork(next_inputs)
            target_f = predicted.clone()

            for k in range(batch_size):
                action = actions[k, self.n:self.n+1]
                next_qvalues = next_predicted[k, :]
                act_probabilities = torch.softmax(next_qvalues / self.temperature, dim=0)
                weighted_qvalues = next_qvalues * act_probabilities
                weighted_sum = weighted_qvalues.sum()

                target = (1-self.alpha) * predicted[k, action] + self.alpha * (rewards[k] + self.gamma * weighted_sum)
                target_f[k, action] = target

        self.optimizer.zero_grad()
        loss = self.criterion(target_f, self.qnetwork(current_inputs))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

class QLearningAgentContinuous:
    def __init__(self, action_search_batch=64, n=100, learning_rate=0.0001, alpha=0.1, gamma=0.95, average_weight=0.5,
                        epsilon=0.5, epsilon_decay=0.9997, epsilon_min=0.05,
                        warmup_steps=1000, learning_rate_decay=0.999988):
        self.action_search_batch = action_search_batch
        self.n = n  # Number of previous states and actions to consider
        self.input_size = n * 2 + 3  # Last n actions and states + current state, current action and setpoint
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.average_weight = average_weight

        # Initialize Q-Network and move it to the GPU if available
        self.qnetwork_1 = QNetwork(self.input_size, 1).to(device)
        self.optimizer_1 = optim.Adam(self.qnetwork_1.parameters(), lr=learning_rate)
        self.criterion_1 = nn.MSELoss()
        # Define Warmup and Decay
        self.warmup_scheduler_1 = LambdaLR(self.optimizer_1, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps))
        self.exponential_scheduler_1 = ExponentialLR(self.optimizer_1, learning_rate_decay)
        self.scheduler_1 = SequentialLR(self.optimizer_1, schedulers=[self.warmup_scheduler_1, self.exponential_scheduler_1], milestones=[warmup_steps])

        # Initialize Q-Network and move it to the GPU if available
        self.qnetwork_2 = QNetwork(self.input_size, 1).to(device)
        self.optimizer_2 = optim.Adam(self.qnetwork_2.parameters(), lr=learning_rate)
        self.criterion_2 = nn.MSELoss()
        # Define Warmup and Decay
        self.warmup_scheduler_2 = LambdaLR(self.optimizer_2, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps))
        self.exponential_scheduler_2 = ExponentialLR(self.optimizer_2, learning_rate_decay)
        self.scheduler_2 = SequentialLR(self.optimizer_2, schedulers=[self.warmup_scheduler_2, self.exponential_scheduler_2], milestones=[warmup_steps])

        # Replay memory
        self.memory = deque(maxlen=40000)
        # Store the last `n` states and actions
        self.past_states = deque([0.0]*(n+1), maxlen=(n+1))
        self.past_actions = deque([0.0]*(n+1), maxlen=(n+1))

    def remember(self, action, reward, next_state, target):
        self.memory.append((list(self.past_states), list(self.past_actions), action, next_state, reward, target))

    def build_input(self, state, action, setpoint, past_states, past_actions):
        return torch.cat([past_states, state, past_actions, action, setpoint], dim=0)

    def searchAction(self, state, setpoint, past_states, past_actions):
        inputs = torch.empty((self.action_search_batch, self.input_size), device=device)
        actions = torch.rand([self.action_search_batch], device=device)
        for i in range(self.action_search_batch):
            inputs[i] = self.build_input(state, actions[i:i+1], setpoint, past_states, past_actions)

        act_values_1 = self.qnetwork_1(inputs)
        act_values_2 = self.qnetwork_2(inputs)
        act_values = torch.mean(torch.stack([act_values_1, act_values_2]), dim=0)
        return actions[torch.argmax(act_values)]


    def act(self, state, setpoint):
        result = 0
        if np.random.rand() <= self.epsilon:
            result = np.random.uniform(0.0, 1.0)
        else:
            past_states = torch.tensor( list(self.past_states)[-self.n :], device=device)
            past_actions = torch.tensor( list(self.past_actions)[-self.n :], device=device)
            result = self.searchAction(state, setpoint, past_states, past_actions)

        self.past_actions.append(result)
        self.past_states.append(state)

        return result

    def replay(self, batch_size):
        if len(self.memory) < 13000:
            return

        minibatch = random.sample(self.memory, batch_size)
        current_inputs = torch.empty((batch_size, self.input_size))
        actions = torch.empty((batch_size, 1))
        rewards = torch.empty((batch_size, 1))
        next_inputs = torch.empty((batch_size, self.input_size))
        i = 0
        for past_states, past_actions, action, next_state, reward, target in minibatch:
            current_inputs[i] = self.build_input(past_states[-1], action, target, past_states[:self.n], past_actions[:self.n])
            next_action = self.searchAction(next_state, target, past_states[-self.n:], past_actions[-self.n:])

            next_inputs[i] = self.build_input(next_state, next_action, target, past_states[-self.n:], past_actions[-self.n:])

            actions[i] = action
            rewards[i] = reward
            i += 1

        current_inputs = current_inputs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_inputs = next_inputs.to(device)

        predicted_1 = self.qnetwork_1(current_inputs)
        next_predicted_1 = self.qnetwork_1(next_inputs)
        predicted_2 = self.qnetwork_2(current_inputs)
        next_predicted_2 = self.qnetwork_2(next_inputs)

        next_average_1 = (1 - self.average_weight) * next_predicted_1 + self.average_weight * next_predicted_2
        next_average_2 = (1 - self.average_weight) * next_predicted_2 + self.average_weight * next_predicted_1

        target_1 = (1-self.alpha) * predicted_1 + self.alpha * (rewards + self.gamma * next_average_1)
        target_2 = (1-self.alpha) * predicted_2 + self.alpha * (rewards + self.gamma * next_average_2)

        self.optimizer_1.zero_grad()
        loss_1 = self.criterion_1(target_1, self.qnetwork_1(current_inputs))
        loss_1.backward(retain_graph=True)
        self.optimizer_1.step()
        self.scheduler_1.step()

        self.optimizer_2.zero_grad()
        loss_2 = self.criterion_2(target_2, self.qnetwork_2(current_inputs))
        loss_2.backward()
        self.optimizer_2.step()
        self.scheduler_2.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.qnetwork_1.load_state_dict(torch.load("1_" + name, weights_only=True))
        self.qnetwork_2.load_state_dict(torch.load("2_" + name, weights_only=True))

    def save(self, name):
        torch.save(self.qnetwork_1.state_dict(), "1_" + name)
        torch.save(self.qnetwork_2.state_dict(), "2_" + name)

lower_bound = torch.tensor([0.0], device=device)
upper_bound = torch.tensor([1.0], device=device)
class QLearningAgentSoft:
    def __init__(self, action_search_batch=64, n=100, learning_rate=0.0001, alpha=0.1, gamma=0.95, average_weight=0.5,
                        temperature=0.1,
                        warmup_steps=1000, learning_rate_decay=0.9998):
        self.action_search_batch = action_search_batch
        self.n = n  # Number of previous states and actions to consider
        self.input_size = n * 2 + 4  # Last n actions and states + current state, current action distribution and setpoint
        self.gamma = gamma  # Discount rate
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.average_weight = average_weight
        self.temperature = temperature

        # Initialize Q-Network and move it to the GPU if available
        self.qnetwork_1 = QNetwork(self.input_size, 1).to(device)
        self.optimizer_1 = optim.Adam(self.qnetwork_1.parameters(), lr=learning_rate)
        self.criterion_1 = nn.MSELoss()
        # Define Warmup and Decay
        self.warmup_scheduler_1 = LambdaLR(self.optimizer_1, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps))
        self.exponential_scheduler_1 = ExponentialLR(self.optimizer_1, learning_rate_decay)
        self.scheduler_1 = SequentialLR(self.optimizer_1, schedulers=[self.warmup_scheduler_1, self.exponential_scheduler_1], milestones=[warmup_steps])

        # Initialize Q-Network and move it to the GPU if available
        self.qnetwork_2 = QNetwork(self.input_size, 1).to(device)
        self.optimizer_2 = optim.Adam(self.qnetwork_2.parameters(), lr=learning_rate)
        self.criterion_2 = nn.MSELoss()
        # Define Warmup and Decay
        self.warmup_scheduler_2 = LambdaLR(self.optimizer_2, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps))
        self.exponential_scheduler_2 = ExponentialLR(self.optimizer_2, learning_rate_decay)
        self.scheduler_2 = SequentialLR(self.optimizer_2, schedulers=[self.warmup_scheduler_2, self.exponential_scheduler_2], milestones=[warmup_steps])

        # Replay memory
        self.memory = deque(maxlen=8000)
        # Store the last `n` states and actions
        self.past_states = deque([torch.tensor(0.0, device=device)]*(n+1), maxlen=(n+1))
        self.past_actions = deque([torch.tensor(0.0, device=device)]*(n+1), maxlen=(n+1))
        self.past_action_distribution = None
        self.past_entropy = None

    def remember(self, reward, next_state, target):
        self.memory.append((
            torch.tensor(list(self.past_states), device=device),
            torch.tensor(list(self.past_actions), device=device),
            self.past_action_distribution,
            self.past_entropy,
            next_state,
            reward,
            target))

    def build_input(self, state, action_distribution, setpoint, past_states, past_actions):
        input_vector = torch.cat((
            past_states,
            state,
            past_actions,
            action_distribution,
            setpoint), dim=0)
        return input_vector

    def searchAction(self, state, setpoint, past_states, past_actions):
        inputs = torch.empty((self.action_search_batch, self.input_size), device=device)
        actions = torch.rand((self.action_search_batch, 2), device=device)
        actions[(... , 1)] = actions[(... , 1)]**4 + 0.001 # avoid 0 standart deviation. more samples with low values
        for i in range(self.action_search_batch):
            inputs[i] = self.build_input(state, actions[i, :], setpoint, past_states, past_actions)

        act_values_1 = self.qnetwork_1(inputs)
        act_values_2 = self.qnetwork_2(inputs)
        act_values = torch.mean(torch.stack([act_values_1, act_values_2]), dim=0)
        return actions[torch.argmax(act_values)]

    def act(self, state, setpoint):
        past_states = torch.tensor(list(self.past_states)[-self.n :], device=device)
        past_actions = torch.tensor(list(self.past_actions)[-self.n :], device=device)
        action_distribution = self.searchAction(state, setpoint, past_states, past_actions)
        #random sample from probability distribution
        u = action_distribution[(... , 0)]
        s = action_distribution[(... , 1)]
        action = truncated_normal_inverse_cmd(u, s, lower_bound, upper_bound, torch.rand(state.size(), device=device))

        self.past_action_distribution = action_distribution
        self.past_entropy = truncated_normal_entropy(action_distribution[0], action_distribution[1], lower_bound, upper_bound)
        self.past_states.append(state)
        self.past_actions.append(action)
        return action, u, s

    def replay(self, batch_size):
        if len(self.memory) < 7000:
            return

        minibatch = random.sample(self.memory, batch_size)
        current_inputs = torch.empty((batch_size, self.input_size), device=device)
        rewards = torch.empty((batch_size, 1), device=device)
        next_inputs = torch.empty((batch_size, self.input_size), device=device)
        i = 0
        for past_states, past_actions, action_distribution, entropy, next_state, reward, target in minibatch:
            current_inputs[i] = self.build_input(past_states[-1:], action_distribution, target, past_states[:self.n], past_actions[:self.n])
            next_action = self.searchAction(next_state, target, past_states[-self.n:], past_actions[-self.n:])
            next_inputs[i] = self.build_input(next_state, next_action, target, past_states[-self.n:], past_actions[-self.n:])
            rewards[i] = reward + self.temperature * entropy # reward for high entropy
            i += 1

        predicted_1 = self.qnetwork_1(current_inputs)
        next_predicted_1 = self.qnetwork_1(next_inputs)
        predicted_2 = self.qnetwork_2(current_inputs)
        next_predicted_2 = self.qnetwork_2(next_inputs)

        next_average_1 = (1 - self.average_weight) * next_predicted_1 + self.average_weight * next_predicted_2
        next_average_2 = (1 - self.average_weight) * next_predicted_2 + self.average_weight * next_predicted_1

        target_1 = (1-self.alpha) * predicted_1 + self.alpha * (rewards + self.gamma * next_average_1)
        target_2 = (1-self.alpha) * predicted_2 + self.alpha * (rewards + self.gamma * next_average_2)

        if np.random.uniform(0.0, 1.0) > 0.5:
            self.optimizer_1.zero_grad()
            loss_1 = self.criterion_1(target_1, self.qnetwork_1(current_inputs))
            loss_1.backward()
            self.optimizer_1.step()
            self.scheduler_1.step()
        else:
            self.optimizer_2.zero_grad()
            loss_2 = self.criterion_2(target_2, self.qnetwork_2(current_inputs))
            loss_2.backward()
            self.optimizer_2.step()
            self.scheduler_2.step()

    def load(self, name):
        self.qnetwork_1.load_state_dict(torch.load("1_" + name, weights_only=True))
        self.qnetwork_2.load_state_dict(torch.load("2_" + name, weights_only=True))

    def save(self, name):
        torch.save(self.qnetwork_1.state_dict(), "1_" + name)
        torch.save(self.qnetwork_2.state_dict(), "2_" + name)

