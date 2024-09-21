import torch
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, stored_episodes, samples_per_epsiode, device):
        self.stored_epsiodes = stored_episodes
        self.samples_per_epsisode = samples_per_epsiode
        self.position_episode = 0
        self.position_sample = 0
        self.size = 0
        self.device=device

        self.states = torch.zeros((stored_episodes, samples_per_epsiode), device=device)
        self.actions = torch.zeros((stored_episodes, samples_per_epsiode), device=device, dtype=torch.int)
        self.rewards = torch.zeros((stored_episodes, samples_per_epsiode), device=device)
        self.targets = torch.zeros((stored_episodes, samples_per_epsiode), device=device)

    def add(self, state, action, reward, target):
        self.states[self.position_episode, self.position_sample] = state
        self.actions[self.position_episode, self.position_sample] = action
        self.rewards[self.position_episode, self.position_sample] = reward
        self.targets[self.position_episode, self.position_sample] = target

        self.position_sample += 1
        if self.position_sample == self.samples_per_epsisode:
            self.position_sample = 0
            self.position_episode = (self.position_episode + 1) % self.stored_epsiodes

            if self.size < self.stored_epsiodes:
                self.size += 1

    def getInput(self, seq_length):
        episode = self.position_episode
        sample = self.position_sample
        # when not enough past values. pad zeros as initial state
        if sample == 0:
            return torch.zeros([seq_length], device=self.device), torch.zeros([seq_length], device=self.device)
        if sample < seq_length:
            diff = seq_length - sample
            return F.pad(self.states[episode][0:sample], (diff, 0), "constant", 0), F.pad(self.actions[episode][0:sample], (diff, 0), "constant", 0)
        else:
            start = sample - seq_length
            return self.states[episode][start : sample], self.actions[episode][start : sample]

    def random_sample(self, batch_size, seq_length):
        episode_indices = torch.randint(self.stored_epsiodes, (batch_size,), device=self.device)
        sample_indices = torch.randint(self.samples_per_epsisode - seq_length - 1, (batch_size,), device=self.device) # -1 cause we need one future state

        batch_states = torch.zeros((batch_size, seq_length + 1), device=self.device) # +1 for future state
        batch_actions = torch.zeros((batch_size, seq_length), device=self.device, dtype=torch.int)
        batch_rewards = torch.zeros((batch_size, 1), device=self.device)
        batch_targets = torch.zeros((batch_size, 1), device=self.device)

        for i in range(batch_size):
            idx_sample = sample_indices[i]
            idx_episode = episode_indices[i]
            batch_states[i, :] = self.states[idx_episode, idx_sample : idx_sample+seq_length + 1] # one more element future action
            batch_actions[i, :] = self.actions[idx_episode, idx_sample : idx_sample+seq_length]
            batch_rewards[i, 0] = self.rewards[idx_episode, idx_sample+seq_length - 1] # reward corresponding to last action
            batch_targets[i, 0] = self.targets[idx_episode, idx_sample+seq_length - 1]

        return batch_states, batch_actions, batch_rewards, batch_targets

    def __len__(self):
        return self.size
