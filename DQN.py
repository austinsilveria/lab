import random
import numpy as np

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.seed = config.seed
        if self.seed is not None:
            random.seed(self.seed)

        self.state_size = config.state_size
        self.action_size = config.action_size

        self.batch_size = config.batch_size
        self.update_every = config.update_every
        self.gamma = config.gamma
        self.eps = config.eps
        self.tau = config.tau

        # Q-Network
        self.qnetwork_local = config.model
        self.qnetwork_target = config.model

        # Replay memory
        self.buffer_size = config.buffer_size
        self.memory = ReplayBuffer(config.action_size, self.buffer_size,
                                   self.batch_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted actions (for next states) from local model
        next_local_actions = self.qnetwork_local(next_states).amax(1)[
            1]
        print('Next local actions:', next_local_actions.shape)
        # Evaluate the max predicted actions from the local model on the target model
        # based on Double DQN
        Q_targets_next_values = self.qnetwork_target(
            next_states).take(next_local_actions, axis=1)
        print('Q targets next vales:', Q_targets_next_values.shape)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next_values * (1 - dones))

        # Get expected Q values from local
        Q_expected = self.qnetwork_local(states).take(actions, axis=1)

        self.model.train_from_batch(Q_expected, Q_targets)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # TODO Store next target_params in np array to set all at once
        print('Weight shape:', local_model.get_weights().shape)
        for target_param, local_param in zip(target_model.get_weights(),
                                             local_model.get_weights()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience objects."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Sample a batch of experiences from memory based on TD Error priority.
           Return indexes of sampled experiences in order to update their
           priorities after learning from them.
        """
        # TODO Set to all np arrays
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(
            device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(
                np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
