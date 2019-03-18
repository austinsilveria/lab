import random
import numpy as np
from collections import namedtuple, deque

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
from Config import Config
from Networks import Networks
from SC2NetWrapper import SC2NetWrapper
from Preprocessor import Preprocessor


SIZE = 64


class DQN:
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

        self.batch_size = config.batch_size
        self.update_every = config.update_every
        self.max_steps = config.max_steps
        self.gamma = config.gamma
        self.tau = config.tau

        # Eps
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay

        self.eps = self.eps_start

        # Q-Network
        self.qnetwork_local = config.model_wrapper(config.build_model())
        self.qnetwork_target = config.model_wrapper(config.build_model())

        # Replay memory
        self.buffer_size = config.buffer_size
        self.memory = ReplayBuffer(self.buffer_size,
                                   self.batch_size,
                                   self.seed)

        # Preprocessor
        self.preprocessor = config.preprocessor

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 1

    def step(self, state, action, reward, next_state, next_avail, done):
        # Save experience in replay memory
    #def add(self, state, action_fn, reward, next_state, next_avail, done):
        self.memory.add(state, action, reward, next_state, next_avail, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, available_actions):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        action_ids, args = self.qnetwork_local.predict_actions(state,
                                                               available_actions)
        action_set = self.qnetwork_local.build_actions(action_ids, args)

        # timestep reset to 0 after each episode
        if self.t_step == 0 or self.t_step >= self.max_steps:
            self.eps = max(self.eps_end, self.eps*self.eps_decay)

        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return action_set[0]
        else:
            return self.qnetwork_local.random_action(available_actions)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, action_fns, rewards, next_states, next_avail, dones = experiences

        # Get max predicted actions (for next states) from local model
        next_local_actions = self.qnetwork_local.predict_actions(next_states,
                                                                 next_avail)
        #print('Next local actions:', next_local_action_ids.shape)
        # Evaluate the max predicted actions from the local model on the target model
        # based on Double DQN
        Q_targets_next_values = self.qnetwork_target.predict_action_value(next_states,
                                                                          *next_local_actions)
        #print('Q targets next vales:', Q_targets_next_values.shape)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next_values * (1 - dones))

        target_output = self.qnetwork_target.generate_target(states,
                                                             action_fns,
                                                             Q_targets)

        # Get expected Q values from local
        #Q_expected = self.qnetwork_local.predict_actionfn_value(states,
        #                                                        action_fns)

        self.qnetwork_local.model.train_on_batch(states, target_output)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (Keras model): weights will be copied from
            target_model (Keras model): weights will be copied to
            tau (float): interpolation parameter
        """
        #print('Weight shape:', local_model.get_weights().shape)
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()
        new_weights = []
        for local, target in zip(local_weights, target_weights):
            new_weights.append(tau * local + (1.0 - tau) * target)
        target_model.set_weights(new_weights)


class ReplayBuffer:
    """Fixed-size buffer to store experience objects."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action_fn",
                                                  "reward", "next_state",
                                                  "next_avail", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action_fn, reward, next_state, next_avail, done):
        """Add a new experience to memory."""
        #print('State shape:', np.array(state)[np.newaxis, :].shape)
        e = self.experience(state, action_fn, reward,
                            next_state, next_avail, done)
        self.memory.append(e)

    def sample(self):
        """Sample a batch of experiences from memory randomly.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [np.vstack([e.state[0] for e in experiences if e is not None]),
                  np.vstack([e.state[1] for e in experiences if e is not None]),
                  np.vstack([e.state[2] for e in experiences if e is not None])]
        action_fns = np.vstack([e.action_fn for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = [np.vstack([e.next_state[0] for e in experiences if e is not None]),
                       np.vstack([e.next_state[1] for e in experiences if e is not None]),
                       np.vstack([e.next_state[2] for e in experiences if e is not None])]
        next_avail = np.vstack([e.next_avail for e in experiences if e is not None])
        dones = np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, action_fns, rewards, next_states, next_avail, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class AgentWrapper(base_agent.BaseAgent):
    def __init__(self, agent):
        super(AgentWrapper, self).__init__()
        self.agent = agent

    def step(self, obs):
        available_actions = obs.observation.available_actions
        #print('Available actions:', available_actions)
        processed = self.agent.preprocessor(obs)
        state = processed[0]
        #print('State:', [i.shape for i in state])
        avail = self.agent.preprocessor.process_avail_actions(available_actions)
        return self.agent.act(state, avail)

    def reflect(self, obs, action, next_obs):
        processed = self.agent.preprocessor(obs)
        state = processed[0]

        processed_next = self.agent.preprocessor(next_obs)
        next_state = processed_next[0]
        reward = processed_next[1]
        done = processed_next[2]
        next_available_actions = next_obs.observation.available_actions
        next_avail = self.agent.preprocessor.process_avail_actions(next_available_actions)

        self.agent.step(state, action, reward, next_state, next_avail, done)


def main(unused_argv):
    model_fn = Networks().SC2FullConv
    wrapper = SC2NetWrapper
    config = Config(seed=0,
                    model_fn=model_fn,
                    model_wrapper=wrapper,
                    buffer_size=int(1e5),
                    batch_size=64,
                    update_every=4,
                    max_steps=1e5,
                    gamma=0.99,
                    eps_start=1.0,
                    eps_end=0.01,
                    eps_decay=0.995,
                    tau=1e-3)
    agent = AgentWrapper(DQN(config))
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=SIZE,
                                                               minimap=SIZE),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                preprocessor = Preprocessor(env.observation_spec(), env.action_spec())
                agent.agent.preprocessor = preprocessor
                timesteps = env.reset()
                agent.reset()

                while True:
                    obs = timesteps[0]
                    step_actions = [agent.step(obs)]
                    if timesteps[0].last():
                        break
                    print('Sending action:', step_actions)
                    timesteps = env.step(step_actions)
                    next_obs = timesteps[0]
                    agent.reflect(obs, step_actions[0], next_obs)


    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)

