from AgentBase import AgentBase


class PPO(AgentBase):
    """PPO agent learning from interactions with the environment"""
    def __init__(self, config):
        super(PPO, self).__init__(config)
        self.env = config.env
        self.network = config.network
        self.optimizer = config.optimizer
        # Config settings
        # .
        # .
        # .

    def act(self):
        pass

    def step(self):
        pass
