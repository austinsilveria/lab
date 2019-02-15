class AgentBase:
    def __init__(self, config):
        self.env = config.env
        self.network = config.network
        self.storage = config.storage

    def interact(self):
        """Interacts with the environment for a specified number of steps and
           populates storage with environment and agent information

           Params:
                *All specified in config*
                rollout (int): number of steps to interact with the env for
                storage (Storage): storage obj to store information
        """
        pass

    def act(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
