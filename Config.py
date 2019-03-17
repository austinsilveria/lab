class Config:
    def __init__(self,
                 seed=None,
                 state_size=None,
                 action_size=None,
                 model_fn=None,
                 buffer_size=None,
                 batch_size=None,
                 update_every=None,
                 gamma=None,
                 eps=None,
                 tau=None):
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        self.build_model = model_fn
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
