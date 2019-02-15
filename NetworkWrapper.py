class NetworkWrapper:
    """Wraps networks from PyTorch, Keras, etc. in common interface for
    logging purposes

    TODO: Extend agent use of this network object to allow for agent
          implementations to be independent of network type. Need to route
          optimizer through this as well.

    Params:
        network (PyTorch nn.Module, Keras ___): network model
    """
    def __init__(self, network):
        self.base_network = network
        self.layers = self.get_layers(self.base_network)

    def get_layers(self, network):
        pass


class Layer:
    def __init__(self):
        self.type = None
        self.nodes = None
