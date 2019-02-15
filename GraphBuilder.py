class GraphBuilder:
    """Adds nodes and edges to given graph from given network. Allows
    different visualizations of graphs to come from passing in different
    graph objects.

    Params:
        config (Config): Contains graph object that GraphBuilder will build.
        Expects graph.node and graph.edge

        network (NetworkWrapper): Wrapped network, GraphBuilder expects
        network.layers to be list of Layer objects that specify type and
        connections
    """
    def __init__(self, config, network):
        self.graph = config.graph
        self.network = network

    def build_graph(self):
        pass
