from keras import Input, layers, backend
from keras.models import Model


class Networks:
    def __init__(self):
        pass

    def FullConvBase(self, input):
        conv1 = layers.Conv2D(16, 5, activation='relu', name='conv1')(input)
        conv2 = layers.Conv2D(32, 3, activation='relu', name='conv2')(conv1)
        return conv2

    def SC2FullConv(self):
        reward_input = Input(shape=(1,), dtype='int32', name='reward')
        non_spatial_input = Input(shape=(1, 64, 64, 3),
                                  dtype='float64', name='non_spatial')
        screen_input = Input(shape=(1, 64, 64, 17),
                              dtype='int32', name='screen')
        minimap_input = Input(shape=(1, 64, 64, 7),
                              dtype='int32', name='minimap')
        screen_out = self.FullConvBase(screen_input)
        minimap_out = self.FullConvBase(minimap_input)
        # State representation shape: (1, 64, 64, 27)
        state_rep = backend.concatenate((screen_out,
                                        minimap_out,
                                        non_spatial_input))

