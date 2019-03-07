from keras import Input, layers
from keras.models import Model


class Networks:
    def __init__(self):
        pass

    def build_full_conv(self):
        reward_input = Input(shape=(1,), dtype='int32', name='reward')
        non_spatial_input = Input(shape=(8492,),
                                  dtype='float64', name='non_spatial')
        spatial_input = Input(shape=(17, 84, 84, 1),
                              dtype='int32', name='spatial')
        minimap_input = Input(shape=(7, 64, 64, 1),
                              dtype='int32', name='minimap')
