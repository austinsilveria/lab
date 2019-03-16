from keras import Input, layers, backend
from keras.models import Model


class Networks:
    def __init__(self):
        pass

    def FullConvBase(self, input):
        conv1 = layers.Conv2D(16, 5, input_shape=input.shape, padding='same',
                              activation='relu', name='conv1')(input)
        conv2 = layers.Conv2D(32, 3, padding='same',
                              activation='relu', name='conv2')(conv1)
        return conv2

    def SC2FullConv(self):
        non_spatial_input = Input(shape=(64, 64, 3),
                                  dtype='float32', name='non_spatial')
        screen_input = Input(shape=(64, 64, 17),
                             dtype='float32', name='screen')
        minimap_input = Input(shape=(64, 64, 7),
                              dtype='float32', name='minimap')
        screen_mid = self.FullConvBase(screen_input)
        minimap_mid = self.FullConvBase(minimap_input)
        # State representation shape: (1, 64, 64, 27)
        print('screen_mid:', screen_mid.shape)
        print('minimap_mid:', minimap_mid.shape)
        print('non_spatial_input:', non_spatial_input.shape)
        state_rep = backend.concatenate((screen_mid,
                                        minimap_mid,
                                        non_spatial_input))
        print('State rep:', state_rep.shape)
        screen_out = layers.Conv2D(1, 1, padding='same',
                                   activation='softmax',
                                   name='screen_out')(state_rep)
        screen2_out = layers.Conv2D(1, 1, padding='same',
                                    activation='softmax',
                                    name='screen2_out')(state_rep)
        minimap_out = layers.Conv2D(1, 1, padding='same',
                                    activation='softmax',
                                    name='minimap_out')(state_rep)
        return Model(inputs=[non_spatial_input, screen_input, minimap_input],
                     outputs=[screen_out, screen2_out, minimap_out])

