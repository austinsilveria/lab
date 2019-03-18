from keras import Input, layers, backend
from keras.models import Model


class Networks:
    def __init__(self):
        pass

    def FullConvBase(self, input):
        conv1 = layers.Conv2D(16, 5, input_shape=input.shape, padding='same',
                              activation='relu', name=input.name[0]+'conv1')(input)
        conv2 = layers.Conv2D(32, 3, padding='same',
                              activation='relu', name=input.name[0]+'conv2')(conv1)
        return conv2

    def SC2FullConv(self):
        # Input
        non_spatial_input = Input(shape=(64, 64, 3),
                                  dtype='float32', name='non_spatial')
        screen_input = Input(shape=(64, 64, 17),
                             dtype='float32', name='screen')
        minimap_input = Input(shape=(64, 64, 7),
                              dtype='float32', name='minimap')

        # Spatial mid
        screen_mid = self.FullConvBase(screen_input)
        minimap_mid = self.FullConvBase(minimap_input)

        # State representation shape: (1, 64, 64, 67)
        #     32 * 2 filters + 3 non-spatial channels
        print('screen_mid:', screen_mid.shape)
        print('minimap_mid:', minimap_mid.shape)
        print('non_spatial_input:', non_spatial_input.shape)
        state_rep = layers.Concatenate(name='state_rep')([screen_mid,
                                                          minimap_mid,
                                                          non_spatial_input])
        # TODO: How does this work when LSTM requires time dimension as well?
        # state_rep = layers.ConvLSTM2D(32, 3, activation='tanh',
        #                              name='state_rep')(state_rep)
        print('State rep:', state_rep.shape)

        # Non-spatial mid
        flattened = layers.Flatten()(state_rep)
        non_spatial_state = layers.Dense(256, activation='relu',
                                         name='non_spatial_state')(flattened)

        # Spatial out
        screen_out = layers.Conv2D(1, 1, padding='same',
                                   activation='softmax',
                                   name='screen_out')(state_rep)
        minimap_out = layers.Conv2D(1, 1, padding='same',
                                    activation='softmax',
                                    name='minimap_out')(state_rep)
        screen2_out = layers.Conv2D(1, 1, padding='same',
                                    activation='softmax',
                                    name='screen2_out')(state_rep)

        # Non-spatial out
        action_id_out = layers.Dense(549, activation='linear',
                                     name='action_id_out')(non_spatial_state)
        queued_out = layers.Dense(2, activation='linear', 
                                  name='queued_out')(non_spatial_state)
        control_group_act_out = layers.Dense(5, activation='linear', 
                                             name='contol_group_act_out')(non_spatial_state)
        control_group_id_out = layers.Dense(10, activation='linear', 
                                            name='contol_group_id_out')(non_spatial_state)
        select_point_act_out = layers.Dense(4, activation='linear', 
                                            name='select_point_act_out')(non_spatial_state)
        select_add_out = layers.Dense(2, activation='linear', 
                                      name='select_add_out')(non_spatial_state)
        select_unit_act_out = layers.Dense(4, activation='linear', 
                                           name='select_unit_act_out')(non_spatial_state)
        select_unit_id_out = layers.Dense(500, activation='linear', 
                                          name='select_unit_id_out')(non_spatial_state)
        select_worker_out = layers.Dense(4, activation='linear', 
                                         name='select_worker_out')(non_spatial_state)
        build_queue_id_out = layers.Dense(10, activation='linear', 
                                          name='build_queue_id_out')(non_spatial_state)
        unload_id_out = layers.Dense(500, activation='linear', 
                                     name='unload_id_out')(non_spatial_state)

        model = Model(inputs=[non_spatial_input, screen_input, minimap_input],
                      outputs=[action_id_out, screen_out, minimap_out,
                              screen2_out, queued_out, control_group_act_out,
                              control_group_id_out, select_point_act_out,
                              select_add_out, select_unit_act_out,
                              select_unit_id_out, select_worker_out,
                              build_queue_id_out, unload_id_out])
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])
        return model

