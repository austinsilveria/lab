import Networks as nt
import numpy as np
import ActionFuncs as af
from pysc2.lib import actions


class SC2NetWrapper:
    def __init__(self, keras_model):
        self.model = keras_model
        self.model_out = {'action_id': 0,
                          'screen': 1,
                          'minimap': 2,
                          'screen2': 3,
                          'queued': 4,
                          'control_group_act': 5,
                          'control_group_id': 6,
                          'select_point_act': 7,
                          'select_add': 8,
                          'select_unit_act': 9,
                          'select_unit_id': 10,
                          'select_worker': 11,
                          'build_queue': 12,
                          'unload_id': 13}

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def predict_actions(self, state, available_actions):
        """Returns best action and arg indices from network output

            Args:
                state (array_like): preprocessed state observation
                available_actions (array_like): indices of available actions
                                                for current state

            Returns:
                action_id (int): index of best action (also action_id)
                args (list[tuple[int]]): indices of best arguments for action
        """
        network_out = self.model.predict(state)
        #print('Network out:', [i.shape for i in network_out])
        #print('Network out length', len(network_out))
        action_values = network_out[self.model_out['action_id']]
        #print('Action values:', action_values)
        masked_action_values = np.zeros_like(action_values)
        #print('Masked action values:', masked_action_values.shape)
        masked_action_values[:, available_actions] = action_values[:, available_actions]
        #print('Masked action values data:', masked_action_values)
        action_ids = list(np.argmax(masked_action_values, axis=1))
        #print('Action ids:', action_ids)
        arg_ids = af.get_arg_ids(action_ids)
        #print('Arg ids:', arg_ids)
        batch_ids = np.array(range(arg_ids.shape[0]))
        #print('Batch ids:', batch_ids)
        args = []
        for i in batch_ids:
            i_args = []
            arg_id_list = arg_ids[i]
            arg_id_list = arg_id_list[arg_id_list != 0]
            #print(arg_id_list)
            for arg in arg_id_list:
                batch_arg_out = network_out[arg][i]
                best_arg = np.unravel_index(np.argmax(batch_arg_out),
                                            batch_arg_out.shape)
                #print('Best arg:', best_arg)
                if len(best_arg) == 1:
                    best_arg = best_arg[0]
                i_args.append([arg, i, best_arg])
            #print('Batch:', i, ' args for action', action_ids[i], ':', i_args)
            args.append(i_args)
        #print('All batch args:', args)
        #args = [np.unravel_index(np.argmax(network_out[self.model_out[name]]),
        #                         network_out[self.model_out[name]].shape)
        #        for name in arg_names]
        #args = np.argmax(network_out, axis=1)
        #print('Max of indexed arguments:', args)
        return action_ids, args

    def build_actions(self, action_ids, args):
        #print('Length action_ids:', len(action_ids))
        #print('Length args:', len(args))
        #return [actions.FunctionCall(action_id, arg_set[2:])
        #        for action_id, arg_set in zip(action_ids, args)]
        action_set = []
        for action_id, arg_set in zip(action_ids, args):
            # arg_set comes in form: [arg_id, batch_size, argument]
            # we only want the argument, so arg_set[2]
            passed_args = []
            if action_id != 0:
                for arg in arg_set:
                    try:
                        length = len(arg[2])
                        passed_args.append(arg[2][:2])
                    except TypeError:
                        passed_args.append(arg[2])
            action_set.append(actions.FunctionCall(action_id, passed_args))
        return action_set

    def predict_value(self, state):
        """Returns network output value of all actions and args
        """
        network_out = self.model.predict(state)
        return network_out


def test_wrapper():
    model = nt.Networks().SC2FullConv()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy')
    wrapped = SC2NetWrapper(model)
    #non_spatial = np.random.rand(32, 64, 64, 3)
    #screen = np.random.rand(32, 64, 64, 17)
    #minimap = np.random.rand(32, 64, 64, 7)
    non_spatial = np.random.rand(1, 64, 64, 3)
    screen = np.random.rand(1, 64, 64, 17)
    minimap = np.random.rand(1, 64, 64, 7)
    avail = np.array([0, 1, 2, 3, 4])[np.newaxis, :]
    all_avail = np.repeat(avail, non_spatial.shape[0], axis=0)
    print('Available actions:', all_avail[0])
    print('Sample non-spatial data:', non_spatial.shape)
    print('Sample screen data:', screen.shape)
    print('Sample minimap data:', minimap.shape)
    q_values = wrapped.predict_value([non_spatial, screen, minimap])
    action_ids, args = wrapped.predict_actions([non_spatial, screen, minimap],
                                                all_avail)
    action_funcs = wrapped.build_actions(action_ids, args)
    #print('Q-Values:', q_values)
    #print('Action ids:', action_ids)
    print('Action functions:', action_funcs)


if __name__ == '__main__':
    test_wrapper()
