import Networks as nt
import numpy as np
import ActionFuncs as af
from pysc2.lib import actions
import random


class SC2NetWrapper:
    # TODO: Use better network output representation to allow for np parallel
    #       indexing
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
        self.last_output = None

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
        self.last_output = network_out
        #print('Network out:', [i.shape for i in network_out])
        #print('Network out length', len(network_out))
        action_values = network_out[self.model_out['action_id']]
        #print('Action values:', action_values)
        masked_action_values = np.zeros_like(action_values)
        #print('Available actions:', available_actions)
        #masked_action_values[:, available_actions] = action_values[:, available_actions]
        masked_action_values = np.where(available_actions > 0, action_values, 0)
        #print('Masked action values:', masked_action_values)
        all_action_ids = np.repeat(np.arange(available_actions.shape[1])[np.newaxis, :],
                                   available_actions.shape[0], axis=0)
        action_ids = list(np.argmax(masked_action_values, axis=1))
        #print('Best action ids:', action_ids)
        #print('Masked action values data:', masked_action_values)
        #action_ids = np.where(masked_action_values > 0, all_action_ids, 550)
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
            # arg comes in form: [arg_id, batch_id, argument]
            # we only want the argument, so arg[2]
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

    def predict_action_value(self, state, action_ids, args):
        network_out = self.model.predict(state)
        self.last_output = network_out
        value_sum = np.zeros((len(action_ids), 1))
        #for action_id, arg_set in zip(action_ids, args):
        for batch, tup in enumerate(zip(action_ids, args)):
            action_id = tup[0]
            arg_set = tup[1]
            value_sum[batch] += network_out[0][batch][action_id]
            for arg in arg_set:
                arg_id = arg[0]
                argument = arg[2]
                value_sum[batch] += network_out[arg_id][batch][argument]
        return value_sum

    def predict_value(self, state):
        """Returns network output value of all actions and args
        """
        network_out = self.model.predict(state)
        self.last_output = network_out
        return network_out

    def random_action(self, available_actions):
        if self.last_output is None:
            return actions.FunctionCall(0, [])
        #print('Available actions:', available_actions)
        choice = random.choice(available_actions)
        #print('Choice:', choice)
        arg_ids = af.get_arg_ids([choice])[0]
        arg_ids = arg_ids[arg_ids != 0]
        #print('Arg ids:', arg_ids)
        args = []
        for arg_id in arg_ids:
            last_output = self.last_output[arg_id]

            #print('Last output:', last_output.shape)
            if len(last_output.shape) == 4:
                last_output = last_output[0, :, :, 0]
            else:
                last_output = last_output[0, :]
            #print('New last output shape:', last_output.shape)
            rand_output = np.random.rand(*last_output.shape)
            best = np.unravel_index(np.argmax(rand_output), rand_output.shape)
            if len(best) == 1:
                best = best[0]
            #print('For argument:', arg_id, 'best output:', best)
            args.append(best)
        #print('All args:', args)
        action = actions.FunctionCall(choice, args)
        #print('Action:', action)
        return action




def test_wrapper():
    model = nt.Networks().SC2FullConv()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy')
    wrapped = SC2NetWrapper(model)
    #non_spatial = np.random.rand(32, 64, 64, 3)
    #screen = np.random.rand(32, 64, 64, 17)
    #minimap = np.random.rand(32, 64, 64, 7)
    action_size = 549
    non_spatial = np.random.rand(2, 64, 64, 3)
    screen = np.random.rand(2, 64, 64, 17)
    minimap = np.random.rand(2, 64, 64, 7)
    #avail1 = np.concatenate((avail1, np.repeat(-1, action_size - len(avail1))))
    avail1 = np.zeros(action_size)
    avail_actions_1 = np.array([0, 1, 2, 3, 4])
    avail1[avail_actions_1] = 1
    avail2 = np.zeros(action_size)
    avail_actions_2 = np.array([6, 7, 8, 1])
    avail2[avail_actions_2] = 1
    #avail2 = np.concatenate((avail2, np.repeat(-1, action_size - len(avail2))))
    #all_avail = np.repeat(avail, non_spatial.shape[0], axis=0)
    all_avail = np.vstack((avail1[np.newaxis, :], avail2[np.newaxis, :]))
    #print('Available actions:', all_avail.shape)
    #print('Sample non-spatial data:', non_spatial.shape)
    #print('Sample screen data:', screen.shape)
    #print('Sample minimap data:', minimap.shape)
    q_values = wrapped.predict_value([non_spatial, screen, minimap])
    action_ids, args = wrapped.predict_actions([non_spatial, screen, minimap],
                                                all_avail)
    action_funcs = wrapped.build_actions(action_ids, args)
    action_values = wrapped.predict_action_value([non_spatial, screen, minimap],
                                                 action_ids,
                                                 args)
    #print('Q-Values:', q_values)
    #print('Action ids:', action_ids)
    #print('Action functions:', action_funcs)
    #print('Action values:', action_values.shape)

    avail_random = np.array([0, 5, 20, 14])
    rand_action = wrapped.random_action(avail_random)
    print('Random action given avail:', avail_random, ':', rand_action)

if __name__ == '__main__':
    test_wrapper()
