from pysc2.lib import actions
import numpy as np


def get_arg_ids(function_ids):
    """Returns all argument ids of given function ids

        Args:
            function_ids (List): list of function ids

        Returns:
            arg_ids (array_like): array of shape (len(function_ids), MAX_ARGS)
    """
    MAX_ARGS = 3
    args_ids = []
    for function_id in function_ids:
        action = actions.FUNCTIONS[function_id]
        arg_ids = [action.args[i].id + 1 for i in range(len(action.args))]
        arg_names = [action.args[i].name for i in range(len(action.args))]
        #print([(arg_name, arg_id) for arg_name, arg_id in zip(arg_ids, arg_names)])
        args_ids.append(np.concatenate((np.array(arg_ids), np.zeros(MAX_ARGS - len(arg_ids))))[np.newaxis, :])
    return np.concatenate(args_ids, axis=0).astype(int)


def test_get_arg_names():
    no_op = 0
    move_screen = 1
    select_rect = 3
    attack_minimap = 13
    print(get_arg_ids([no_op, move_screen, select_rect, attack_minimap]))


if __name__ == '__main__':
    test_get_arg_names()
