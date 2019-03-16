from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random


def get_arg_names(function_id):
    action = actions.FUNCTIONS[function_id]
    return [i.name for i in action.args]


def test_get_arg_names():
    move_screen = 1
    attack_minimap = 13
    print(get_arg_names(attack_minimap))


if __name__ == '__main__':
    test_get_arg_names()
