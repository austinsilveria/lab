import numpy as np
import keras
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

from SC2Agent import ZergAgent


class Preprocessor:
    def __init__(self, obs, obs_spec, action_spec):
        self.obs = obs
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def full_input(self):
        reward = self.obs.reward
        #print('Feature Screen Shape: ',
        # self.obs.observation.feature_screen.shape)
        #print('Feature Minimap Shape: ',
        # self.obs.observation.feature_minimap.shape)
        spatial_stack = np.expand_dims(np.copy(self.obs.observation.feature_screen), axis=4)
        minimap_stack = np.expand_dims(np.copy(self.obs.observation.feature_minimap), axis=4)
        #print('New screen shape', spatial_stack.shape)
        #print('New Minimap shape: ', minimap_stack.shape)
        #print('All attr: ', self.obs.observation.__dict__.keys())
        spatial_features = ['feature_minimap', 'feature_screen']
        variable_features = ['cargo', 'multi_select', 'build_queue']
        available_actions = ['available_actions']
        #print('Action Spec: ', self.action_spec[0])
        max_no = {'available_actions': len(self.action_spec[0].functions),
                  'cargo': 500, 'multi_select': 500, 'build_queue': 10}
        nonspatial_stack = []
        for k, v in self.obs.observation.items():
            if k not in spatial_features + variable_features + available_actions:
                v[abs(v) == 0] = 1
                #print('Log Static Features: ', np.log(v.reshape(-1)))
                nonspatial_stack = np.concatenate((nonspatial_stack, np.log(v.reshape(-1))))
            elif k in variable_features:
                v[abs(v) == 0] = 1
                #print('Variable value: ', v.reshape(-1))
                #print('Padding: ', np.zeros(max_no[k] * self.obs_spec[0][
                # 'single_select'][1] - len(v.reshape(-1))))
                padded_v = np.concatenate((np.log(v.reshape(-1)), np.zeros(
                    max_no[k] * self.obs_spec[0]['single_select'][1] - len(v.reshape(-1)))))
                nonspatial_stack = np.concatenate((nonspatial_stack, padded_v))
            elif k in available_actions:
                available_actions_v = [1 if action_id in v else 0 for action_id in
                                             np.arange(max_no['available_actions'])]
                nonspatial_stack = np.concatenate(
                    (nonspatial_stack, available_actions_v))
        # TODO Use keras.backend.resize_images to resize minimap to (84, 84)
        print('Processed Input:\n')
        print('\tReward: ', type(reward))
        print('\tSpatial Stack: ', spatial_stack.dtype, spatial_stack.shape)
        print('\tMinimap Stack: ', minimap_stack.dtype, minimap_stack.shape)
        print('\tNon-Spatial Stack: ', nonspatial_stack)
        return reward, spatial_stack, minimap_stack, nonspatial_stack


def main(unused_argv):
    agent = ZergAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()
                Preprocessor(timesteps[0], env.observation_spec(), env.action_spec()).full_input()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
