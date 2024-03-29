import numpy as np
import keras
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

from SC2Agent import ZergAgent


class Preprocessor:
    def __init__(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def process_avail_actions(self, available_actions):
        avail = np.zeros(len(self.action_spec[0].functions))
        avail[available_actions] = 1
        return avail

    def full_input(self, obs):
        if obs.last():
            done = 1
        else:
            done = 0
        reward = obs.reward
        screen = obs.observation.feature_screen
        minimap = obs.observation.feature_minimap
        #print('Feature Screen Shape: ', screen.shape)
        #print('Feature Minimap Shape: ', minimap.shape)
        screen_stack = np.expand_dims(np.copy(screen).
                                      reshape((screen.shape[1],
                                               screen.shape[2],
                                               screen.shape[0])), axis=0).\
            astype(np.float32)
        minimap_stack = np.expand_dims(np.copy(minimap).
                                       reshape((minimap.shape[1],
                                                minimap.shape[2],
                                                minimap.shape[0])), axis=0).\
            astype(np.float32)
        #print('New screen shape', spatial_stack.shape)
        #print('New Minimap shape: ', minimap_stack.shape)
        #print('All attr: ', obs.observation.__dict__.keys())
        spatial_features = ['feature_minimap', 'feature_screen']
        variable_features = ['cargo', 'multi_select', 'build_queue']
        available_actions = ['available_actions']
        #print('Action Spec: ', len(self.action_spec[0].functions))
        #print('Single select:', self.obs_spec[0]['single_select'][1])
        max_no = {'available_actions': len(self.action_spec[0].functions),
                  'cargo': 500, 'multi_select': 500, 'build_queue': 10}
        nonspatial_stack = []
        for k, v in obs.observation.items():
            if k not in spatial_features + variable_features + available_actions:
                #v[abs(v) == 0] = 1
                #print('Log Static Features: ', np.log(v.reshape(-1)))
                nonspatial_stack = np.concatenate((nonspatial_stack, np.log1p(v.reshape(-1))))
                #print('Non-spatial:', k, ':', v.reshape(-1).shape)
            elif k in variable_features:
                #v[abs(v) == 0] = 1
                #print('Variable value: ', v.reshape(-1))
                #print('Padding: ', np.zeros(max_no[k] * obs_spec[0][
                # 'single_select'][1] - len(v.reshape(-1))))
                padded_v = np.concatenate((np.log1p(v.reshape(-1)), np.zeros(
                    max_no[k] * self.obs_spec[0]['single_select'][1] - len(v.reshape(-1)))))
                nonspatial_stack = np.concatenate((nonspatial_stack, padded_v))
                #print('Variable feature:', k, ':', padded_v.shape)
            elif k in available_actions:
                available_actions_v = [1 if action_id in v else 0 for action_id in
                                             np.arange(max_no['available_actions'])]
                #print('Available actions:', available_actions_v)
                #print('Length before action cat:', len(nonspatial_stack))
                nonspatial_stack = np.concatenate(
                    (nonspatial_stack, available_actions_v))
                #print('Length after action cat:', len(nonspatial_stack))
        state_shape = [shape for shape in screen_stack.shape[:3]]
        #print('State shape:', state_shape)
        #print('Nonspatial length before reshape:', len(nonspatial_stack))
        nonspatial_stack = np.reshape(np.concatenate((nonspatial_stack,
                                                     np.zeros(3*state_shape[1] *
                                                              state_shape[2] -
                                                              len(nonspatial_stack))))
                                      , tuple(state_shape + [3])).astype(np.float32)
        #print('Non-spatial after reshape:', nonspatial_stack.shape)
        #print('Processed Input:\n')
        #print('\tReward: ', type(reward))
        #print('\tSpatial Stack: ', spatial_stack.dtype, spatial_stack.shape)
        #print('\tMinimap Stack: ', minimap_stack.dtype, minimap_stack.shape)
        #print('\tNon-Spatial Stack: ', nonspatial_stack.dtype, nonspatial_stack.shape)
        #print('\tNon-Spatial Data:', nonspatial_stack)
        #return reward, spatial_stack, minimap_stack, nonspatial_stack
        return [nonspatial_stack, screen_stack, minimap_stack], reward, done

    def __call__(self, obs):
        return self.full_input(obs)


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
                        feature_dimensions=features.Dimensions(screen=64,
                                                               minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()
                preprocessor = Preprocessor(env.observation_spec(), env.action_spec())
                processed = preprocessor(timesteps[0])
                print('Non-spatial input:', processed[0][0].shape)
                print('Screen input:', processed[0][1].shape)
                print('Minimap input:', processed[0][2].shape)
                print('Reward:', processed[1])
                print('Done:', processed[2])

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
