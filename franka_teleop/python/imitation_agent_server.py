"""This node must be executed in virtual environment.
"""

import rospy
from franka_teleop.srv import *
import tensorflow as tf
import time

import pickle
import numpy as np
from rlbench.backend.observation import Observation
from PIL import Image

from integral.models.lstm import LSTM
from integral.agents.agent import Agent
from integral.models.image_vae import ConvVae

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'lstm_path', None, 'path of LSTM model for imitation model')
flags.DEFINE_string(
    'vae_path', None, 'path of VAE model for image processing')
flags.DEFINE_integer(
    'action_dim', 16, 'length of action dimention')


class ImitationAgentServer:

    def __init__(self, lstm_path, image_vae_path=None, action_dim=None):
        rospy.init_node('imitation_agent_server')
        self.srv = rospy.Service('/command_imitation_agent', CommandImitationAgent, self.handle_srv)
        rospy.loginfo('Imitation agent server is ready to recive command...')
        lstm_builder = LSTM(saved_model=lstm_path)
        self.lstm_model = lstm_builder.model
        if not image_vae_path:
            self._use_fake_embedding = True
        else:
            self._use_fake_embedding = False
            self.image_vae = ConvVae(saved_model=image_vae_path)
            self.image_encorder = self.image_vae.encoder
        self.action_dim = action_dim
        self.internal_count = 0

    def handle_srv(self, req):
        """Handle ROS service in here
        """
        start = time.time()        
        if req.command_type == 'reset':
            self.reset()
            return CommandImitationAgentResponse(ok=True,
                                                 calc_time=time.time()-start)
        elif req.command_type == 'act':
            obs = self._process_observation(req.input_vector,
                                            req.input_images)
            
            action, end_prob = self.act(obs)
            return CommandImitationAgentResponse(ok=True,
                                                 action=action,
                                                 calc_time=time.time()-start,
                                                 end_prob=end_prob)
        elif req.command_type == 'get_trajectory':
            obs = self._process_observation(req.input_vector,
                                            req.input_images)
            trajectory = self.get_predicted_trajectory(req.obs,
                                                       req.trajectory_steps,
                                                       execlude_action=False)
            return CommandImitationAgentResponse(ok=True,
                                                 trajectory=trajectory,
                                                 calc_time=time.time()-start)
        

    def act(self, obs, action_size=None):
        if (action_size is None) and (self.action_dim is not None):
            action_size = self.action_dim
        obs = np.expand_dims(np.expand_dims(obs, axis=0), axis=0)
        pred = self.lstm_model.predict(obs)[0][0]
        if action_size:
            # Here we are assuming that actions & flag are at the end of output vector.
            self.action = pred[-(action_size + 1):-1]
        else:
            # Output is purely actions & flag.
            self.action = pred[:-1]

        return self.action, pred[-1]  # action and end_prob

    def reset(self):
        self.internal_count = 0
        self.action = None
        # resetting lstm internal memory
        self.lstm_model.reset_states()
        rospy.loginfo('Resetting LSTM state...')
        

    def get_predicted_trajectory(self, obs, steps=1, exclude_action=True, action_size=None):
        """Get n-step ahead trajectory that is generated by reccurent inference.
        The internal states of the all LSTM layers are restored to the initial internal state after inference.
        """
        obs_first = obs  # first observatio feeded for model
        obs = np.expand_dims(np.expand_dims(obs, axis=0), axis=0)
        current_lstm_states = self._get_lstm_states()
        predicted_trajectory = []
        for i in range(steps):
            pred_obs = self.lstm_model.predict(obs)
            if exclude_action:
                predicted_trajectory.append(pred_obs[0][0][:-(self.action_dim + 1)])
            else:
                predicted_trajectory.append(pred_obs[0][0])
            obs = pred_obs
        predicted_trajectory = np.array(predicted_trajectory)
        self._restore_lstm_states(current_lstm_states)
        
        return obs_first, predicted_trajectory

    def _get_lstm_states(self) -> list:
        """Save lstm internal states (h and c)
        """
        states = []
        for l in self.lstm_model.layers:
            if 'lstm'in l.name:
                states.append([l.states[0].numpy(), l.states[1].numpy()])
        return states

    def _restore_lstm_states(self, states: list):
        """Restore lstm internal states (h and c)
        """
        idx = 0
        for l in self.lstm_model.layers:
            if 'lstm'in l.name:
                l.reset_states(states[idx])
                idx += 1
    
    @property
    def name(self):
        return self.lstm_model.name

    def get_predicted_state(self, obs, exclude_action=True, action_size=None):
        """Returns one step ahead prediction
        """
        return self.get_predicted_trajectory(obs, 1, exclude_action, action_size)[0]

    def _process_observation(self, input_vector, input_images):
        """concatenate observations to make a vector for one instance time step.
        """
        # TODO
        input_vector = np.array(input_vector)
        embs = []
        for im_msg in input_images:
            im = np.frombuffer(im_msg.data, np.uint8).reshape(im_msg.height, im_msg.width, 3)
            img = Image.fromarray(im)
            input_shape = (self.image_encorder.input.shape[1], self.image_encorder.input.shape[2])
            img = np.array([np.array(img.resize(input_shape))])
            img = img/255.0
            emb = self.image_encorder([img])  # extract latent vector form image
            embs.append(np.array(emb).flatten())
        return np.concatenate((np.concatenate(embs), input_vector, [0]))

    def run(self):
        rospy.spin()


def main(argv):
    node = ImitationAgentServer(lstm_path='/home/ykawamura/integral_models/rlbench_pick_data/default_f_128x128_0epis/image_conv_vae_16dim_stride2_fc_bce_1.0kl/adam.001_itdecay1k/model_epoch0800_4.57e-01_4.59e-01/maxlen300_pred_shift1_emb-ja-go_emb-ja-go_noise-0.001-0.001_18epis/lstm_256units_split-mse_relu_n_lstm_layers_2_n_dense_layers_2_emb_loss_scale0.001/adam.001_itdecay1k_l2wreg1e-05/model_last_epoch1867.h5',
                                image_vae_path='/home/ykawamura/integral_models/rlbench_pick_data/default_f_128x128_0epis/image_conv_vae_16dim_stride2_fc_bce_1.0kl/adam.001_itdecay1k/model_epoch0800_4.57e-01_4.59e-01.h5',
                                action_dim=16)
    node.run()


if __name__ == '__main__':
    app.run(main)
