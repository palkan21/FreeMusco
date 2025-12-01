'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''
#THIS CODE IS BASED ON THE FOLLOWING REPOSITORY: Control-VAE (https://github.com/heyuanYao-pku/Control-VAE)
#AND MODIFIED BY THE AUTHORS OF THE SIGGRAPH ASIA 2025 CONFERENCE PAPER, CGR LAB, HANYANG UNIVERSITY.
#FREEMUSCO: MOTION-FREE LEARNING OF LATENT CONTROL FOR MORPHOLOGY-ADAPTIVE LOCOMOTION IN MUSCULOSKELETAL CHARACTERS

from torch import nn
import torch
import FreeMuscoCore.Utils.pytorch_utils as ptu
import FreeMuscoCore.Utils.diff_quat as DiffRotation
from FreeMuscoCore.Utils.locomotion_utils import *

# @torch.jit.script
def integrate_obs_vel(states, delta):
    shape_ori = states.shape

    if len(states.shape) == 2:
        states = states[None]
    assert len(states.shape) == 3, "state shape error"

    if len(delta.shape) == 2:
        delta = delta[None]
    assert len(delta.shape) == 3, "acc shape error"
    
    assert delta.shape[-1] == 6
    assert states.shape[-1] == 13

    batch_size, num_body, _ = states.shape

    
    vel = states[..., 7:10].view(-1, 3)
    avel = states[..., 10:13].view(-1, 3)

    delta = delta.view(-1, 6)
    local_delta = delta[..., 0:3]
    local_ang_delta = delta[..., 3:6]

    root_rot = torch.tile(states[:, 0, 3:7], [1, 1, num_body]).view(-1, 4)
    true_delta = quat_apply(root_rot,local_delta)
    true_ang_delta = quat_apply(root_rot, local_ang_delta)

    next_vel = vel + true_delta
    next_avel = avel + true_ang_delta
    return next_vel, next_avel


def integrate_state(states, delta, dt):
    pos = states[..., 0:3].view(-1, 3)
    rot = states[..., 3:7].view(-1, 4)
    
    next_vel, next_avel = integrate_obs_vel(states, delta)
    next_pos = pos + next_vel * dt
    next_rot = quat_integrate(rot, next_avel, dt)
    
    batch_size, num_body, _ = states.shape
    
    next_state = torch.cat([next_pos.view(batch_size, num_body, 3),
                                 next_rot.view(batch_size, num_body, 4),
                                 next_vel.view(batch_size, num_body, 3),
                                 next_avel.view(batch_size, num_body, 3)
                                 ], dim=-1)

    return next_state.view(states.shape)

class RigidMuscleWorldModel(nn.Module):
    def __init__(self, ob_size, ac_size, delta_size, dt, statistics, **kargs) -> None:
        super(RigidMuscleWorldModel, self).__init__()
        self.use_normalize = kargs['use_normalize']
        #self.use_rot6d_action = kargs['use_rot6d_action']

        self.use_muscle_state_energy_prediction = kargs['use_muscle_state_energy_prediction']
        self.use_muscle_state_energy_prediction_only = kargs['use_muscle_state_energy_prediction_only']

        self.model_idx = kargs['model_idx']

        if(kargs['use_model_change'] == True):
            self.model_idx = 1 #ostrich

        world_model_input_dim = ob_size + ac_size
        #model idx

        #fullbody, ostrich
        len_muscle = 120 

        self.num_muscles = len_muscle


        if(True):
            out_mlp1 = delta_size #240
            if(self.use_muscle_state_energy_prediction == True):
                out_mlp1 = delta_size + len_muscle #240 + 120
            elif(self.use_muscle_state_energy_prediction_only == True):
                out_mlp1 = delta_size + len_muscle #120 + 120
                #world_model_input_dim = ob_size + ac_size

            self.mlp1 = ptu.build_mlp(
                input_dim = world_model_input_dim,
                #input_dim = ob_size + ac_size*2, # rotvec to 6d vector
                #input_dim = ob_size + ac_size, # observation + action
                output_dim= out_mlp1,
                hidden_layer_num=kargs['world_model_hidden_layer_num'],
                hidden_layer_size=kargs['world_model_hidden_layer_size'],
                activation=kargs['world_model_activation']
                #activation='tanh'
            )
            self.mlp2 = None #muscle_world_model

        self.dt = dt
        self.weight = {}
        for key,value in kargs.items():
            if 'world_model_weight' in key:
                self.weight[key.replace('world_model_weight_','')] = value

    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]
        if(self.use_normalize == False):
            return observation
        observation = ptu.normalize(observation, self.obs_mean, self.obs_std)
        return observation

    @staticmethod
    def integrate_state(states, delta, dt): #rigid state
        #assert state 323 delta 120 
        pos = states[..., 0:3].view(-1, 3)
        rot = states[..., 3:7].view(-1, 4)
        batch_size, num_body, _ = states.shape

        vel = states[..., 7:10].view(-1, 3)
        avel = states[..., 10:13].view(-1, 3)

        root_rot = states[:, 0, 3:7].view(-1,1,4)        
        delta = delta.view(batch_size, -1, 3)
        
        true_delta = broadcast_quat_apply(root_rot, delta).view(batch_size, -1, 6)
        true_delta, true_ang_delta = true_delta[...,:3].view(-1,3), true_delta[...,3:].view(-1,3)
        
        next_vel = vel + true_delta
        next_avel = avel + true_ang_delta
        
        next_pos = pos + next_vel * dt
        next_rot = quat_integrate(rot, next_avel, dt)
        next_state = torch.cat([next_pos.view(batch_size, num_body, 3),
                                    next_rot.view(batch_size, num_body, 4),
                                    next_vel.view(batch_size, num_body, 3),
                                    next_avel.view(batch_size, num_body, 3)
                                    ], dim=-1)
        return next_state.view(states.shape)

        
    @staticmethod
    def integrate_muscle_state(states, delta, dt): #muscle state
        num_muscles = 120

        muscle_len = states[:, :num_muscles]
        muscle_vel = states[:, num_muscles:num_muscles*2]

        next_vel = muscle_vel + delta #delta == 120
        next_len = muscle_len + next_vel * dt

        next_state = torch.cat([next_len, next_vel], dim=1)
        return next_state


    def loss(self, pred, tar): #rigid
        pred_pos, pred_rot, pred_vel, pred_avel = decompose_state(pred)
        tar_pos, tar_rot, tar_vel, tar_avel = decompose_state(tar)

        weight_pos, weight_vel, weight_rot, weight_avel = self.weight[
            "pos"], self.weight["vel"], self.weight["rot"], self.weight["avel"]

        batch_size = tar_pos.shape[0]

        pos_loss = weight_pos * \
            torch.mean(torch.norm(pred_pos - tar_pos, p=1, dim=-1))
        vel_loss = weight_vel * \
            torch.mean(torch.norm(pred_vel - tar_vel, p=1, dim=-1))
        avel_loss = weight_avel * \
            torch.mean(torch.norm(pred_avel - tar_avel, p=1, dim=-1))

        # special for rotation
        pred_rot_inv = quat_inv(pred_rot)
        tar_rot = DiffRotation.flip_quat_by_w(tar_rot.view(-1,4))
        pred_rot_inv = DiffRotation.flip_quat_by_w(pred_rot_inv.view(-1, 4))
        dot_pred_tar = torch.norm( DiffRotation.quat_to_rotvec( DiffRotation.quat_multiply(tar_rot,
                                              pred_rot_inv) ), p =2, dim=-1)
        rot_loss = weight_rot * \
            torch.mean(torch.abs(dot_pred_tar))

        return pos_loss, rot_loss, self.dt * vel_loss, self.dt * avel_loss


    def loss_energy(self, pred, tar):
        weight_energy = 0.1 #predict_energy_version_multi

        energy_pred_loss = weight_energy * \
            torch.mean(torch.norm(pred - tar, p=1, dim= -1))

        return energy_pred_loss

    def loss_muscle(self, pred, tar):
        weight_muscle = 0.1

        muscle_pred_loss = weight_muscle * \
            torch.mean(torch.norm(pred - tar, p=1, dim= -1))

        return muscle_pred_loss

    def forward(self, state, muscle_state, action, **obs_info):

        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                observation = state2ob(state)
            n_observation = self.normalize_obs(observation)

        n_muscle = muscle_state

        batch_size = 1 if len(action.shape)==1 else action.shape[0]

        n_delta_rigid = None
        n_delta_muscle = None
        n_delta_energy = None

        index_rigid = 120 #fullbody

        if(self.model_idx == 1):
            index_rigid = 32 * 6 #ostrich
        if(self.model_idx == 2):
            index_rigid = 25 * 6

        if(True):
            if(self.use_muscle_state_energy_prediction_only == True):
                n_delta = self.mlp1( torch.cat([n_observation, action], dim = -1) )
                n_delta_rigid = n_delta[:, :index_rigid]
                n_delta_energy = n_delta[:, index_rigid:index_rigid+self.num_muscles]

            else: 
                n_delta = self.mlp1( torch.cat([n_observation, n_muscle, action], dim = -1) )
    
                n_delta_rigid = n_delta[:, :index_rigid]
                n_delta_muscle = n_delta[:, index_rigid:index_rigid+self.num_muscles]
                if(self.use_muscle_state_energy_prediction == True):
                    n_delta_energy = n_delta[:, index_rigid+self.num_muscles:index_rigid+self.num_muscles*2]

        rigid_state = self.integrate_state(state, n_delta_rigid, self.dt)
        if(self.use_muscle_state_energy_prediction_only == False):
            muscle_state = self.integrate_muscle_state(n_muscle, n_delta_muscle, self.dt)
        energy_state = n_delta_energy #NONE

        contact_state = None
        return rigid_state, muscle_state, energy_state, contact_state


#class SimpleWorldModel(nn.Module): #deprecated
