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

import numpy as np
import torch
import typing

from diff_quat import *
import diff_quat as DiffRotation
# import DiffRotation as DiffRotation # a cuda speed up, but not significant...
#----------------------------------State Utils----------------------------------------------#

@torch.jit.script
def broadcast_quat_apply(q: torch.Tensor, vec3: torch.Tensor):
    #return q
    t = 2 * torch.linalg.cross(q[..., :3], vec3, dim=-1)
    xyz: torch.Tensor = vec3 + q[..., 3, None] * t + torch.linalg.cross(q[..., :3], t, dim=-1)
    return xyz
@torch.jit.script
def broadcast_quat_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    multiply 2 quaternions. p.shape == q.shape
    """
    
    w: torch.Tensor = p[..., 3:4] * q[..., 3:4] - torch.sum(p[..., :3] * q[..., :3], dim=-1, keepdim=True)
    xyz: torch.Tensor = (
                p[...,3,None] * q[..., :3] + q[..., 3, None] * p[..., :3] + torch.linalg.cross(p[..., :3], q[..., :3], dim=-1))

    return torch.cat([xyz, w], dim=-1)

def resample(old_pos, old_rot, cur_pos, cur_rot, dt):
    '''
        resample velocity and angular velocity to dt using backward finite difference
    '''
    vel = (cur_pos - old_pos)/dt
    sign = np.sign((cur_rot * old_rot).sum(axis=-1, keepdims=True))
    qd = (cur_rot * sign - old_rot)/dt
    q_conj = old_rot.copy()
    q_conj[..., :3] *= -1
    avel = 2 * quat_product(qd.reshape(-1,4), q_conj)
    return vel, avel[...,:3]   

def character_pos_rot(character):
    pos = character.body_info.get_body_pos()
    rot = character.body_info.get_body_quat()
    rot = MathHelper.flip_quat_by_w(rot)
    return pos, rot

def character_state(character, old_state = None, dt = None):
    '''
        return full state: {pos, rot, vel, avel}
        :param old_state: if old state is not None, it will try to recompute avel 
        and vel according to dt, otherwise it will just return current vel and avel 
    '''
    pos, rot = character_pos_rot(character)
    if old_state is not None:
        # recompute vel
        assert dt is not None
        old_pos, old_rot = old_state[:,:3], old_state[:,3:7]
        vel, avel = resample(old_pos, old_rot, pos, rot, dt)
    else:
        vel = character.body_info.get_body_velo()
        avel = character.body_info.get_body_ang_velo()
    state = np.concatenate([pos, rot, vel, avel], axis=-1, dtype=np.float32)
    return state

def state_to_BodyInfoState(state):
    res = BodyInfoState.BodyInfoState()
    res.pos = np.ascontiguousarray(state[..., 0:3].flatten(), dtype=np.float64)
    res.quat = np.ascontiguousarray(state[..., 3:7].flatten(), dtype=np.float64)
    res.linear_vel = np.ascontiguousarray(state[..., 7:10].flatten(), dtype=np.float64)
    res.angular_vel = np.ascontiguousarray(state[..., 10:13].flatten(), dtype=np.float64)
    res.rot = np.ascontiguousarray(Rotation.from_quat(state[...,3:7].reshape(-1,4)).as_matrix().flatten(), dtype=np.float64)
    return res

def decompose_state(state):
    assert state.shape[-1] ==13
    
    return state[...,0:3], state[...,3:7], state[...,7:10], state[...,10:13]

#---------------------------Observation Utils--------------------------------------------#
# @torch.jit.script
def state2ob(states, model_idx=0):

    # needs pytorch >= 11.0
    if len(states.shape) == 2:
        states = states[None]
    batch_size, num_body, _ = states.shape

    pos = states[..., 0:3] #1, 20, 3
    rot = states[..., 3:7]
    vel = states[..., 7:10]
    avel = states[..., 10:13]

    root_pos = pos[:,0,:].view(-1,1,3)
    root_rot_inv = quat_inv(rot[:,0,:].view(-1,1,4))

    local_pos = broadcast_quat_apply(root_rot_inv, pos - root_pos ).view(batch_size, -1)


    local_vel = broadcast_quat_apply(root_rot_inv, vel).view(batch_size, -1)
    local_avel = broadcast_quat_apply(root_rot_inv, avel).view(batch_size, -1)
    local_rot = flip_quat_by_w(broadcast_quat_multiply(root_rot_inv, rot
                                             )).view(-1,4)
    local_rot = quat_to_vec6d(local_rot).view(batch_size,-1)

    mode_ostrich = False
    if(model_idx == 1):
        mode_ostrich = True

    #print(height[0])
    height = pos[..., 1].view(batch_size, -1)
    up_dir = torch.as_tensor([0, 1, 0]).view(1,1, 3).float().to(states.device)

    if(mode_ostrich == True):
        height = pos[..., 2].view(batch_size, -1)
        up_dir = torch.as_tensor([0, 0, 1]).view(1,1, 3).float().to(states.device)


    local_up_dir = broadcast_quat_apply(root_rot_inv, up_dir.view(1,1,3).float().to(root_rot_inv.device)).view(-1,3)

    if batch_size == 1:
        local_pos = local_pos.flatten()
        local_rot = local_rot.flatten()
        local_vel = local_vel.flatten()
        local_avel = local_avel.flatten()
        height = height.flatten()
        local_up_dir = local_up_dir.flatten()

    return torch.cat([local_pos, local_rot, local_vel, local_avel, height, local_up_dir], dim=-1)


# add jit will be slower... why...?
# @torch.jit.script
def decompose_obs(obs):
    num_dim = obs.shape[-1]
    assert (num_dim - 3) % 16 == 0, "dim error"
    num_body = (num_dim - 3)//16
    pos = obs[...,0:3*num_body]
    rot = obs[...,3*num_body:9*num_body]
    vel = obs[...,9*num_body:12*num_body]
    avel = obs[...,12*num_body:15*num_body]
    height = obs[...,15*num_body:16*num_body]
    up_dir = obs[...,16*num_body:]
    return pos, rot, vel, avel, height, up_dir

#---------------------------Loss Utils (Locomotion Objective Loss) --------------------------------------------#
@torch.jit.script
def locomotion_objective(obs, target, weight:typing.Dict[str, float], dt:float = 1/20, additional=None):
    return 0, 0, 0, 0, 0, 0
    #will be updated

@torch.jit.script
def locomotion_objective_ostrich(obs, target, weight:typing.Dict[str, float], dt:float = 1/20, additional=None):
    return 0, 0, 0, 0, 0, 0
    #will be updated

