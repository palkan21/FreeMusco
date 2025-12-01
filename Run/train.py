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


import sys
print(sys.path)


from FreeMuscoCore.Model.freemusco import FreeMusco
from FreeMuscoCore.Utils.misc import load_data, load_yaml
import FreeMuscoCore.Utils.pytorch_utils as ptu
from FreeMuscoCore.Env.mujoco_muscle_env import MujocoMuscleEnv
import argparse


from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

def build_arg(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'panda', type = str)
    parser.add_argument('--show', default = False, action='store_true')

    parser = MujocoMuscleEnv.add_specific_args(parser)
    parser = FreeMusco.add_specific_args(parser)
    args = vars(parser.parse_args())
    ptu.init_gpu(True)

    config = load_yaml(path ='Data/Pretrained/config.yml')
    args.update(config)

    return args

args = None
env = None
free_musco = None
if(mpi_rank == 0):
    args = build_arg()

else:
    pass

#ptu.init_gpu(True)    

args = mpi_comm.bcast(args, root=0)

env = MujocoMuscleEnv(**args)

free_musco= None
if(True): #use_mujoco True
    if(env.mujoco_sim_type == 'muscle'):
        free_musco = FreeMusco(323, 120, 120, env, ** args)

if(True):
    free_musco.save_before_train(args)
    free_musco.train_loop()

