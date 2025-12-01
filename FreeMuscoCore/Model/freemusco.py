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

import random
from typing import List,Dict
from numpy import dtype
import torch
from torch import nn
import torch.distributions as D
from FreeMuscoCore.Model.trajectory_collection import TrajectorCollector
from FreeMuscoCore.Model.world_model import RigidMuscleWorldModel 
from FreeMuscoCore.Utils.mpi_utils import gather_dict_ndarray
from FreeMuscoCore.Utils.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
from modules import *
from FreeMuscoCore.Utils.locomotion_utils import *
from FreeMuscoCore.Utils import pytorch_utils as ptu
from FreeMuscoCore.Utils import diff_quat
#from ..Utils import pytorch_utils as ptu
import time
import sys
from FreeMuscoCore.Utils.radam import RAdam
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# whether this process should do tasks such as trajectory collection....
# it's true when it's not root process or there is only root process (no subprocess)
should_do_subprocess_task = mpi_rank > 0 or mpi_world_size == 1

class FreeMusco(nn.Module):
    """
    A FreeMusco agent which includes encoder, decoder and world model
    """
    def __init__(self, observation_size, action_size, delta_size, env, **kargs):
        super().__init__()
        
        target_info_size = observation_size

        self.use_fullbody_change = kargs['use_fullbody_change']
        self.use_model_change = kargs['use_model_change']
        self.model_change_mode = kargs['model_change_mode']
        self.model_idx = kargs['model_idx']

        self.target_mode_trajectory = kargs['posterior_input_change_for_trajectory']

        #self.model_idx = 0
        if(self.use_model_change == True):
            self.model_idx = 1 #ostrich

        self.mujoco_sim_type = kargs['mujoco_sim_type']

        self.use_random_target_vel = kargs['use_random_target_vel']

        if(True):
            if(self.target_mode_trajectory == True):
                target_info_size = 246 # 6 + 120 * 2

                if(self.use_model_change == True):
                    pass

        if(self.use_model_change == True): #ostrich
            #observation_size = 20 * 16 + 3
            #siggraph
            if(True):
                observation_size = 32 * 16 + 3 #input: world, policy
                action_size = 120 #output: policy
                delta_size = 32 * 6 #output: world

        self.init_pose_tensor = None 
        self.use_normalize = kargs['use_normalize'] 

        self.use_hypersphere_z = kargs['use_hypersphere_z']

        self.use_muscle_state_prediction = kargs['use_muscle_state_prediction'] #muscle state as 
        self.use_muscle_state_energy_prediction = kargs['use_muscle_state_energy_prediction']
        self.use_muscle_state_energy_prediction_only = kargs['use_muscle_state_energy_prediction_only']

        self.use_policy_without_vae = False
        self.use_random_target_for_latent_ver_energy = kargs['use_random_target_for_latent_ver_energy']
        self.use_random_target_for_latent_ver_double = kargs['use_random_target_for_latent_ver_double']
        self.ver_double_mk1 = kargs['ver_double_mk1']
        self.ver_double_mk2 = kargs['ver_double_mk2']
        self.ver_double_mk3 = kargs['ver_double_mk3']
        self.ver_double_mk4 = kargs['ver_double_mk4']
        self.mk4_mode = kargs['mk4_mode']

        self.without_vae_version_goal = kargs['without_vae_version_goal']
        self.without_vae_version_goal_target_vel_rot = kargs['without_vae_version_goal_target_vel_rot']
        self.without_vae_version_goal_global_vel_rot = kargs['without_vae_version_goal_global_vel_rot']


        self.use_target_parameter_network = False
        self.target_parameter_network_split_grad = False

        if(self.without_vae_version_goal_target_vel_rot == True):
            target_info_size = 4

            if(self.use_random_target_for_latent_ver_energy == True):
                #target_info_size = 8
                target_info_size = 6

            if(self.use_random_target_for_latent_ver_double == True):
                target_info_size = 54 # 6 + 96
                #target_info_size = 6
                if(self.ver_double_mk1 == True):
                    target_info_size = 60
                if(self.ver_double_mk2 == True):
                    target_info_size = 78
                if(self.ver_double_mk3 == True): #use_fullbody_change
                    target_info_size = 18
                if(self.ver_double_mk4 == True): #use_fullbody_change
                    target_info_size = 24

        if(self.without_vae_version_goal_global_vel_rot == True):
            target_info_size = 8

        self.observation_size_muscle = 0
        self.delta_size_muscle = 0

        if(self.use_muscle_state_prediction == True):
            self.num_muscles = env.num_muscles
            self.num_links = env.scene.num_body
            self.index_rigid_body_end = 13

            #version1 len,vel -> dvel
            self.observation_size_muscle = self.num_muscles * 2
            self.delta_size_muscle = self.num_muscles

            if(self.use_muscle_state_energy_prediction_only == True):
                self.observation_size_muscle = 0
                self.delta_size_muscle = 0

        self.encoder = SimpleLearnablePriorEncoder(
            input_size = target_info_size,
            condition_size= observation_size,
            output_size= kargs['latent_size'],
            fix_var = kargs['encoder_fix_var'],
            **kargs).to(ptu.device)


        if(True):
            #use_policy_withtout_vae False
            self.agent = GatingMixedDecoder(
            # latent_size= kargs['latent_size'],
            condition_size= observation_size,
            output_size=action_size,
            **kargs
            ).to(ptu.device)


        statistics = None
        self.obs_mean = None
        self.obs_std = None

        self.world_model = RigidMuscleWorldModel(observation_size + self.observation_size_muscle, action_size, delta_size + self.delta_size_muscle, env.dt, statistics, **kargs).to(ptu.device)

        self.wm_optimizer = RAdam(self.world_model.parameters(), kargs['world_model_lr'], weight_decay=1e-3)

        self.vae_optimizer = RAdam( list(self.encoder.parameters()) + list(self.agent.parameters()), kargs['freemusco_lr'])

        self.beta_scheduler = ptu.scheduler(0,8,0.009,0.09,500*8)
        
        #hyperparameters....
        self.action_sigma = 0.05
        self.max_iteration = kargs['max_iteration']
        self.collect_size = kargs['collect_size']
        self.sub_iter = kargs['sub_iter']
        self.save_period = kargs['save_period']
        self.evaluate_period = kargs['evaluate_period']
        self.world_model_rollout_length = kargs['world_model_rollout_length']
        self.freemusco_rollout_length = kargs['freemusco_rollout_length']
        self.world_model_batch_size = kargs['world_model_batch_size']
        self.freemusco_batch_size = kargs['freemusco_batch_size']
        
        # policy training weights                                    
        self.weight = {}
        for key,value in kargs.items():
            if 'freemusco_weight' in key:
                self.weight[key.replace('freemusco_weight_','')] = value
        
        # for real trajectory collection
        self.runner = TrajectorCollector(venv = env, actor = self, runner_with_noise = True,
            use_muscle_state_prediction=self.use_muscle_state_prediction,
            use_muscle_state_energy_prediction=self.use_muscle_state_energy_prediction)

        self.env = env    

        self.replay_buffer = ReplayBuffer(self.replay_buffer_keys, kargs['replay_buffer_size']) if mpi_rank ==0 else None

        if(True): #not used now, for angular momentum loss??
            self.tensor_inertia = ptu.from_numpy(self.env.arr_inertia)
            self.tensor_mass = ptu.from_numpy(self.env.arr_mass)
            self.tensor_com = ptu.from_numpy(self.env.arr_com)

        if(False): #for inference
            self.env.world_model = self.world_model

    #--------------------------------for MPI sync------------------------------------#
    def parameters_for_sample(self):
        '''
        this part will be synced using mpi for sampling, world model is not necessary
        '''

        return {
            'encoder': self.encoder.state_dict(),
            'agent': self.agent.state_dict()
        }
    def load_parameters_for_sample(self, dict):
       
        self.encoder.load_state_dict(dict['encoder'])
        self.agent.load_state_dict(dict['agent'])


    #-----------------------------for replay buffer-----------------------------------#
    @property
    def world_model_data_name(self):
        return ['state', 'action']
    
    @property
    def policy_data_name(self):
        return ['state', 'target']
    
    @property
    def replay_buffer_keys(self):
        return ['state', 'action', 'target']

    #----------------------------for training-----------------------------------------#
    def train_one_step(self):        
        time1 = time.perf_counter()

        # data used for training world model
        name_list = self.world_model_data_name #state, action
        if(self.use_muscle_state_prediction == True):
            #name_list.append('energy')
            name_list.append('muscle_state')
            if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                name_list.append('energy')

        if(True):
            rollout_length = self.world_model_rollout_length
            data_loader = self.replay_buffer.generate_data_loader(name_list, 
                                rollout_length+1, # needs additional one state...
                                self.world_model_batch_size, 
                                self.sub_iter)
            for batch in  data_loader:
                world_model_log = self.train_world_model(*batch)
    
            print('world training... DONE')

        time2 = time.perf_counter()

        # data used for training policy
        name_list = self.policy_data_name

        if(self.use_muscle_state_prediction == True):
            name_list.append('muscle_state')

        policy_log = None

        rollout_length = self.freemusco_rollout_length #state, target
        data_loader = self.replay_buffer.generate_data_loader(name_list, 
                            rollout_length, 
                            self.freemusco_batch_size, 
                            self.sub_iter)

        if(True):
            for batch in data_loader:
                policy_log = self.train_policy(*batch)

            print('task training... DONE')


        # log training time...
        time3 = time.perf_counter()     

        if(True):
            world_model_log['training_time'] = (time2-time1)
            policy_log['training_time'] = (time3-time2)

        # merge the training log...
        return self.merge_dict([world_model_log, policy_log], ['WM','Policy'])


    def mpi_sync(self):
        # sample trajectories
        if should_do_subprocess_task:
            with torch.no_grad():
                path : dict = self.runner.trajectory_sampling( math.floor(self.collect_size/max(1, mpi_world_size -1)), self )
                self.env.update_val(path['done'], path['rwd'], path['frame_num'])
        else:
            path = {}

        tmp = np.zeros_like(self.env.val)
        mpi_comm.Allreduce(self.env.val, tmp)        
        self.env.val = tmp / mpi_world_size

        self.env.update_p()

        res = gather_dict_ndarray(path)
        if mpi_rank == 0:
            print(mpi_rank, 'mpi_sync')
            paramter = self.parameters_for_sample()
            mpi_comm.bcast(paramter, root = 0)
            self.replay_buffer.add_trajectory(res)
            info = {
                'rwd_mean': np.mean(res['rwd']),
                'rwd_std': np.std(res['rwd']),
                'episode_length': len(res['rwd'])/(res['done']!=0).sum()
            }
            #print(self.replay_buffer)
        else:
            print(mpi_rank, 'mpi_sync')
            paramter = mpi_comm.bcast(None, root = 0)
            self.load_parameters_for_sample(paramter)    
            info = None
        return info
    

    def train_loop(self):
        import time
        """training loop, MPI included
        """

        self.weight['avel'] = 0.2
        self.current_train_iteration = 0

        for i in range(self.max_iteration):
            self.current_train_iteration += 1

            # if i ==0:
            before_time = time.time()
            info = self.mpi_sync() # communication, collect samples and broadcast policy
            
            print('trajectory sampling... DONE')

            if mpi_rank == 0:
                print(f"----------training {i} step-------- BEGIN")
                sys.stdout.flush()
                log = self.train_one_step()   
                log.update(info)       
                self.try_save(i)
                self.try_log(log, i)

                print(log)

            if should_do_subprocess_task:
                self.try_evaluate(i)
                print(i, 'sub process')

            after_time = time.time()
            print('time:', after_time - before_time)
            print('')
                
    # -----------------------------------for logging----------------------------------#
    @property
    def dir_prefix(self):
        return 'Experiment'
    
    def save_before_train(self, args):
        """build directories for log and save
        """
        import os, time, yaml
        time_now = time.strftime("%Y%m%d %H-%M-%S", time.localtime())
        dir_name = args['experiment_name']+'_'+time_now
        dir_name = mpi_comm.bcast(dir_name, root = 0)
        
        self.log_dir_name = os.path.join(self.dir_prefix,'log',dir_name)
        self.data_dir_name = os.path.join(self.dir_prefix,'checkpoint',dir_name)
        if mpi_rank == 0:
            os.makedirs(self.log_dir_name)
            os.makedirs(self.data_dir_name)

        mpi_comm.barrier()
        if mpi_rank > 0:
            f = open(os.path.join(self.log_dir_name,f'mpi_log_{mpi_rank}.txt'),'w')
            sys.stdout = f
            return
        else:
            yaml.safe_dump(args, open(os.path.join(self.data_dir_name,'config.yml'),'w'))
            self.logger = SummaryWriter(self.log_dir_name)
            
    def try_evaluate(self, iteration):
        return #not used
        if iteration % self.evaluate_period == 0:
            bvh_saver = self.runner.eval_one_trajectory(self)
            bvh_saver.to_file(os.path.join(self.data_dir_name,f'{iteration}_{mpi_rank}.bvh'))
        pass    
    
    def try_save(self, iteration):
        if iteration % self.save_period ==0:
            check_point = {
                'self': self.state_dict(),
                'wm_optim': self.wm_optimizer.state_dict(),
                'vae_optim': self.vae_optimizer.state_dict(),
                'balance': self.env.val
            }
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))
    
    def try_load(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        self.load_state_dict(data['self'], strict = False)
        self.wm_optimizer.load_state_dict(data['wm_optim'])
        self.vae_optimizer.load_state_dict(data['vae_optim'])

        if 'balance' in data:
            self.env.val = data['balance']
            self.env.update_p()
        return data
        
    def try_log(self, log, iteration):
        for key, value in log.items():
            self.logger.add_scalar(key, value, iteration)
        self.logger.flush()
    
    def cal_rwd(self, **obs_info): 
        #observation = obs_info['observation']
        #target = obs_info['target']

        if(self.target_mode_trajectory == True):
            return 0.01

    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--latent_size", type = int, default = 64, help = "dim of latent space")
        arg_parser.add_argument("--max_iteration", type = int, default = 20001, help = "iteration for freemusco training")
        arg_parser.add_argument("--collect_size", type = int, default = 2048, help = "number of transition collect for each iteration")
        arg_parser.add_argument("--sub_iter", type = int, default = 8, help = "num of batch in each iteration")
        arg_parser.add_argument("--save_period", type = int, default = 100, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--evaluate_period", type = int, default = 100, help = "save checkpoints for every * iterations")
        arg_parser.add_argument("--replay_buffer_size", type = int, default = 50000, help = "buffer size of replay buffer")

        return arg_parser
    
    #--------------------------API for encode and decode------------------------------#
    
    def encode(self, normalized_obs, normalized_target, **kargs):
        #encode observation and target into posterior distribution

        return self.encoder(normalized_obs, normalized_target)
    
    def decode(self, normalized_obs, latent, **kargs):
        #decode latent code into action space

        if(self.use_hypersphere_z == True): #hypersphere #model_idx
            import torch.nn.functional as F
            latent_normalized = F.normalize(latent, p=2, dim=1, eps=1e-12)
            action = self.agent(latent_normalized, normalized_obs) 
            return action

        action = self.agent(latent, normalized_obs) 
        return action
    
    def normalize_obs(self, observation):
        if isinstance(observation, np.ndarray):
            observation = ptu.from_numpy(observation)
        if len(observation.shape) == 1:
            observation = observation[None,...]

        if(self.use_normalize == False):
            return observation
        return ptu.normalize(observation, self.obs_mean, self.obs_std) #(obs - mean) / std
    
    def obsinfo2n_obs(self, obs_info):
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']
        else:
            if 'observation' in obs_info:
                observation = obs_info['observation']
            else:
                observation = state2ob(obs_info['state'])
            n_observation = self.normalize_obs(observation)
        return n_observation

    def act_policy(self, **obs_info):

        target = obs_info['target']

        n_target = self.normalize_obs(target)

        n_observation = self.obsinfo2n_obs(obs_info)

        info = {}

        #action = self.decode(n_observation, latent_code) #original
        #action = self.decode(n_observation, None) #version1 (obs)
        action = self.decode(n_observation, n_target) #version2 (obs, target)

        return action, info
    
    def act_tracking(self, **obs_info):

        target = obs_info['target']

        n_target = self.normalize_obs(target)
        n_observation = self.obsinfo2n_obs(obs_info)

        
        info = {}

        if(True):
            latent_code, mu_post, mu_prior = self.encode(n_observation, n_target) 
            info = {"mu_prior": mu_prior, "mu_post": mu_post}
        
        action = self.decode(n_observation, latent_code)

        return action, info
    
    def act_prior(self, obs_info, to_tensor=True):

        n_observation = self.obsinfo2n_obs(obs_info)


        latent_code, mu_prior, logvar = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent_code)


        return action
        #info = {"mu_prior": mu_prior, "mu_post": torch.zeros_like(mu_prior)}
        #return action, info
    

    #----------------------------------API imitate PPO--------------------------------#
    def act_determinastic(self, obs_info):
        action, _ = self.act_tracking(**obs_info)
        return action
                
    def act_distribution(self, obs_info):
        """
        Add noise to the output action
        """
        action = self.act_determinastic(obs_info)
        action_distribution = D.Independent(D.Normal(action, self.action_sigma), -1)
        return action_distribution
    
    #--------------------------------------Utils--------------------------------------#
    @staticmethod
    def merge_dict(dict_list: List[dict], prefix: List[str]):
        """Merge dict with prefix, used in merge logs from different model

        Args:
            dict_list (List[dict]): different logs
            prefix (List[str]): prefix you hope to add before keys
        """
        res = {}
        for dic, prefix in zip(dict_list, prefix):
            for key, value in dic.items():
                res[prefix+'_'+key] = value
        return res

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def try_load_world_model(self, data_file):
        data = torch.load(data_file, map_location=ptu.device)
        wm_dict = data['self']
        wm_dict = {k.replace('world_model.',''):v for k,v in wm_dict.items() if 'world_model' in k}
        self.world_model.load_state_dict(wm_dict)
        return data
    #--------------------------------Training submodule-------------------------------#
    
    def train_policy(self, states, targets, additional=None): #additional == future #additional2
        #will be updated
        #locomotion_utils.locomotion_objective()
        res = None
        return res

    def train_world_model(self, states, actions, additional=None, additional2=None, additional3=None): #additional == energy # or muscle_state
        rollout_length = states.shape[1] -1
        loss_name = ['pos', 'rot', 'vel', 'avel']

        loss_energy_list = []
        loss_muscle_list = []
        loss_contact_list = []

        loss_num = len(loss_name)
        loss = list( ([] for _ in range(loss_num)) )
        states = states.transpose(0,1).contiguous().to(ptu.device)
        actions = actions.transpose(0,1).contiguous().to(ptu.device)

        if(self.use_muscle_state_prediction == True):
            #print(additional.shape) #512, 9, 240 #muscle_state
            additional = additional.transpose(0,1).contiguous().to(ptu.device)
            #9, 512, 240

            if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                additional2 = additional2.transpose(0,1).contiguous().to(ptu.device)
                #9, 512, 120


        cur_state = states[0]

        cur_muscle_state = None
        next_muscle_state = None

        if(self.use_muscle_state_prediction == True):
            cur_rigid_state = states[0]
            cur_muscle_state = additional[0]
            cur_energy_state = None
            cur_contact_state = None

            next_rigid_state = None
            next_muscle_state = None
            next_energy_state = None
            next_contact_state = None

            if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                cur_energy_state = additional2[0]

            for i in range(rollout_length):
                loss_tmp = None
                loss_muscle = None
                loss_energy = None
                
                next_rigid_state = states[i+1]
                next_muscle_state = additional[i+1]
                if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                    next_energy_state = additional2[i+1]

                #todo world model forward args
                current_action = actions[i]

                pred_rigid_state, pred_muscle_state, pred_energy_state, pred_contact_state = self.world_model(cur_rigid_state, cur_muscle_state, current_action)

                #print(pred_energy_state.shape, pred_contact_state.shape)

                loss_tmp = self.world_model.loss(pred_rigid_state, next_rigid_state)

                if(self.use_muscle_state_energy_prediction_only == False): #False
                    loss_muscle = self.world_model.loss_muscle(pred_muscle_state, next_muscle_state)
                    loss_muscle_list.append(loss_muscle)

                cur_rigid_state = pred_rigid_state
                cur_muscle_state = pred_muscle_state
                #cur_energy_state = None

                if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                    loss_energy = self.world_model.loss_energy(pred_energy_state, next_energy_state)
                    cur_energy_state = pred_energy_state 
                    #actually, not used as world model input
                    loss_energy_list.append(loss_energy)


                for j in range(loss_num):
                    loss[j].append(loss_tmp[j])
        #print('')
        
        loss_value = [sum(i) for i in loss]
        loss = sum(loss_value)

        sum_energy_loss = sum(loss_energy_list)
        sum_muscle_loss = sum(loss_muscle_list)
        sum_contact_loss = sum(loss_contact_list)

        #print(loss, sum_muscle_loss)

        loss = loss + sum_energy_loss + sum_muscle_loss + sum_contact_loss


        self.wm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1, error_if_nonfinite=True)
        self.wm_optimizer.step()
        res= {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['loss'] = loss

        #my
        res['energy_pred_loss'] = sum_energy_loss
        res['muscle_pred_loss'] = sum_muscle_loss
        res['contact_loss'] = sum_contact_loss

        return res

def build_arg(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'panda', type = str)
    parser.add_argument('--show', default = False, action='store_true')
    #print(parser)
    #print(xxx)
    parser = MujocoMuscleEnv.add_specific_args(parser)
    parser = FreeMusco.add_specific_args(parser)
    args = vars(parser.parse_args())
    ptu.init_gpu(True)
    # con
    #print(args)
    config = load_yaml(initialdir ='Data/NNModel/Pretrained')
    args.update(config)
    #print('')
    #print(args)
    #print(xxx)

    return args
