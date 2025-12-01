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

import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..Utils.locomotion_utils import character_state, state2ob, state_to_BodyInfoState
from ..Utils import diff_quat
from ..Utils import pytorch_utils as ptu

import mujoco

#contact_area_prediction
#from scipy.spatial import ConvexHull


class MujocoScene(object):
    def __init__(self, model, data, model_idx=0):
        self.characters = []

        self.mj_model = model
        self.mj_data = data


        self.contact_type = None
        self.self_collision = False

        self.body_info = self
        self.characters = [self, self]

        self.num_body = self.mj_model.nbody
        self.num_muscles = len(self.mj_data.act)

        #config for model change
        self.gravity_axis = 1 #y
        self.root_joint_name = 'pelvis_tx'

        self.init_vel = 1.2

        self.vx = 0
        self.vy = 0
        self.vz = 0

        self.model_idx = model_idx

    def get_name_list(self): #scene.characters[0].body_info.get_name_list
        name_list= []
        num_body = self.num_body
        for i in range(num_body):
            current_body = self.mj_data.body(i)
            #current_body_name = current_body["name"]
            current_body_name = current_body.name
            name_list.append(current_body_name)
        
        return name_list
    
    def get_body_velo(self):
        pass

    def get_body_ang_velo(self):
        pass

    def get_body_pos(self):
        pass

    def get_body_quat(self):
        pass

    def get_muscle_info(self):
        pass

    def damped_simulate(self, count):
        pass

    def random_leg_up(self):
        pass

    def random_root_vel(self, init_pos=None, init_rot=None):
        #random_root_vel = np.random.uniform(0, 1)
        random_root_vel = 0

        root_joint = self.mj_data.joint(self.root_joint_name)
        root_joint.qvel = np.ones(1) * random_root_vel


    def reset_sim(self, init_pos=None, init_rot=None):
        #
        #self.mj_model.opt.gravity = np.zeros(3)
        self.mj_model.opt.gravity = np.array([0., -9.81, 0.])
        if(self.gravity_axis == 2): #ostrich #dog #rajagopal
            self.mj_model.opt.gravity = np.array([0., 0, -9.81])

        #print(self.mj_model.opt.integrator) #default 0
        #mjINT_EULER         = 0,        // semi-implicit Euler
        #mjINT_RK4,                      // 4th-order Runge Kutta
        #mjINT_IMPLICIT,                 // implicit in velocity
        #mjINT_IMPLICITFAST 

        mujoco.mj_resetData(self.mj_model, self.mj_data)


        self.random_root_vel(init_pos, init_rot)
        #self.random_leg_up() #deprecated

        mujoco.mj_step(self.mj_model, self.mj_data)


    def simulate(self):
        mujoco.mj_step(self.mj_model, self.mj_data)
        
    def character_muscle_state(self, version_velocity=False, version_force=False):
        muscle_state = np.array(self.mj_data.actuator_length)
        if(version_velocity == True):
            muscle_state = np.concatenate([self.mj_data.actuator_length, self.mj_data.actuator_velocity])
       
        if(version_force == True):
            f_m_opt = np.array(self.mj_model.actuator_gainprm[:self.num_muscles, 2])
            f_norm = np.abs(self.mj_data.actuator_force) / f_m_opt

            muscle_state = np.concatenate([self.mj_data.actuator_length, self.mj_data.actuator_length, f_norm])

        return muscle_state

    def build_inertia_tensor(self):
        return
        #print(self.mj_model.body_inertia)
        #print(self.mj_model.body_ipos)
        #print(self.mj_model.body_mass)
        #print(xxx)


    def character_state(self): #20, 13
        #base: root local frame
        state_list = []

        base_pos = np.zeros(3)

        #self.build_inertia_tensor()

        for i in range(self.num_body):

            current_state = np.zeros(13, dtype=np.float32)
            current_body = self.mj_data.body(i)

            pos = current_body.xpos #- base_pos
            quat = current_body.xquat
            vel = current_body.cvel[3:6] #center of body-frame
            ang_vel = current_body.cvel[0:3]

            if(True): 
                #y as height
                current_state[0:3] = pos
                
                current_state[3] = quat[1]
                current_state[4] = quat[2]
                current_state[5] = quat[3]
                current_state[6] = quat[0]

                #current_state[3] = quat[0]
                #current_state[4] = quat[1]
                #current_state[5] = quat[2]
                #current_state[6] = quat[3]

                current_state[7:10] = vel
                current_state[10:13] = ang_vel


            state_list.append(current_state)

        if(True):
            state_list[0] = np.array(state_list[1])

        state = np.array(state_list)


        return state


    def state2ob(self, state): # 323, 
        return obs

    def load_character_state(self, state, init_pos=None, init_rot=None): #,set_state=False):
        #return
        self.reset_sim(init_pos, init_rot)
        return

    def save(self):
        pass


class MuscleDynamics(object):
    def __init__(self):
        pass

    def calculateMuscleMassVector(self, l_m_opt, f_m_opt):
        Am = f_m_opt / 350000
        mass = 1060 * l_m_opt * Am
        return mass

    def calculateMuscleActivationHeatRate(self, muscle_mass_vector, muscle_excitation_level_vector):
        #use activation instead of muscle_excitation_level
        def f_A_u(u):
            fraction = 0.5
            half_pi = 0.5 * np.pi
            f_A_u = 40 * fraction * np.sin(half_pi * u) + 133 * (1 - fraction) * (1 - np.cos(half_pi * u))
            return f_A_u

        A = muscle_mass_vector * f_A_u(muscle_excitation_level_vector)
        return A

    def calculateMuscleActivationHeatRateTensor(self, muscle_mass_vector, muscle_excitation_level_vector):
        fraction = 0.5
        half_pi = 0.5 * 3.141592
        f_A_u = 40 * fraction * torch.sin(half_pi * muscle_excitation_level_vector) + 133 * (1 - fraction) * (1 - torch.cos(half_pi * muscle_excitation_level_vector))
        
        A = muscle_mass_vector * f_A_u
        return A

    def calculateMuscleMaintenanceHeatRate(self, muscle_mass_vector, muscle_activation_vector, l_ce_vector):
        def g_l_ce(l_ce): #use np.map

            len_l_ce = len(l_ce)
            g_l_ce_vector = np.zeros(len_l_ce)
            for i in range(len_l_ce):
                current_l_ce = l_ce[i]
                if(current_l_ce >= 0 and current_l_ce <= 0.5): # original : >0 
                    g_l_ce_vector[i] = 0.5
                elif(current_l_ce > 0.5 and current_l_ce <= 1):
                    g_l_ce_vector[i] = current_l_ce
                elif(current_l_ce > 1. and current_l_ce <= 1.5):
                    g_l_ce_vector[i] = -2 * current_l_ce + 3
                elif(current_l_ce > 1.5):
                    g_l_ce_vector[i] = 0.
                else:
                    g_l_ce_vector[i] = 0.
                    #print('helper_g_l_ce: wrong input')
                    #print(xxx)
                return g_l_ce_vector

        def f_M_a(act):
            fraction = 0.5
            half_pi = 0.5 * np.pi
            f_M_a = 74 * fraction * np.sin(half_pi * act) + 111 * (1 - fraction) * (1 - np.cos(half_pi * act))
            return f_M_a

        M = muscle_mass_vector * g_l_ce(l_ce_vector) * f_M_a(muscle_activation_vector)
        return M

    def calculateMuscleMaintenanceHeatRateTensor(self, muscle_mass_vector, muscle_activation_vector, g_l_ce_vector):
        g_l_ce_vector = torch.where((g_l_ce_vector > 0) | (g_l_ce_vector <= 0.5), 0.5, g_l_ce_vector)
        g_l_ce_vector = torch.where((g_l_ce_vector > 1.0) | (g_l_ce_vector <= 1.5), -2 * g_l_ce_vector + 3, g_l_ce_vector)
        g_l_ce_vector = torch.where((g_l_ce_vector > 1.5) | (g_l_ce_vector < 0), 0, g_l_ce_vector)

        fraction = 0.5
        half_pi = 0.5 * 3.141592
        f_M_a = 74 * fraction * torch.sin(half_pi * muscle_activation_vector) + 111 * (1 - fraction) * (1 - torch.cos(half_pi * muscle_activation_vector))

        M = muscle_mass_vector * g_l_ce_vector * f_M_a
        return M


    def calculateMuscleShorteningHeatRate(self, f_mtu_vector, v_ce_vector):
        S = 0.25 * f_mtu_vector * v_ce_vector #* -1
        #S_ = S * -1
        #print('S:', S - S)
        return S

    def calculatePossibleMechanicalWorkRate(self, f_ce_vector, v_ce_vector):
        W = f_ce_vector * v_ce_vector #* -1
        #W_ = W * -1
        #print('W:', W - W_)
        return W

    def init_muscle_parameter_for_tensor(self, l_m_opt=None, l_t_opt=None, f_m_opt=None):
        #size = None
        num_muscles = len(l_m_opt) #120
        #num_batch = -1

        #assert size 512, 120

        self.l_m_opt = ptu.from_numpy(l_m_opt).reshape((-1, num_muscles))
        self.l_t_opt = ptu.from_numpy(l_t_opt) #not used ~ from env (mujoco lm - lmopt)
        self.f_m_opt = ptu.from_numpy(f_m_opt).reshape((-1, num_muscles))

        self.muscle_mass = self.calculateMuscleMassVector(self.l_m_opt, self.f_m_opt)

    def calculateMuscleStateFromRigidStateTensor(self, rigid_state_tensor):
        #path point tensor
        pass


class FakeMotionDataSet(object):
    def __init__(self):
        pass


class MujocoMuscleEnv():
    def __init__(self, **kargs) -> None:
        super(MujocoMuscleEnv, self).__init__()
        self.object_reset(**kargs)
    
    def object_reset(self, **kargs):
        if 'seed' in kargs:
            self.seed(kargs['seed']) # using global_seed    

        self.use_mujoco = kargs['use_mujoco']
        self.mujoco_sim_type = kargs['mujoco_sim_type']
        self.use_leg_collision = kargs['use_leg_collision']
        self.target_mode_trajectory = kargs['posterior_input_change_for_trajectory']
        self.use_model_change = kargs['use_model_change']
        self.use_fullbody_change = kargs['use_fullbody_change']
        self.model_change_mode = kargs['model_change_mode']
        self.model_idx = kargs['model_idx']

        if(self.use_model_change == True):
            self.model_idx = 1 #ostrich
            self.model_config_file_name = kargs['model_config_file_name']
            #self.model_config_num_link = kargs['model_config_num_link']
            self.model_config_num_muscles = kargs['model_config_num_muscles']
            self.model_config_num_dofs = kargs['model_config_num_dofs']
            self.model_config_axis_change = kargs['model_config_axis_change'] #z axis
            self.model_config_root_init_height = kargs['model_config_root_init_height']
            self.model_config_root_joint_name = kargs['model_config_root_joint_name']

        self.root_link_idx = 1 #fullbody, ostrich

        self.use_muscle_state_prediction = kargs['use_muscle_state_prediction']
        self.use_muscle_state_energy_prediction = kargs['use_muscle_state_energy_prediction']
        self.use_muscle_state_energy_prediction_only = kargs['use_muscle_state_energy_prediction_only']

        self.use_random_target_for_latent_ver_energy = kargs['use_random_target_for_latent_ver_energy']
        self.use_random_target_for_latent_ver_double = kargs['use_random_target_for_latent_ver_double']
        self.ver_double_mk1 = kargs['ver_double_mk1']
        self.ver_double_mk2 = kargs['ver_double_mk2']
        self.ver_double_mk3 = kargs['ver_double_mk3']
        self.ver_double_mk4 = kargs['ver_double_mk4']
        self.mk4_mode = kargs['mk4_mode']
        #self.use_random_target_for_latent_ver_triple = kargs['use_random_target_for_latent_ver_triple']

        self.without_vae_version_goal_target_vel_rot = kargs['without_vae_version_goal_target_vel_rot']
        self.without_vae_version_goal_global_vel_rot = kargs['without_vae_version_goal_global_vel_rot']

        self.use_goal_transition = kargs['use_goal_transition']
        self.goal_transition_probability = 0.03 #0.02

        self.use_random_target_vel = kargs['use_random_target_vel']

        if(self.target_mode_trajectory == True):
            self.target_velocity = -1. #fullbody
            self.target_energy = 1.5 

            if(self.without_vae_version_goal_target_vel_rot == True):
                self.target_velocity = 1.0
                self.target_velocity_side = 0.0

        if(self.use_mujoco == True):
            #import mujoco
            self.num_muscles = 120
            self.num_pd = 27
            if(self.use_model_change == True):
                self.num_pd = self.model_config_num_dofs
                #63 for dog

            file_path = './Data/Muscle/'
            file_name = kargs['mujoco_file_name']

            if(self.use_fullbody_change == True):
                file_name = 'Fullbody/chimanoid' #length 1.3x, 0.7x

            file_type = '.xml'

            if(self.use_model_change == True):
                file_name = self.model_config_file_name
                self.num_muscles = kargs['model_config_num_muscles']

            file_full_name = file_path + file_name + file_type
    
            model = mujoco.MjModel.from_xml_path(file_full_name)

            group_right_leg = ['femur', 'tibia', 'fibula',  'talus', 'foot_attachedto_talus_r', 'bofoot',
                            'r_foot1', 'r_foot2', 'r_foot3', 'r_bofoot1', 'r_bofoot2', 'r_bofoot3', 'r_bofoot4', 'r_bofoot5'] #added for capsule foot (25.04.16)
            group_left_leg = ['l_femur', 'l_tibia', 'l_fibula', 'l_talus', 'l_foot_attachedto_talus_l', 'l_bofoot',
                            'l_foot1', 'l_foot2', 'l_foot3', 'l_bofoot1', 'l_bofoot2', 'l_bofoot3', 'l_bofoot4', 'l_bofoot5'] #added for capsule foot

            if(False): #ver leg
            #if(self.use_fullbody_change == True):
                group_right_leg.append('r_hand1')
                group_right_leg.append('r_hand2')
                group_right_leg.append('r_hand3')
                group_left_leg.append('l_hand1')
                group_left_leg.append('l_hand2')
                group_left_leg.append('l_hand3')


            render_only_hand_r = []
            render_only_hand_l = []

            #if(False):
            #if(True): #revision
            if(self.use_fullbody_change == True): #ver limb
                group_right_arm = ['humerus', 'ulna', 'radius', 'r_hand1', 'r_hand2', 'r_hand3']

                group_left_arm = ['humerus_l', 'ulna_l', 'radius_l', 'l_hand1', 'l_hand2', 'l_hand3']

                group_right_leg = ['femur', 'tibia', 'fibula',  'talus', 'foot_attachedto_talus_r', 'bofoot',
                                'r_foot1', 'r_foot2', 'r_foot3', 'r_bofoot1', 'r_bofoot2', 'r_bofoot3', 'r_bofoot4', 'r_bofoot5']

                group_left_leg = ['l_femur', 'l_tibia', 'l_fibula', 'l_talus', 'l_foot_attachedto_talus_l', 'l_bofoot',
                                'l_foot1', 'l_foot2', 'l_foot3', 'l_bofoot1', 'l_bofoot2', 'l_bofoot3', 'l_bofoot4', 'l_bofoot5']

                if(False):
                    ["pisiform", "lunate"    ,"scaphoid"  ,"triquetrum","hamate"    ,"capitate"  ,"trapezoid" ,"trapezium" ,"1mc" ,"2mc"       ,"3mc"       ,"4mc"       ,"5mc"       ,"thumbprox" ,"thumbdist" ,"2proxph"   ,"2midph"    ,"2distph"   ,"3proxph"   ,"3midph"    ,"3distph"   ,"4proxph"   ,"4midph"   ,"4distph"   ,"5proxph"   ,"5midph"   ,"5distph"]


            if(self.use_model_change == True and self.model_idx == 1): #ostrich
                group_right_leg = ['r_femur', 'r_tibiotarsus', 'r_tarsometatarsus', 'r_phalanx1', 'r_phalanx2','r_phalanx3','r_phalanx4', 'r_thumb']
                group_left_leg = ['l_femur', 'l_tibiotarsus', 'l_tarsometatarsus', 'l_phalanx1', 'l_phalanx2','l_phalanx3','l_phalanx4', 'l_thumb']


            #no collision
            #can be used for all model_idx (basic option for collision check)
            #if(self.use_model_change == True or self.use_fullbody_change == True): #all model self collision or not
            if(True):
            #if(self.use_model_change == True): #ostrich : no collision
                #no self collision
                for i in range(model.ngeom):
                    current_geom = model.geom(i)
                    if(i == 0): 
                        current_geom.contype = 0
                        current_geom.conaffinity = 1
                    else:
                        current_geom.contype = 1
                        current_geom.conaffinity = 0

                    #group_right_leg = ['femur', 'tibia', 'fibula',  'talus', 'foot_attachedto_talus_r', 'bofoot']
                    #group_left_leg = ['l_femur', 'l_tibia', 'l_fibula', 'l_talus', 'l_foot_attachedto_talus_l', 'l_bofoot']

                    #foot render only
                    if(current_geom.name == 'foot_attachedto_talus_r_v' or current_geom.name == 'foot_attachedto_talus_l_v' or current_geom.name == 'bofoot_v' or current_geom.name == 'l_bofoot_v'):
                        current_geom.contype = 0
                        current_geom.conaffinity = 0

                #all self collsion
                #for i in range(model.ngeom):
                #    current_geom = model.geom(i)
                #    if(i == 0): 
                #        current_geom.contype = 1
                #        current_geom.conaffinity = 1
                #    else:
                #        current_geom.contype = 1
                #        current_geom.conaffinity = 1

                    if(current_geom.name == "ball"): #collision with ground, fullbody
                        current_geom.contype = 1 
                        current_geom.conaffinity = 1


            if(self.use_model_change == False and self.use_fullbody_change == False): #fullbody: leg collision
            #if(self.model_idx == 0): # 
            #if(self.model_idx == 0 or self.model_idx == 3): #fullbody, rajagopal
                for i in range(model.ngeom): #84
                    #print(model.ngeom)
                    #print(xxx)
                    current_geom = model.geom(i)
                    #print(i, current_geom.name)

                    if(self.use_leg_collision == False): #ver1 ~ disable self collision
                        if(i == 0):
                            current_geom.contype = 0
                            current_geom.conaffinity = 1
                        else:
                            current_geom.contype = 1
                            current_geom.conaffinity = 0
                        #print(i, current_geom.name, current_geom.contype, current_geom.conaffinity)

                    if(self.use_leg_collision == True):
                        if(i == 0):
                            current_geom.contype = 1
                            current_geom.conaffinity = 1
                        else:
                            current_geom.contype = 0
                            current_geom.conaffinity = 0
    
                        current_geom_name = current_geom.name
    
                        for item in group_right_leg:
                            if(current_geom_name == item):
                                current_geom.contype = 1
                                current_geom.conaffinity = 0
                                break
    
                        for item in group_left_leg:
                            if(current_geom_name == item):
                                current_geom.contype = 0
                                current_geom.conaffinity = 1
                                break

                        if(current_geom.name == "ball"): #collision with ground, fullbody
                            current_geom.contype = 1 
                            current_geom.conaffinity = 1
                    
                    if(current_geom.name == 'foot_attachedto_talus_r_v' or current_geom.name == 'foot_attachedto_talus_l_v' or current_geom.name == 'bofoot_v' or current_geom.name == 'l_bofoot_v'):
                        current_geom.contype = 0
                        current_geom.conaffinity = 0


            #if(True): #revision
            if(self.use_fullbody_change == True): #visual hand + collision 
                hand_name_list = ["pisiform" ,"lunate","scaphoid" ,"triquetrum","hamate","capitate" ,"trapezoid","trapezium"
                                ,"1mc","2mc","3mc","4mc","5mc","thumbprox","thumbdist","2proxph","2midph","2distph" ,"3proxph" ,"3midph" ,"3distph" ,"4proxph" ,"4midph" ,"4distph","5proxph","5midph","5distph"
                                ,"pisiform_l" ,"lunate_l","scaphoid_l" ,"triquetrum_l","hamate_l","capitate_l" ,"trapezoid_l","trapezium_l",
                                "1mc_l","2mc_l","3mc_l","4mc_l","5mc_l","thumbprox_l","thumbdist_l","2proxph_l","2midph_l","2distph_l" ,"3proxph_l" ,"3midph_l" ,"3distph_l" ,"4proxph_l" ,"4midph_l" ,"4distph_l",
                                "5proxph_l","5midph_l","5distph_l"]

                for i in range(model.ngeom): #84
                    current_geom = model.geom(i)
                
                    for item in hand_name_list:
                        if(current_geom.name == item):
                            current_geom.contype = 0
                            current_geom.conaffinity = 0
                            #print(current_geom.name)
                            break


            data = mujoco.MjData(model)

            self.scene = MujocoScene(model, data, self.model_idx)

            self.dummy_data = None
            if(True):
            #if(self.use_random_target_for_latent_ver_double == True): #inference
                self.dummy_data = mujoco.MjData(model)
                mujoco.mj_resetData(model, self.dummy_data)
                mujoco.mj_step(model, self.dummy_data)


            if(self.use_model_change == True):
                if(self.model_config_axis_change == True):
                    self.scene.gravity_axis = 2
                self.scene.root_joint_name = self.model_config_root_joint_name


            self.init_rigid() #inertia info
            if(self.mujoco_sim_type == 'muscle'):
                
                self.init_muscle()
                self.muscle_dynamics = MuscleDynamics()
                self.muscle_dynamics.init_muscle_parameter_for_tensor(l_m_opt=self.l0, l_t_opt=self.l_t_opt, f_m_opt=self.f_m_opt)

                self.energy_consumption_list = []

        self.fps = kargs['env_fps']
        self.dt = 1/self.fps
        self.substep = kargs['env_substep']

        if(self.use_mujoco == True):
            #1/500
            self.dt = 1/33
            self.substep = 15 #500 0.002

        # terminate condition
        self.min_length = kargs['env_min_length']
        self.max_length = kargs['env_max_length']
        self.err_threshod = kargs['env_err_threshod']
        self.err_length = kargs['env_err_length']

        name_list = self.sim_character.body_info.get_name_list()

        #mujoco
        #['world', 'pelvis', 'femur_r', 'tibia_r', 'talus_r', 'toes_r', 'femur_l', 'tibia_l', 'talus_l', 'toes_l', 
        #'thorax_dummy', 'thorax', 'humerus', 'ulna', 'radius', 'hand_r', 'humerus_l', 'ulna_l', 'radius_l', 'hand_l']

        self.head_idx = 1 #PELVIS FOR FULLBODY RAJAGOPAL OSTRICH FULLBODY(DART)

        self.balance = not kargs['env_no_balance']
        self.random_count = 600


        #not used
        self.motion_data = None 
        self.frame_num = 5227
        self.init_index = [0] #[0, 1, ... 5223]

        if self.balance:
            import numpy as np
            self.val = np.zeros(self.frame_num)    
            self.update_p()
        return 

        
    def init_rigid(self):
        if(True): #for inertia
            self.arr_inertia = np.zeros((20, 3, 3))
            self.arr_mass = np.zeros((20, 1))
            self.arr_com = np.zeros((20, 3))

            for i in range(20):
                self.arr_inertia[i, 0, 0] = self.scene.mj_model.body_inertia[i, 0]
                self.arr_inertia[i, 1, 1] = self.scene.mj_model.body_inertia[i, 1]
                self.arr_inertia[i, 2, 2] = self.scene.mj_model.body_inertia[i, 2]
                self.arr_mass[i] = self.scene.mj_model.body_mass[i]
                self.arr_com[i] = self.scene.mj_model.body_ipos[i]

    def init_muscle(self):

        num_muscles = self.num_muscles

        self.f_m_opt = np.array(self.scene.mj_model.actuator_gainprm[:num_muscles, 2])

        self.l0 = np.array(self.scene.mj_model.actuator_length0)[:num_muscles]
        self.l_m_opt = (np.array(self.scene.mj_model.actuator_lengthrange[:num_muscles,0]) + np.array(self.scene.mj_model.actuator_lengthrange[:num_muscles, 1])) / 2
        self.l_t_opt = self.l0 - self.l_m_opt


        #use_other_model
        for i in range(num_muscles):
            #print(i, self.l_m_opt[i], self.l_t_opt[i])
            if(self.l_t_opt[i] < 0):
                self.l_t_opt[i] = 0
                #print('negative')

    def get_energy_consumption(self, mode_debug=False, mode_multi=False, mode_debug2=False):
        mode_debug = True
        #self.energy_debug = False
        self.energy_debug = mode_debug

        num_muscles = self.num_muscles
        muscle_activation = np.array(self.scene.mj_data.act)
        muscle_excitation = muscle_activation

        muscle_mass = self.muscle_dynamics.calculateMuscleMassVector(self.l_m_opt, self.f_m_opt)

        l_mtu = np.array(self.scene.mj_data.actuator_length)[:num_muscles]

        l_m = l_mtu
        #only fiber length or not
        l_m = l_mtu - self.l_t_opt 

        if(self.energy_debug == True):
            l_m = l_mtu

        l_ce = l_m
        #normalize or not
        l_ce = l_m / self.l_m_opt
        if(self.energy_debug == True):
            l_ce = l_m / self.l0

        f_ce = None
        f_pe = None
        f_mtu = np.abs(np.array(self.scene.mj_data.actuator_force)[:num_muscles]) #241218
        f_ce = f_mtu

        if(True): 
            f_mtu = np.abs(f_mtu)
            f_ce = np.abs(f_ce)
            

        if(self.energy_debug == False): 
            f_mtu = f_mtu / self.f_m_opt
            f_ce = f_mtu
            #f_ce = f_mtu - f_pe

        v_ce = np.array(self.scene.mj_data.actuator_velocity)[:num_muscles] #* self.l0

        for i in range(num_muscles):
            if(v_ce[i] >= 0):
                v_ce[i] = 0
            else:
                v_ce[i] = v_ce[i] * -1

        A = self.muscle_dynamics.calculateMuscleActivationHeatRate(muscle_mass, muscle_excitation)
        M = self.muscle_dynamics.calculateMuscleMaintenanceHeatRate(muscle_mass, muscle_activation, l_ce)
        S = self.muscle_dynamics.calculateMuscleShorteningHeatRate(f_mtu, v_ce)
        W = self.muscle_dynamics.calculatePossibleMechanicalWorkRate(f_ce, v_ce)

        if(mode_multi == True):
            return (A + M + S + W)

        return np.sum(A + M + S + W)

    @property
    def stastics(self):
        return None
        #return self.motion_data.stastics
    
    @property
    def sim_character(self):
        return self.scene.characters[0]

    @property
    def ref_character(self):
        return self.scene.characters[1]

    @staticmethod
    def seed( seed):
        pass
        #SetInitSeed(seed) #remove_ode
    
    @staticmethod
    def add_specific_args(arg_parser):
        arg_parser.add_argument("--env_contact_type", type=int, default=0, help="contact type, 0 for LCP and 1 for maxforce")
        arg_parser.add_argument("--env_min_length", type=int, default=26, help="episode won't terminate if length is less than this")
        arg_parser.add_argument("--env_max_length", type=int, default=512, help="episode will terminate if length reach this")
        arg_parser.add_argument("--env_err_threshod", type = float, default = 0.5, help="height error threshod between simulated and tracking character")
        arg_parser.add_argument("--env_err_length", type = int, default = 20, help="episode will terminate if error accumulated ")
        arg_parser.add_argument("--env_fps", type = int, default = 20, help="fps of control policy")
        arg_parser.add_argument("--env_substep", type = int, default = 6, help="substeps for simulation")
        arg_parser.add_argument("--env_no_balance", default = False, help="whether to use distribution balance when choose initial state", action = 'store_true')
        return arg_parser
    
    def update_and_get_target(self):
        self.step_cur_frame()
        return 
    
    @staticmethod
    def isnotfinite(arr):
        res = np.isfinite(arr)
        return not np.all(res)

    def cal_done(self, state, obs):
        #return 0
        height = state[...,self.head_idx,1]

        if(self.target_mode_trajectory == True):
            target_height = 0.94
            if(self.use_fullbody_change == True):
                target_height = 0.8 #chimanoid

            if(self.use_model_change == True):
                target_height = self.model_config_root_init_height
                if(self.model_config_axis_change == True):
                    height = state[..., self.head_idx, 2] 


        if abs(height - target_height) > self.err_threshod:
            self.done_cnt +=1

        else:
            self.done_cnt = max(0, self.done_cnt - 1)
        
        if self.isnotfinite(state):
            return 2

        if np.any( np.abs(obs) > 50): #75 100
            return 2

        if self.step_cnt >= self.min_length:
            if self.done_cnt >= self.err_length:
                return 2
            if self.step_cnt >= self.max_length:
                return 1
        return 0
    
    def update_val(self, done, rwd, frame_num):
        return
        '''
        tmp_val = self.val / 2
        last_i = 0
        for i in range(frame_num.shape[0]):
            if done[i] !=0:
                tmp_val[frame_num[i]] = rwd[i] if done[i] == 1 else 0
                for j in reversed(range(last_i, i)):
                    tmp_val[frame_num[j]] = 0.95*tmp_val[frame_num[j+1]] + rwd[j]
                last_i = i
        self.val = 0.9 * self.val + 0.1 * tmp_val
        '''
        
        
    def update_p(self):
        return
        '''
        self.p = 1/ self.val.clip(min = 0.01)
        self.p_init = self.p[self.init_index]
        self.p_init /= np.sum(self.p_init)
        '''
        
    def get_info(self):
        return {
            'frame_num': self.counter
        }

    def get_target_energy_height_pose(self, x=0):
        self.target_height = None
        #self.target_energy = np.random.uniform(0.15, 0.21) #0.25

        if(self.model_idx == 1):
            pass
            #self.target_energy = np.random.uniform(0.17, 0.24) #0.3?
            #self.target_energy = 0.18
            if(False):
                self.target_energy = 0.18
        if(self.use_fullbody_change == True):
            pass
            #self.target_energy = np.random.uniform(0.15, 0.21) 

        #self.target_energy = 0.15 #0.17
        #self.target_velocity = np.random.uniform(0, 3.0)
        #self.target_velocity_side = self.target_velocity * -1

        if(self.use_random_target_for_latent_ver_double == True):
            joint_range_min = np.array([-1.5, -1.5, -2, -2])
            joint_range_max = np.array([1.5, 1.5, 0.17, 0.17])

            
            if(self.ver_double_mk1 == True):
                joint_range_min = np.array([-1.5, -2, -1.5])
                joint_range_max = np.array([1.5, 0.17, 1.5])

                if(self.model_idx == 1): #ostrich
                    joint_range_min = np.array([-0.3, 0, -2.3])
                    joint_range_max = np.array([1.5, 2, 0])
                    #r_hip_y default 0, forward -0.3 backward 1.5
                    #r_knee_y default 0, backward 2.0
                    #r_ankle_y default 0, forward -2.3
                    #NONE r_mtp_y

                    if(False): #change (actually, it is original ostrichRL model)
                        joint_range_min = np.array([-1.5, 0, -2.3])
                        joint_range_max = np.array([0.3, 2, 0])


            if(self.ver_double_mk2 == True):
                joint_range_min = np.array([-1.5, -2, -1.5, 0])
                joint_range_max = np.array([1.5, 0.17, 1.5, 2.])

                #change shoulder
                if(False):
                    joint_range_min = np.array([-1.5, -2, -0.75, 0])
                    joint_range_max = np.array([1.5, 0.17, 0.75, 2.])
                    #
                    self.temp_angle_test = np.random.uniform(-1.5, 1.5, size=2)[0]

            if(self.ver_double_mk3 == True):
                if(True):
                #if(self.model_idx == 0 and self.use_fullbody_change == True):
                    joint_range_min = np.array([-1.5])
                    joint_range_max = np.array([1.5])

            if(self.ver_double_mk4 == True):
                #joint_range_min = np.array([-1.5, -2, -1.5])
                #joint_range_max = np.array([1.5, 0.17, 1.5])

                joint_range_min = np.array([-1.5]) #shoulder
                joint_range_max = np.array([1.5]) #shoulder


                joint_range_min = np.array([0]) #for chimanoid, test

                if(self.model_idx == 1): #ostrich
                    joint_range_min = np.array([-0.3, 0, -2.3])
                    joint_range_max = np.array([1.5, 2, 0])
                    #r_hip_y default 0, forward -0.3 backward 1.5
                    #r_knee_y default 0, backward 2.0
                    #r_ankle_y default 0, forward -2.3
                    #NONE r_mtp_y

                if(True):
                    #mk4_mode_idx = 0
                    joint_range_min = np.array([joint_range_min[self.mk4_mode]]) #0 for chimanoid
                    joint_range_max = np.array([joint_range_max[self.mk4_mode]])
                                
            num_target_pose = len(joint_range_min)

            joint_range_gap = joint_range_max - joint_range_min
            rand_angle = np.random.uniform(0, 1, size=num_target_pose)

            #rand_angle[0] = 0.9 #revision
            self.temp_angle = joint_range_min + rand_angle * joint_range_gap

            #print(self.target_energy, self.temp_angle)

            if(x == 0 or x == 100):
                self.temp_angle = np.zeros(num_target_pose)

            self.target_pose = np.zeros(6 * num_target_pose)

            for tp in range(num_target_pose):
                change = False
                #if(tp == 2): #change shoulder
                #    change = True
                self.target_pose[6*tp : 6*tp+6] = self.angle_to_rotation_1d(self.temp_angle[tp], change)


    def get_target(self):
        def helper_norm(v1, v2):
            threshold = 0.1
            magnitude = np.sqrt(v1 * v1 + v2 * v2)
            if(magnitude < threshold):
                return 0, 0, 0
            else:
                return (v1 / magnitude), (v2 / magnitude), magnitude
            #return v1_norm, v2_norm, v_magnitude

        if(self.without_vae_version_goal_target_vel_rot == True): #latent


            target = np.zeros(4)
            target[0] = self.target_velocity
            target[1] = self.target_velocity_side
            target[2] = self.target_velocity - self.state[0, 7]
            target[3] = self.target_velocity_side - self.state[0, 9]
            if(self.model_idx == 1): #ostrich
                target[3] = self.target_velocity_side - self.state[0, 8]

            if(self.use_random_target_for_latent_ver_energy == True):
                new_target = np.zeros(6)
                new_target[0:4] = target
                new_target[4] = self.target_energy
                new_target[5] = self.target_energy - np.mean(self.energy)
                return new_target

            if(self.use_random_target_for_latent_ver_double == True): #target_vel_rot == True
                new_target = np.zeros(54) # 6 + 24 + 24
                new_target[0:4] = target
                new_target[4] = self.target_energy
                new_target[5] = self.target_energy - np.mean(self.energy)
                
                if(self.ver_double_mk1 == False and self.ver_double_mk2 == False and self.ver_double_mk3 == False and self.ver_double_mk4 == False):
                    new_target[6:30] = self.target_pose
    
                    new_target[30:36] = self.target_pose[0:6] - self.observation[72:78]
                    new_target[36:42] = self.target_pose[6:12] - self.observation[96:102] #hip
                    new_target[42:48] = self.target_pose[12:18] - self.observation[78:84]
                    new_target[48:54] = self.target_pose[18:24] - self.observation[102:108] #knee


                if(self.ver_double_mk1 == True):
                    #target & energy -> 6
                    #femur -> 6 6 6
                    #knee -> 6 6 6
                    #talus -> 6 6 6 
                    new_target = np.zeros(60)
                    new_target[0:4] = target
                    new_target[4] = self.target_energy
                    new_target[5] = self.target_energy - np.mean(self.energy)

                    new_target[6:24] = self.target_pose
                    #femur, knee, talus

                    new_target[24:30] = self.target_pose[0:6] - self.observation[72:78]
                    new_target[30:36] = self.target_pose[0:6] - self.observation[96:102] #femur
                    new_target[36:42] = self.target_pose[6:12] - self.observation[78:84]
                    new_target[42:48] = self.target_pose[6:12] - self.observation[102:108] #knee
                    new_target[48:54] = self.target_pose[12:18] - self.observation[84:90]
                    new_target[54:60] = self.target_pose[12:18] - self.observation[108:114] #talus

                    if(self.model_idx == 1): #ostrich
                        new_target[24:30] = self.target_pose[0:6] - self.observation[96+6*2:96+6*3] #2
                        new_target[30:36] = self.target_pose[0:6] - self.observation[96+6*6:96+6*7] #6
                        new_target[36:42] = self.target_pose[6:12] - self.observation[96+6*3:96+6*4] #3
                        new_target[42:48] = self.target_pose[6:12] - self.observation[96+6*7:96+6*8] #7
                        new_target[48:54] = self.target_pose[12:18] - self.observation[96+6*4:96+6*5] #4
                        new_target[54:60] = self.target_pose[12:18] - self.observation[96+6*8:96+6*9] #8

                        #pos 0:96
                        #rot 96:288

                if(self.ver_double_mk2 == True):
                    #target & energy -> 6
                    #femur -> 6 6 6 
                    #knee -> 6 6 6
                    #arm -> 6 6 6
                    #arm -> 6 6 6 
                    new_target = np.zeros(78)
                    new_target[0:4] = target
                    new_target[4] = self.target_energy
                    new_target[5] = self.target_energy - np.mean(self.energy)
                    new_target[6:30] = self.target_pose

                    #femur, knee, arm1, arm2
                    new_target[30:36] = self.target_pose[0:6] - self.observation[72:78]
                    new_target[36:42] = self.target_pose[0:6] - self.observation[96:102] #femur
                    
                    new_target[42:48] = self.target_pose[6:12] - self.observation[78:84]
                    new_target[48:54] = self.target_pose[6:12] - self.observation[102:108] #knee

                    new_target[54:60] = self.target_pose[12:18] - self.observation[132:138] #132~138 #156~162
                    new_target[60:66] = self.target_pose[12:18] - self.observation[156:162] #shoulder

                    new_target[66:72] = self.target_pose[18:24] - self.observation[144:150] 
                    new_target[72:78] = self.target_pose[18:24] - self.observation[168:174] #elbow(ulna X -> radius O)

                if(self.ver_double_mk3 == True):
                    if(True):
                    #if(self.use_fullbody_change == True):
                        new_target = np.zeros(18)
                        new_target[0:4] = target
                        new_target[4] = self.target_energy
                        new_target[5] = self.target_energy - np.mean(self.energy)
                        new_target[6:12] = self.target_pose
                        #new_target[12:18] = self.target_pose[0:6] - self.observation[120:126] 
                        new_target[12:18] = self.target_pose[0:6] - self.observation[126:132]

                if(self.ver_double_mk4 == True):
                    #target & energy -> 6
                    #femur -> 6 6 6
                    #knee -> 6 6 6
                    #talus -> 6 6 6 
                    new_target = np.zeros(24)
                    new_target[0:4] = target
                    new_target[4] = self.target_energy
                    new_target[5] = self.target_energy - np.mean(self.energy)

                    new_target[6:12] = self.target_pose
                    #femur, knee, talus

                    target_idx_left_begin = None
                    target_idx_left_end = None
                    target_idx_right_begin = None
                    target_idx_right_end = None

                    new_target[12:18] = self.target_pose[0:6] - self.observation[132:138]
                    new_target[18:24] = self.target_pose[0:6] - self.observation[156:162] #mk4_mode 1

                    if(self.model_idx == 1):
                        #2-3-4-5?
                        #6-7-8-9?
                        if(self.mk4_mode == 0):
                            new_target[12:18] = self.target_pose[0:6] - self.observation[96+6*2:96+6*3]
                            new_target[18:24] = self.target_pose[0:6] - self.observation[96+6*6:96+6*7]

                        elif(self.mk4_mode == 1):
                            new_target[12:18] = self.target_pose[0:6] - self.observation[96+6*3:96+6*4]
                            new_target[18:24] = self.target_pose[0:6] - self.observation[96+6*7:96+6*8]

                        elif(self.mk4_mode == 2):
                            new_target[12:18] = self.target_pose[0:6] - self.observation[96+6*4:96+6*5]
                            new_target[18:24] = self.target_pose[0:6] - self.observation[96+6*8:96+6*9]

                    if(False):
                        new_target[24:30] = self.target_pose[0:6] - self.observation[72:78]
                        new_target[30:36] = self.target_pose[0:6] - self.observation[96:102] #femur
                        new_target[36:42] = self.target_pose[6:12] - self.observation[78:84]
                        new_target[42:48] = self.target_pose[6:12] - self.observation[102:108] #knee
                        new_target[48:54] = self.target_pose[12:18] - self.observation[84:90]
                        new_target[54:60] = self.target_pose[12:18] - self.observation[108:114] #talus
    
                        if(self.model_idx == 1): #ostrich
                            new_target[24:30] = self.target_pose[0:6] - self.observation[96+6*2:96+6*3] #2
                            new_target[30:36] = self.target_pose[0:6] - self.observation[96+6*6:96+6*7] #6
                            new_target[36:42] = self.target_pose[6:12] - self.observation[96+6*3:96+6*4] #3
                            new_target[42:48] = self.target_pose[6:12] - self.observation[96+6*7:96+6*8] #7
                            new_target[48:54] = self.target_pose[12:18] - self.observation[96+6*4:96+6*5] #4
                            new_target[54:60] = self.target_pose[12:18] - self.observation[96+6*8:96+6*9] #8

                return new_target

            return target

        if(True):
            if(self.without_vae_version_goal_global_vel_rot == True):
                if(self.model_idx == 0):
                    target = np.zeros(8)

                    rot_target = np.zeros(3)
                    rot_target[0] = self.target_velocity
                    rot_target[2] = self.target_velocity_side

                    rot_target = rot_target / (np.linalg.norm(rot_target) + 0.01)

                    rot_current = np.zeros(3)
                    quat_x = self.state[0, 3]
                    quat_y = self.state[0, 4]
                    quat_z = self.state[0, 5]
                    quat_w = self.state[0, 6]
                    rot_current[0] = 1 - 2 * (quat_y * quat_y + quat_z * quat_z)
                    rot_current[2] = 2 * (quat_x * quat_z - quat_w * quat_y)

                    rot_current = rot_current / (np.linalg.norm(rot_current) + 0.01)

                    target[0] = rot_target[0]
                    target[1] = rot_target[2]
                    target[2] = rot_target[0] - rot_current[0]
                    target[3] = rot_target[2] - rot_current[2]
                    target[4] = self.global_velocity
                    target[5] = self.global_velocity_side
                    target[6] = self.global_velocity - self.state[0, 7]
                    target[7] = self.global_velocity_side - self.state[0, 9]

                    return target

        if(self.target_mode_trajectory == True):
            #ver: vel, upvec, muscle
            num_muscles = self.num_muscles

            target = np.zeros(6 + num_muscles * 2)
            target[0:6] = np.array([self.target_velocity, 0, 0, 0, 1, 0])
            target[6: 6 + num_muscles] = self.scene.mj_data.actuator_length[0:num_muscles]
            target[6 + num_muscles:] = self.scene.mj_data.actuator_velocity[0:num_muscles]

            return target

    def step_counter(self, random = False, align = False):
        self.counter += 1

    def align_current(self):
        return
        #self.initial_root_pos = self.state[0][0:3]
        #self.initial_root_quat = self.state[0][3:7]
        #self.initial_root_rotmat = diff_quat.quat_to_matrix(ptu.from_numpy(self.initial_root_quat))

    def align_target(self):
        return

    def helper_rotation_z(self, angle):
        res = np.identity(3)
        res[0, 0] = np.cos(angle)
        res[0, 1] = -np.sin(angle)
        res[1, 0] = np.sin(angle)
        res[1, 1] = np.cos(angle)
        return res

    def helper_rotation_x(self, angle):
        res = np.identity(3)
        res[1, 1] = np.cos(angle)
        res[1, 2] = -np.sin(angle)
        res[2, 1] = np.sin(angle)
        res[2, 2] = np.cos(angle)
        return res

    def helper_rotation_y(self, angle):
        res = np.identity(3)
        res[0, 0] = np.cos(angle)
        res[0, 2] = np.sin(angle)
        res[2, 0] = -np.sin(angle)
        res[2, 2] = np.cos(angle)
        return res

    def angle_to_rotation_1d(self, a1, change=False): #elbow, knee, ankle
        res = self.helper_rotation_z(a1)

        if(self.model_idx == 1): #ostrich
            res = self.helper_rotation_y(a1)

        #return res
        return res[:, 0:2].reshape((-1))

    def angle_to_rotation_3d(self, ar): #hip, ankle, shoulder, hand
        a1 = ar[0]
        a2 = ar[1]
        a3 = ar[2]
        res = self.helper_rotation_z(a1) @ self.helper_rotation_x(a2) @ self.helper_rotation_y(a3)
        #return res
        return res[:, 0:2].reshape((-1))

    def helper_angle_to_rotation(self, angle_array):
        #root

        res = np.zeros(6 * 16) #14 + 2 (radius)
        res[0:6] = self.angle_to_rotation_3d(angle_array[0:3]) 
        res[6:12] = self.angle_to_rotation_1d(angle_array[3]) 
        res[12:18] = self.angle_to_rotation_3d(angle_array[4:7]) 
        res[18:24] = self.angle_to_rotation_1d(angle_array[7]) 

        res[24:30] = self.angle_to_rotation_3d(angle_array[8:11])
        res[30:36] = self.angle_to_rotation_1d(angle_array[11])
        res[36:42] = self.angle_to_rotation_3d(angle_array[12:15])
        res[42:48] = self.angle_to_rotation_1d(angle_array[15])

        #thorax1
        #thorax2

        res[48:54] = self.angle_to_rotation_3d(angle_array[16:19])
        res[54:60] = self.angle_to_rotation_1d(angle_array[19])
        res[60:66] = res[54:60] #radius
        res[66:72] = self.angle_to_rotation_3d(angle_array[20:23])

        res[72:78] = self.angle_to_rotation_3d(angle_array[23:26])
        res[78:84] = self.angle_to_rotation_1d(angle_array[26])
        res[84:90] = res[78:84] #radius
        res[90:96] = self.angle_to_rotation_3d(angle_array[27:30])

        return res

    def helper_pose_target(self, pose):

        target_pose_array = np.array([
            [1, 0, 0, 1], #0
            [ 0.9987502,  -0.04997917,  0.04997917,  0.9987502 ], 
            [ 0.9950042,  -0.09983341,  0.09983341,  0.9950042 ], #0.1
            [ 0.9887711,  -0.14943813 , 0.14943813,  0.9887711           ],
            [ 0.9800666,  -0.19866933,  0.19866933,  0.9800666           ], #0.2
            [ 0.9689125,  -0.24740396,  0.24740396,  0.9689125           ], 
            [ 0.95533645, -0.2955202,   0.2955202 ,  0.95533645          ], #0.3
            [ 0.9393727, -0.3428978,  0.3428978 , 0.9393727    ],
            [ 0.92106104, -0.38941833,  0.38941833,  0.92106104          ], #0.4
            [ 0.9004471,  -0.43496555,  0.43496555,  0.9004471           ],
            [ 0.87758255, -0.47942555,  0.47942555,  0.87758255          ], #0.5
            ])
        #11, 4

        final_pose = np.zeros(16)
        for i in range(4):
            new_pose = np.zeros(4)
            if(pose[i] == 0):
                new_pose = target_pose_array[0]
            elif(pose[i] == 0.05 or pose[i] == -0.05):
                new_pose = target_pose_array[1]
            elif(pose[i] == 0.1 or pose[i] == -0.1):
                new_pose = target_pose_array[2]
            elif(pose[i] == 0.15 or pose[i] == -0.15):
                new_pose = target_pose_array[3]
            elif(pose[i] == 0.2 or pose[i] == -0.2):
                new_pose = target_pose_array[4]
            elif(pose[i] == 0.25 or pose[i] == -0.25):
                new_pose = target_pose_array[5]
            elif(pose[i] == 0.3 or pose[i] == -0.3):
                new_pose = target_pose_array[6]
            elif(pose[i] == 0.35 or pose[i] == -0.35):
                new_pose = target_pose_array[7]
            elif(pose[i] == 0.4 or pose[i] == -0.4):
                new_pose = target_pose_array[8]
            elif(pose[i] == 0.45 or pose[i] == -0.45):
                new_pose = target_pose_array[9]
            elif(pose[i] == 0.5 or pose[i] == -0.5):
                new_pose = target_pose_array[10]

            if(pose[i] < 0):
                new_pose[1] = new_pose[1] * -1
                new_pose[2] = new_pose[2] * -1

            final_pose[i*4 : i*4 + 4] = new_pose

        return final_pose
        
    def reset(self, frame = -1, set_state = True ):
        #reset enviroment

        self.before_goal = np.zeros(2)
        self.before_l_m = np.zeros(120)

        self.energy_pred = 0 #for inference

        self.contact_state = np.zeros(2) #contact flag 2d / contact force 2d / contact area 2d
        self.is_contact_left = False
        self.is_contact_right = False
        self.left_contact_point_list = []
        self.right_contact_point_list = []

        self.accumulated_height = 0
        self.accumulated_velocity = 0
        self.accumulated_energy = 0
        self.accumulated_distance = 0
        self.x_pos_before = 0
        self.accumulated_target_pose_diff = 0

        if(self.use_random_target_vel == True):
            #import numpy as np

            self.target_velocity = np.random.uniform(1.0, 2.0)

            if(self.without_vae_version_goal_global_vel_rot == True):
                self.target_velocity = np.random.uniform(-1, 1)
                self.target_velocity_side = np.random.uniform(-1, 1) #root direction

                self.global_velocity = np.random.uniform(-3, 3)
                self.global_velocity_side = np.random.uniform(-3, 3) #root 


                if(False): #revision
                    self.global_velocity = -3.5
                    self.global_velocity_side = 0 

                    self.target_velocity = 0
                    self.target_velocity_side = -1.2 #side walk

                    #self.target_velocity = 1.2
                    #self.target_velocity_side = 0 #backward walk

                if(self.use_random_target_for_latent_ver_energy == True):
                    self.target_energy = np.random.uniform(0.15, 0.21)

                if(self.use_random_target_for_latent_ver_double == True): #femur + knee
                    self.get_target_energy_height_pose()
                    #self.target_energy = 0.18

            if(self.without_vae_version_goal_target_vel_rot == True): #goal_target_vel_rot
                #self.target_velocity = np.random.uniform(0, 1.0)
                #self.target_velocity_side = np.random.uniform(-1.5, 1.5)
                self.target_velocity = np.random.uniform(-3.0, 3.0) #1.0
                self.target_velocity_side = np.random.uniform(-3.0, 3.0) #1.0

                if(self.use_random_target_for_latent_ver_energy == True):
                    self.target_energy = np.random.uniform(0.15, 0.21)
                    #self.target_energy = np.random.uniform(0.15, 0.25)


                if(self.use_random_target_for_latent_ver_double == True): #femur + knee
                    self.get_target_energy_height_pose()
                    self.target_energy = 0.18

        init_pos = None
        init_rot = None

        self.counter = frame
        if frame == -1:
             pass
             #self.step_counter(random=True)
        if set_state:
            self.scene.load_character_state(None, init_pos, init_rot) 

        self.state = None
        self.observation = None
        if(True): #use_mujoco True
            self.state = self.scene.character_state()
            #self.observation = self.scene.state2ob(self.state)
            self.observation = state2ob(torch.from_numpy(self.state), model_idx=self.model_idx).numpy()

            if(self.use_muscle_state_prediction == True):
                #assert version1 (len, vel) #force?
                #assert verions1 (dvel) #energy?
                self.muscle_state = self.scene.character_muscle_state(True, False)

            if(self.mujoco_sim_type == 'muscle'):
                self.energy_consumption_list = []

        #self.align_current()
        #self.align_target()


        future = None
        info = self.get_info()
        
        #self.step_counter(random = False)
        #self.step_counter(random = False, align = True)
        self.step_cnt = 0
        self.done_cnt = 0

        self.energy = 0
        if(self.use_muscle_state_prediction == True and self.use_muscle_state_energy_prediction == True):
            self.energy = np.zeros(120)

        if(self.use_muscle_state_prediction == True and self.use_muscle_state_energy_prediction_only == True):
            self.energy = np.zeros(120)

        if(self.use_muscle_state_prediction == True):
            return {
                'state': self.state,
                'observation': self.observation,
                'target': self.get_target(),
                'energy': self.energy, #0
                'muscle_state': self.muscle_state,
                'contact_state': self.contact_state
            }, info


        return {
            'state': self.state,
            'observation': self.observation,
            'target': self.get_target()#,
            #'future': future
        }, info

    def after_step(self, **kargs):
        """usually used to update target...
        """
        #self.step_counter(random = (self.step_cnt % self.random_count)==0 and self.step_cnt!=0)
        #self.step_counter(random = (self.step_cnt % self.random_count)==0 and self.step_cnt!=0, align = True)
        self.target = self.get_target()
        self.step_cnt += 1
        
    def step_core_muscle(self, action, using_yield = False, **kargs):
        pass

    def save_state(self):
        pass

    def step_core(self, action, using_yield = False, **kargs):

        if(self.use_goal_transition == True):
            random_dice = np.random.uniform(0, 1)
            #random_dice = 1
            #self.goal_transition_probability = 0.0
            self.goal_transition_probability = 0.03
            if(random_dice <= self.goal_transition_probability): #0.02 -> 0.03
                #self.target_velocity = np.random.uniform(0.5, 3.0) #vmin, vmax

                #goal_target_vel_rot
                self.target_velocity = np.random.uniform(-3.0, 3.0) 
                self.target_velocity_side = np.random.uniform(-3.0, 3.0)

                #goal_global_vel_rot
                if(self.without_vae_version_goal_global_vel_rot == True):
                    self.target_velocity = np.random.uniform(-1, 1)
                    self.target_velocity_side = np.random.uniform(-1, 1)
                    self.global_velocity = np.random.uniform(-3, 3)
                    self.global_velocity_side = np.random.uniform(-3, 3)

                #self.target_velocity = np.random.uniform(0, 2.0)
                #self.target_velocity_side = np.random.uniform(-0.1, 0.1)

                #print('goal change:', self.target_velocity)

                if(True): #assert without vae
                    if(self.use_random_target_for_latent_ver_energy == True):
                        self.target_energy = np.random.uniform(0.15, 0.21)
                        #self.target_energy = np.random.uniform(0.15, 0.25)
    
                    if(self.use_random_target_for_latent_ver_double == True): #femur + knee
                        self.get_target_energy_height_pose()
                        #self.target_energy = 0.18



        self.act_mean = np.mean(action)

        self.contact_state = np.zeros(2)

        if(self.use_mujoco == True):
            #mujoco muscle / torque / pd
            action_scale = 1
            
            if(True): #motor, muscle
                self.scene.mj_data.ctrl = action * action_scale

            if(self.mujoco_sim_type == 'muscle'):
                self.scene.mj_data.act = action[:self.num_muscles]

        #self.contact_force_prediction
        left_contact_force = 0
        right_contact_force = 0

        for i in range(self.substep):
            if(self.use_mujoco == True):

                mujoco.mj_step(self.scene.mj_model, self.scene.mj_data)

                if(self.mujoco_sim_type == 'muscle'):
                    mode_multi_energy = False
                    if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                        mode_multi_energy = True

                    self.energy_consumption_list.append(self.get_energy_consumption(mode_multi=mode_multi_energy))
                    #print(self.get_energy_consumption(), np.mean(action))

            if using_yield:
                yield self.sim_character.save()

        if(self.use_mujoco == True):
            self.state = self.scene.character_state()

            if(self.mujoco_sim_type == 'muscle'):
                mass_character = np.sum(self.scene.mj_model.body_mass) #74.9646
                #mass_character = 76
                energy_consumption = np.array(self.energy_consumption_list)

                #case1 15, 120 -> 120,
                #case2 15, -> scalar

                #energy_consumption_mean_for_time = np.mean(np.array(self.energy_consumption_list)) #deprecated
                energy_consumption_mean_for_time = np.mean(energy_consumption, axis=0)
                energy_normalized_by_mass =  (energy_consumption_mean_for_time + 1.51 * mass_character) / mass_character
                self.energy_consumption_list = []
                self.energy = energy_normalized_by_mass * 0.01 
                #45 -> 4.5 -> 0.45

                #print('energy:', np.sum(self.energy))

                if(self.use_muscle_state_energy_prediction == True or self.use_muscle_state_energy_prediction_only == True):
                    self.energy = self.energy * 10 #* 5

                self.act_mean = np.mean(self.energy)
                
                #print('current eng:', self.act_mean * 100 - 15, 'act:', np.mean(action))
                #print('')
                self.accumulated_energy += self.act_mean

        future = None
        
        self.observation = state2ob(torch.from_numpy(self.state), model_idx=self.model_idx).numpy()

        if(self.use_muscle_state_prediction == True):
            self.muscle_state = self.scene.character_muscle_state(True, False)


        reward = 0 
        done = self.cal_done(self.state, self.observation)
        info = self.get_info()
        self.after_step()


        self.accumulated_distance += (self.state[0, 0] - self.x_pos_before)
        self.x_pos_before = self.state[0, 0]


        observation = {
            'state': self.state,
            'target': self.target,
            #'target': self.get_target(),
            'observation': self.observation,
            #'contact_state': self.contact_state
        }
        #print(self.contact_state)

        if(self.use_muscle_state_prediction == True):
            observation={
            'state': self.state,
            'target': self.target,
            #'target': self.get_target(),
            'observation': self.observation,
            'energy': self.energy,
            'muscle_state': self.muscle_state,
            'contact_state': self.contact_state
        }

        #return observation, reward, done, info
        if not using_yield: # for convenient, so that we do not have to capture exception
            yield observation, reward, done, info
        else:
            return observation, reward, done, info  
    
    def step(self, action, **kargs):
        step_generator = self.step_core(action, **kargs)
        #print('current step')
        return next(step_generator)

    def step_render(self, action, **kargs): #set_action, check_done
        set_action = kargs['set_action']


        if(True):
            #self.target_velocity = 0
            #self.target_velocity_side = -1.2

            self.target = self.get_target()


        if(set_action == True):
            self.scene.mj_data.ctrl = action
            if(self.mujoco_sim_type == 'muscle'):
                self.scene.mj_data.act = action

            self.act_mean = np.mean(action)

        mujoco.mj_step(self.scene.mj_model, self.scene.mj_data)
        


        self.state = self.scene.character_state()
        self.energy = 0

        self.observation = state2ob(torch.from_numpy(self.state), model_idx=self.model_idx).numpy()
        if(self.use_muscle_state_prediction == True):
            self.muscle_state = self.scene.character_muscle_state(True, False)

        reward = 0 
        #done = self.cal_done(self.state, self.observation)
        info = self.get_info()
        #self.after_step()
        done = 0

        #to do
        #mode_done (reset)


        observation = {
            'state': self.state,
            'target': self.target,
            #'target': self.get_target(),
            'observation': self.observation,
            #'contact_state': self.contact_state
        }
        #print(self.contact_state)

        if(self.use_muscle_state_prediction == True):
            observation={
            'state': self.state,
            'target': self.target,
            #'target': self.get_target(),
            'observation': self.observation,
            'energy': self.energy,
            'muscle_state': self.muscle_state,
            'contact_state': self.contact_state
        }

        #yield or not
        #just sim
        return observation, reward, done, info

    
