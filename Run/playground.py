
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
from direct.task import Task
import time
import numpy as np
import torch

from FreeMuscoCore.Model.freemusco import decompose_obs
from FreeMuscoCore.Utils import diff_quat
from FreeMuscoCore.Utils import locomotion_utils

class Joystick:
    def __init__(self, joystick_idx=4, virtual_mode=False):
        self._buttons = None
        self._axes = None
        self._hat = None
        self._joystick = None
        self._joystick_idx = joystick_idx

        self._axes = list()
        self._buttons = list()
        self._hat = list()

        # if virtual mode, it can emulate joystick.
        self._virtual_mode = virtual_mode
        if self._virtual_mode:
            self._axes = (*(np.random.rand(2) - .5) * 2, 0, *(np.random.rand(2) - .5) * 2, 0)
            self._buttons = (0,)


        if pygame.joystick.get_count() < 1:
            print("please connect joystick")
            # sys.exit()
        else:
            try:
                self._joystick = pygame.joystick.Joystick(self._joystick_idx)
                self._joystick.init()


                print("Joystick Name:", self._joystick.get_name())
                print("Number of Buttons:", self._joystick.get_numbuttons())
                print("Number of Axes:", self._joystick.get_numaxes())
                print("Number of Hats:", self._joystick.get_numhats())

                self._buttons = list(range(self._joystick.get_numbuttons()))
                self._axes = list(range(self._joystick.get_numaxes()))
                self._hat = int()
            except:
                self._joystick = None
    
    def is_available(self):
        return self._joystick

    def is_virtual_mode(self):
        return self._virtual_mode
    
    def command(self):
        play = True
        if self.is_available():
            while play:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                for i in range(self._joystick.get_numbuttons()):
                    self._buttons[i] = self._joystick.get_button(i)

                for i in range(self._joystick.get_numaxes()):
                    self._axes[i] = self._joystick.get_axis(i)

                for i in range(self._joystick.get_numhats()):
                    self._hat = self._joystick.get_hat(i)

                yield self._buttons, self._axes, self._hat

        elif self._virtual_mode:
            while play:
                yield self._buttons, self._axes, self._hat
        else:
            return None, None, None
    
    def update_virtual_command(self):
        self._axes = (*(np.random.rand(2) - .5) * 2, 0, *(np.random.rand(2) - .5) * 2, 0)
        self._buttons = (0,)

class Playground(FreeMusco):
    def __init__(self, observation_size, action_size, delta_size, env, **kargs):
        super(Playground, self).__init__(observation_size, action_size, delta_size, env, **kargs)
        self.mode = kargs['mode']
        self.show = kargs['show']
        
        self.observation = self.env.reset()[0]
        self.step_generator = None
        self.other_objects = {}

        self.cnt = 0
        self.env.reset()
        
    def get_action(self, **obs_info): 
        #print(obs_info)
        #print(xxx)
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, logvar  = self.encoder.encode_prior(n_observation)
        action = self.decode(n_observation, latent)
        info = {'mu': mu, 'logvar': logvar}
        return action, info
    
    def get_generator(self):
        action, info = self.get_action(**self.observation)
        action = ptu.to_numpy(action)
        return self.env.step_core(action, using_yield = True) #original
        #return self.env.step_core(action, using_yield = False) #for mujoco
    
    def yield_step(self):
        if self.step_generator is None:
            self.step_generator = self.get_generator()        
        try:
            self.step_generator.__next__()
        except StopIteration as e:
            self.observation = e.value[0]
            self.step_generator = self.get_generator() 
            #self.env.load_character_state(self.env.ref_character, self.env.motion_data.state[self.env.counter]) #original
            #self.env.load_character_state(self.env.motion_data.state[self.env.counter]) #mujoco
            self.step_generator.__next__()

    def post_step(self, observation):
        pass

    def run(self):
        self.env.reset()
        if self.mode == 'mujoco':
            pass
            #init viewer
            #update viewer
    
    def after_step(self, server_scene):
        pass

    @staticmethod
    def build_arg(parser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default = 'panda', type = str)
        parser.add_argument('--show', default = False, action='store_true')
        parser = MujocoMuscleEnv.add_specific_args(parser)
        parser = FreeMusco.add_specific_args(parser)
        args = vars(parser.parse_args())
        ptu.init_gpu(True)
        # con
        #config = load_yaml(initialdir ='Data/NNModel/Pretrained')
        config = load_yaml(initialdir ='Data/Pretrained/') #ask file box (default)
        #config = load_yaml(path ='Data/Pretrained/config.yml') #just open file (my)
        args.update(config)

        return args


class MotionFreePlayground(Playground):
    def get_action(self, **obs_info):
        info = {}

        n_observation = self.obsinfo2n_obs(obs_info)
        target = obs_info['target']
        n_target = self.normalize_obs(target)


        latent_code, mu_post, mu_prior = None, None, None        

        if(True):
            #default
            # latent, mu, logvar = self.encoder.encode_post(n_observation, n_target)
            latent_code, mu_post, mu_prior = self.encode(n_observation, n_target) #here

        #random_sample
        if(False): #250217 #inference prior
            latent_code, mu_prior, logvar = self.encoder.encode_prior(n_observation)
            mu_post = mu_prior 
            mu_post = latent_code

        #mu_post = latent_code
        action = self.decode(n_observation, mu_post)
        #action = self.decode(n_observation, latent_code)

        # info = {'mu': mu, 'logvar': logvar}
        return action, info
    
    def yield_step(self):
        super().yield_step()
        self.env.load_character_state(self.env.ref_character, self.env.motion_data.state[self.env.counter])
    

paused = False #SPACE
need_reset = False #R

need_transitionQ = False
need_transitionW = False
need_transitionE = False

need_energyUp = False
need_energyDown = False
need_poseOrigin = False #B
need_poseRandom = False #N
need_poseThird = False #V

import numpy as np

if __name__ == '__main__':

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use', type = str, default = 'run')
    #parser.add_argument('--use', type = str, default = 'visualize')
    args = Playground.build_arg(parser)
    args['show'] = True


    task_ver = 0

    if(True): 
        args['env_max_length'] = 2048

    env = MujocoMuscleEnv(**args)

    playground = None
    if(args['use_mujoco'] == True):
        if(env.mujoco_sim_type == 'muscle'):
            playground = MotionFreePlayground(323, 120, 120, env, ** args)

    import tkinter.filedialog as fd
    
    if(True):
        #data_file = fd.askopenfilename(filetypes=[('DATA','*.data')], initialdir="./Data/Pretrained")
        data_file = args['checkpoint_path']
        print('Load Checkpoint From Path:', data_file)
        playground.try_load(data_file)

    render_command = True
    render_root = False
    render_text = False

    render_pose = True
    render_energy = True


    import mujoco
    from mujoco import viewer
    import time

    use_joystick = False

    import pygame
    pygame.init()
    joystick = None
    joystick_coroutine = None

    if(use_joystick == True):
        joystick = Joystick(joystick_idx=0)
        joystick_coroutine = joystick.command() if joystick.is_available() else None

    if(args['use_mujoco'] == True):
        playground.mode = 'mujoco'
        obs = playground.env.reset()

        action = playground.get_action(**obs[0])

        num_step = 0

        action = action[0].detach().cpu().numpy()
        action = action[0]

        state1, state2, state3, state4 = playground.env.step(action)
        obs = state1

        model = playground.env.scene.mj_model
        data = playground.env.scene.mj_data

        num_step += 1
        print(num_step, data.time)

        def add_visual_capsule(scene, point1, point2, radius, rgba):

            if scene.ngeom >= scene.maxgeom:
                return
            scene.ngeom += 1

            #rot = np.zeros(9)
            mesh_type = mujoco.mjtGeom.mjGEOM_CAPSULE


            #rot = np.ones(9)
            rot = np.array([0.7544065, -0.1330222,  0.6427876, 0.4668948,  0.7970591, -0.3830222, -0.4613892,  0.5890687,  0.6634139 ])
            #mesh_type = mujoco.mjtGeom.mjGEOM_BOX
            mesh_type = mujoco.mjtGeom.mjGEOM_CYLINDER

            mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1], 
                mesh_type, np.zeros(3),
                np.zeros(3), rot, rgba.astype(np.float32))
            mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1], mesh_type, radius,
                point1[0], point1[1], point1[2], point2[0], point2[1], point2[2])
            #+ 0.1

        def add_visual_arrow(scene, point1, point2, radius, rgba):
            if scene.ngeom >= scene.maxgeom:
                return
            scene.ngeom += 1

            mesh_type = mujoco.mjtGeom.mjGEOM_CAPSULE

            rot = np.array([0.7544065, -0.1330222,  0.6427876, 0.4668948,  0.7970591, -0.3830222, -0.4613892,  0.5890687,  0.6634139 ])
            #mesh_type = mujoco.mjtGeom.mjGEOM_BOX
            #mesh_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            
            mesh_type = mujoco.mjtGeom.mjGEOM_ARROW
            #mesh_type = mujoco.mjtGeom.mjGEOM_MESH

            mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1], 
                mesh_type, np.zeros(3),
                np.zeros(3), rot, rgba.astype(np.float32))
            mujoco.mjv_makeConnector(scene.geoms[scene.ngeom - 1], mesh_type, radius,
                point1[0], point1[1], point1[2], point2[0], point2[1], point2[2])


        def add_visual_label(viewer, text_input_list, text_size, text_pos, text_color=np.array([0, 0, 0, 0])):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_LABEL,
                size=text_size, #size=np.array([0.3, 0.3, 0.3]),  # label_size
                pos=text_pos, #pos=obs['state'][0][0:3] +np.array([-1.0, 0.0, 0.0]),  # label position, here is 1 meter above the root joint
                mat=np.eye(3).flatten(),  # label orientation, here is no rotation
                rgba=text_color #rgba is currently not supported
                #rgba=np.array([0, 0, 0, 0])  # invisible
            )
            #geom.label = 'root vel:' + str(obs['state'][0][7])  # receive string input only

            text_input = ''
            for text in text_input_list:
                if (isinstance(text, str) == True):
                    text_input += text
                else: #float, int 
                    text_input += str(round(text, 3))

            geom.label = text_input
            viewer.user_scn.ngeom += 1

        def key_callback(keycode):
            #print(keycode)
            if chr(keycode) == ' ':
                global paused
                paused = not paused
            elif int(keycode) == 82:
            #elif chr(keycode) == 'r':
                global need_reset
                need_reset = True
            #reset
            #q 71 w 77 e 65
            elif chr(keycode) == 'q' or chr(keycode) == 'Q':
                global need_transitionQ
                need_transitionQ = True
                #print(xxx)

            #elif chr(keycode) == 'w' or chr(keycode) == 'W':
            elif chr(keycode) == 'z' or chr(keycode) == 'Z':
                global need_transitionW
                need_transitionW = True
                #print(xxx)

            elif chr(keycode) == 'e' or chr(keycode) == 'E':
                global need_transitionE
                need_transitionE = True
                #print(xxx)

            elif chr(keycode) == 'o' or chr(keycode) == 'O':
                global need_energyDown
                need_energyDown = True
                #print(xxx)

            elif chr(keycode) == 'l' or chr(keycode) == 'L': #for screenshot ctrl p
            #elif chr(keycode) == 'p' or chr(keycode) == 'P':
                global need_energyUp
                need_energyUp = True
                #print(xxx)

            elif chr(keycode) == 'b' or chr(keycode) == 'B':
                global need_poseOrigin
                need_poseOrigin = True
                #print(xxx)

            elif chr(keycode) == 'n' or chr(keycode) == 'N':
                global need_poseRandom
                need_poseRandom = True
                #print(xxx)

            elif chr(keycode) == 'v' or chr(keycode) == 'V':
                global need_poseThird
                need_poseThird = True
                #print(xxx

        start_root = np.zeros(3)
        target_root = np.zeros(3)
        is_first = True

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            time.sleep(1)
            start = time.time()

            if(playground.env.use_model_change == False): #fullbody
                viewer.cam.azimuth = -90.25
                viewer.cam.elevation = 89.0
                viewer.cam.distance = 5.3
                viewer.cam.lookat = np.array([3.04611614,  0.03706812, -0.31322622])

                if(True): #for render video (fullbody, chimanoid, ostrich)
                    viewer.cam.azimuth = -89.83202427780579
                    viewer.cam.elevation = 69.84692003180919
                    viewer.cam.distance = 6.93750325477825
                    viewer.cam.lookat = np.array([-0.05703088,  0.03705797, -0.04690183])

                    # for revision (2-splits, no sky))
                    viewer.cam.azimuth = -90
                    viewer.cam.elevation = 64.97192003180919

            if(playground.env.use_model_change == True): #ostrich
                if(True):
                    viewer.cam.azimuth = -90.47320988734893
                    viewer.cam.elevation = -18.786946097905982
                    viewer.cam.distance = 6.72962654570831
                    viewer.cam.lookat = np.array([ 0.03530752, -0.01134967,  0.18327653])

            #global paused, need_reset
            while viewer.is_running() and time.time() - start < 100000:
                if(paused == False):
                    step_start = time.time()
                    time_for_render_step = 0.002 * 15
                    #time_for_render_step = 0.003 * 15

                    #time_for_render_step = 0.0025 * 15
                    #ostrich
                    
                    if(use_joystick == True):
                        #for buttons, axes, hat in joystick.command():
                        #    print(buttons, axes, hat)
                        cmd_gen = joystick.command()  
                        buttons, axes, hat = next(cmd_gen)

                        playground.env.target_velocity = -1 * axes[0] * 2
                        playground.env.target_velocity_side = -1 * axes[1] * 2

                        if(buttons[0] == True):
                            need_reset = True

                        playground.env.target_energy = 0.17 + axes[4] * -1 * 0.04

                        if(playground.env.model_idx == 1):
                            playground.env.target_velocity_side = 1 * axes[1] * 2
                            playground.env.target_energy = 0.2 + axes[4] * -1 * 0.04

                        print('energy:', playground.env.target_energy)
                        print('velocity:', playground.env.target_velocity, playground.env.target_velocity_side)

                    before = time.time()

                    action = playground.get_action(**obs)

                    action = action[0].detach().cpu().numpy()
                    action = action[0]


                    obs, state2, state3, state4 = playground.env.step(action + np.random.randn(120) * 0.) #0.12


                    #if(True):
                    if(need_transitionQ == True or need_transitionW == True or need_transitionE == True): #without_vae_version_goal_target_vel_rot

                        target_velocity = np.random.uniform(-1.5, 1.5) 
                        target_velocity_side = np.random.uniform(-1.5, 1.5)

                        magnitude = np.sqrt(target_velocity * target_velocity + target_velocity_side * target_velocity_side)
                        if(magnitude > 0.01):
                            target_velocity = target_velocity / magnitude
                            target_velocity_side = target_velocity_side / magnitude


                        if(need_transitionQ == True):
                            magnitude = np.random.uniform(0.5, 1.2) 

                            need_transitionQ = False

                            if(playground.env.without_vae_version_goal_target_vel_rot == True):
                                playground.env.target_velocity = target_velocity * magnitude
                                playground.env.target_velocity_side = target_velocity_side * magnitude

                            if(playground.env.without_vae_version_goal_global_vel_rot == True):
                                playground.env.global_velocity_side = 0
                                playground.env.global_velocity = 0 


                        if(need_transitionW == True):
                            magnitude = np.random.uniform(1.2, 2.5)

                            need_transitionW = False

                            if(playground.env.without_vae_version_goal_target_vel_rot == True):
                                playground.env.target_velocity = target_velocity * magnitude
                                playground.env.target_velocity_side = target_velocity_side * magnitude

                            if(playground.env.without_vae_version_goal_global_vel_rot == True):
                                
                                playground.env.target_velocity = np.random.uniform(-1, 1)
                                playground.env.target_velocity_side = np.random.uniform(-1 ,1 )

                                magnitude = np.sqrt(playground.env.target_velocity * playground.env.target_velocity + playground.env.target_velocity_side * playground.env.target_velocity_side)

                                playground.env.target_velocity = playground.env.target_velocity / magnitude

                                playground.env.target_velocity_side = playground.env.target_velocity_side / magnitude
                            print('change')

                            
                        if(need_transitionE == True):

                            magnitude = np.random.uniform(2.8, 4.0)

                            if(playground.env.model_idx == 1):
                                magnitude = np.random.uniform(4.5, 6.0)


                            need_transitionE = False
                            if(playground.env.without_vae_version_goal_target_vel_rot == True):
                                playground.env.target_velocity = target_velocity * magnitude
                                playground.env.target_velocity_side = target_velocity_side * magnitude

                            if(playground.env.without_vae_version_goal_global_vel_rot == True):
                                playground.env.global_velocity = np.random.uniform(-2.5, 2.5)
                                playground.env.global_velocity_side = np.random.uniform(-2.5, 2.5)
                            print('change')

                    if(need_energyUp == True or need_energyDown == True or need_poseOrigin == True or need_poseRandom == True or need_poseThird == True):
                        if(need_energyUp == True):
                            playground.env.target_energy = playground.env.target_energy + 0.005 #0.01
                            #playground.env.target_height = playground.env.target_height + 0.05
                            #print(playground.env.target_height)
                            need_energyUp = False

                        if(need_energyDown == True):
                            playground.env.target_energy = playground.env.target_energy - 0.005 #0.01
                            #playground.env.target_height = playground.env.target_height - 0.05
                            #print(playground.env.target_height)
                            need_energyDown = False

                        if(playground.env.model_idx == 0):
                            if(playground.env.target_energy < 0.15):
                                playground.env.target_energy = 0.15
                            if(playground.env.target_energy > 0.21): # and playground.env.use_fullbody_change == False):
                                playground.env.target_energy = 0.21

                        if(playground.env.model_idx == 1):
                            if(playground.env.target_energy < 0.16):
                                playground.env.target_energy = 0.16
                            if(playground.env.target_energy > 0.28):
                                playground.env.target_energy = 0.28

                        if(playground.env.use_random_target_for_latent_ver_double == True):
                            
                            if(need_poseOrigin == True):
                                playground.env.get_target_energy_height_pose(100)
                                #playground.env.get_target_energy_height_pose(1) 
                                #playground.env.get_target_energy_height_pose(100) #rest pose (revision)
                                need_poseOrigin = False

                            if(need_poseRandom == True):
                                playground.env.get_target_energy_height_pose(20)
                                #playground.env.get_target_energy_height_pose(-1)
                                need_poseRandom = False

                            if(need_poseThird == True):
                                playground.env.get_target_energy_height_pose(10)
                                #playground.env.get_target_energy_height_pose(-1)
                                need_poseThird = False
                    
                    #print(obs['state'][1][1], obs['state'][10][1], obs['state'][11][1]) #root, thorax1, thorax2

                    before_look = np.array([viewer.cam.lookat[2]])

                    if(True): #track
                        #render_camera
                        if(playground.env.model_idx == 0):
                            viewer.cam.lookat[0] = obs['state'][0][0]
                            #viewer.cam.lookat[1] = obs['state'][0][1]
                            viewer.cam.lookat[2] = obs['state'][0][2]

                            #revision
                            #viewer.cam.distance = 5.0
                            #viewer.cam.distance = viewer.cam.distance - (viewer.cam.lookat[2] - before_look)

                        if(playground.env.model_idx == 1): #ostrich
                            viewer.cam.lookat[0] = obs['state'][0][0]
                            viewer.cam.lookat[1] = obs['state'][0][1]
                            #viewer.cam.lookat[2] = obs['state'][0][2]      


                        print('viewer info:', viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance, viewer.cam.lookat)
                        viewer.cam.azimuth = -90 #render_camera #for revision

                    if(True):
                        start_root_pos = obs['state'][0][0:3]

                    if(render_text == True): #add text on viewer
                        #viewer, text, pos, size
                        text_size = np.ones(3) * 0.3
                        text_pos = obs['state'][0][0:3]
                        if(playground.env.model_idx == 0): #fullbody
                            if(playground.env.without_vae_version_goal_target_vel == True):
                                add_visual_label(viewer, ['target vel: ', playground.env.target_velocity, '  policy step: ', num_step], text_size, text_pos + np.array([-1.0, 0.0, 0.0]))
                                add_visual_label(viewer, ['root vel (global): ', obs['state'][0][7], ' (local): ', obs['observation'][180]], text_size, text_pos + np.array([-1.0, -0.1, 0.0]))
                                add_visual_label(viewer, ['act mean: ', action.mean()], text_size, text_pos + np.array([-1.0, -0.2, 0.0]))
                                add_visual_label(viewer, ['energy mean(simul): ', np.mean(playground.env.energy)], text_size, text_pos + np.array([-1.0, -0.3, 0.0]))
                                
                                #add_visual_label(viewer, ['energy mean(world): ', np.mean(playground.env.energy_pred)], text_size, text_pos + np.array([-1.0, -0.4, 0.0]))
        
                                if(playground.env.contact_state[0] > 0):
                                    add_visual_label(viewer, ['contact L: ', playground.env.contact_state[0]], text_size, text_pos + np.array([-1.0, -0.6, 0.0]))
                                if(playground.env.contact_state[1] > 0):
                                    add_visual_label(viewer, ['contact R: ', playground.env.contact_state[1]], text_size, text_pos + np.array([-1.0, -0.7, 0.0]))
                            else: #goal_target_vel_rot
                                target_vel = np.sqrt(playground.env.target_velocity * playground.env.target_velocity + playground.env.target_velocity_side * playground.env.target_velocity_side)
                                add_visual_label(viewer, ['target vel: ', target_vel, '  policy step: ', num_step], text_size, text_pos + np.array([-1.0, 0.0, 0.0]))
                                add_visual_label(viewer, ['act mean: ', action.mean()], text_size, text_pos + np.array([-1.0, -0.2, 0.0]))
                                add_visual_label(viewer, ['energy mean(simul): ', np.mean(playground.env.energy)], text_size, text_pos + np.array([-1.0, -0.3, 0.0]))

                                #latent_ver_height
                                add_visual_label(viewer, ['root height: ', np.mean(obs['observation'][300])], text_size, text_pos + np.array([-1.0, -0.4, 0.0])) 
                                #add_visual_label(viewer, ['root height: ', np.mean(obs['observation'][304])], text_size, text_pos + np.array([-1.0, -0.4, 0.0])) #foot #thorax 311
                                #add_visual_label(viewer, ['foot height R: ', np.mean(obs['observation'][304]), '  foot height L: ', np.mean(obs['observation'][305])], text_size, text_pos + np.array([-1.0, -0.4, 0.0])) 
                                #add_visual_label(viewer, ['knee angle: ', playground.env.temp_angle], text_size, text_pos + np.array([-1.0, -0.4, 0.0])) 
                                #playground.env.target_height


                        if(playground.env.model_idx == 1): #ostrich
                            if(playground.env.without_vae_version_goal_target_vel == True):
                                add_visual_label(viewer, ['target vel: ', playground.env.target_velocity, '  policy step: ', num_step], text_size, text_pos + np.array([-2.0, 0.0, 0.0]))
                                add_visual_label(viewer, ['root vel (global): ', obs['state'][0][7], ' (local): ', obs['observation'][288]], text_size, text_pos + np.array([-2.0, 0.0, -0.1])) #ostrich
                                add_visual_label(viewer, ['act mean: ', action.mean()], text_size, text_pos + np.array([-2.0, 0.0, -0.2]))
                                add_visual_label(viewer, ['energy mean(simul): ', np.mean(playground.env.energy)], text_size, text_pos + np.array([-2.0, 0.0, -0.3]))
                                
                                add_visual_label(viewer, ['energy mean(world): ', np.mean(playground.env.energy_pred)], text_size, text_pos + np.array([-2.0, 0.0, -0.4]))

                            else: #goal_target_vel_rot
                                target_vel = np.sqrt(playground.env.target_velocity * playground.env.target_velocity + playground.env.target_velocity_side * playground.env.target_velocity_side)
                                add_visual_label(viewer, ['target vel: ', target_vel, '  policy step: ', num_step], text_size, text_pos + np.array([-2.0, 0.0, 0.0]))
                                add_visual_label(viewer, ['act mean: ', action.mean()], text_size, text_pos + np.array([-2.0, 0.0, -0.2]))
                                add_visual_label(viewer, ['energy mean(simul): ', np.mean(playground.env.energy)], text_size, text_pos + np.array([-2.0, 0.0, -0.3]))



                    num_step += 1
                    if(True):
                        if(playground.env.model_idx == 1): #ostrich
                            print('vel:', obs['state'][0][7], 'height:', obs['state'][0][2], 'act:', action.mean()) #energy
                        else: #humanoid # dog
                            print('global vel:', obs['state'][0][7], 'height:', obs['state'][0][1], 'local vel:', obs['observation'][180], 'act:', action.mean()) #energy
                            #print('energy:', np.mean(playground.env.energy), 'act:', action.mean())
                        #ostrich
    
                        print(num_step)
                    

                    if(render_command == True): #command
                        #draw target vel only

                        current_root_pos = np.array(obs['state'][0][0:3])
                        current_root_quat = np.array(obs['state'][0][3:7])

                        if(playground.env.model_idx == 0):
                            
                            current_root_pos[1] = 0. #projected to ground

                            goal_vel = 1 * np.array([playground.env.target_velocity, 0, playground.env.target_velocity_side])  #/ np.sqrt(np.sum((playground.env.target_velocity* playground.env.target_velocity) * (playground.env.target_velocity* playground.env.target_velocity))) + 0.02
                            add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 0, 1]))


                        #target pose, target energy
                        elif(playground.env.model_idx == 1):
                            current_root_pos[2] = 0

                            goal_vel = np.array([playground.env.target_velocity, playground.env.target_velocity_side, 0])
                            add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 0, 1]))

                        if(playground.env.without_vae_version_goal_global_vel_rot == True):
                            goal_vel = np.array([playground.env.global_velocity, 0, playground.env.global_velocity_side])
                            add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 1, 1]))


                    if(render_energy == True and (playground.env.use_random_target_for_latent_ver_double == True or playground.env.use_random_target_for_latent_ver_energy == True) ):
                        energy_count = 0
                        energy_range = 0.145 #model_idx == 1
                        idx_y = 1
                        energy_offset = 1.6
                        goal_vel = np.array([0, 0.04, 0.0])
                        human_offset = 0.25

                        if(playground.env.model_idx == 1):  #model_idx == 1
                            energy_range = 0.165
                            idx_y = 2
                            energy_offset = 1.1
                            goal_vel = np.array([0, 0., 0.04])
                            human_offset = 0

                        energy_pos = np.zeros(3)
                        energy_count = int (((playground.env.target_energy - energy_range) * 100) / 0.5)

                        if(True):
                            current_root_pos = playground.env.state[0, 0:3]
                            energy_pos[0] = current_root_pos[0] - human_offset # -0.3
                            energy_pos[1] = current_root_pos[1]
                            energy_pos[2] = current_root_pos[2] - human_offset #-1.0
                            #ostrich goal_vel = np.array([0, 0.01, 0.0]) #0.04-0.05-0.04
                            #human
                            goal_vel = np.array([0, 0.0, 0.01])
                            for e in range(energy_count):
                                energy_pos[idx_y] = e * 0.012 + energy_offset
                                add_visual_capsule(viewer.user_scn, energy_pos, energy_pos + goal_vel , 0.03, np.array([0, 1, 1, 1]))


                    #if(False):
                    if(render_pose == True and (playground.env.use_random_target_for_latent_ver_double == True and playground.env.dummy_data is not None)): #mk1 mk2

                        if(playground.env.model_idx == 0):
                            mujoco.mj_resetData(playground.env.scene.mj_model, playground.env.dummy_data)
                            playground.env.dummy_data.joint('pelvis_tx').qpos = playground.env.scene.mj_data.joint('pelvis_tx').qpos + 0.6 #for revision: 1 -> 0.4
                            playground.env.dummy_data.joint('pelvis_ty').qpos = playground.env.scene.mj_data.joint('pelvis_ty').qpos + 0.1
                            playground.env.dummy_data.joint('pelvis_tz').qpos = playground.env.scene.mj_data.joint('pelvis_tz').qpos + 1
                            playground.env.dummy_data.joint('pelvis_rotation').qpos = playground.env.scene.mj_data.joint('pelvis_rotation').qpos
                            playground.env.dummy_data.joint('pelvis_tilt').qpos = playground.env.scene.mj_data.joint('pelvis_tilt').qpos
                            playground.env.dummy_data.joint('pelvis_list').qpos = playground.env.scene.mj_data.joint('pelvis_list').qpos

                            if(playground.env.ver_double_mk1 == False and playground.env.ver_double_mk2 == False and playground.env.ver_double_mk3 == False and playground.env.ver_double_mk4 == False):
                                playground.env.dummy_data.joint('hip_flexion_r').qpos = playground.env.temp_angle[0]
                                playground.env.dummy_data.joint('hip_flexion_l').qpos = playground.env.temp_angle[1]

                                playground.env.dummy_data.joint('knee_angle_r').qpos = playground.env.temp_angle[2]
                                playground.env.dummy_data.joint('knee_angle_l').qpos = playground.env.temp_angle[3]

                            if(playground.env.ver_double_mk1 == True or playground.env.ver_double_mk2 == True):
                                playground.env.dummy_data.joint('hip_flexion_r').qpos = playground.env.temp_angle[0]
                                playground.env.dummy_data.joint('hip_flexion_l').qpos = playground.env.temp_angle[0]

                                playground.env.dummy_data.joint('knee_angle_r').qpos = playground.env.temp_angle[1]
                                playground.env.dummy_data.joint('knee_angle_l').qpos = playground.env.temp_angle[1]

                                if(playground.env.ver_double_mk1 == True): #mk1
                                    playground.env.dummy_data.joint('ankle_angle_r').qpos = playground.env.temp_angle[2]
                                    playground.env.dummy_data.joint('ankle_angle_l').qpos = playground.env.temp_angle[2]

                                if(playground.env.ver_double_mk2 == True): #mk2
                                    playground.env.dummy_data.joint('elv_angle').qpos = playground.env.temp_angle[2]
                                    playground.env.dummy_data.joint('elv_angle_l').qpos = playground.env.temp_angle[2]

                                    #playground.env.dummy_data.joint('shoulder_rot').qpos = playground.env.temp_angle[2]
                                    #playground.env.dummy_data.joint('shoulder_rot_l').qpos = playground.env.temp_angle[2]
    
                                    playground.env.dummy_data.joint('elbow_flexion').qpos = playground.env.temp_angle[3]
                                    playground.env.dummy_data.joint('elbow_flexion_l').qpos = playground.env.temp_angle[3]

                            if(playground.env.ver_double_mk4 == True): #for chimanoid
                                playground.env.dummy_data.joint('elv_angle').qpos = playground.env.temp_angle[0]
                                playground.env.dummy_data.joint('elv_angle_l').qpos = playground.env.temp_angle[0]

                            mujoco.mj_step(playground.env.scene.mj_model, playground.env.dummy_data)

                            #'hip_adduction_l', 'hip_adduction_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_rotation_l', 'hip_rotation_r', 
                            #'knee_angle_l', 'knee_angle_r'
                            #'ankle_angle_l', 'ankle_angle_l2', 'ankle_angle_l3', 'ankle_angle_r', 'ankle_angle_r2', 'ankle_angle_r3', 
                            # 'mtp_angle_l', 'mtp_angle_r'
                
                            #'shoulder_elv', 'shoulder_elv_l', 'shoulder_rot', 'shoulder_rot_l', 'elv_angle', 'elv_angle_l', 
                            #'elbow_flexion', 'elbow_flexion_l', 
                            #'wrist_3_l', 'wrist_3_r', 'wrist_dev_l', 'wrist_dev_r', 'wrist_flex_l', 'wrist_flex_r'
                            
                            # 'lumbar_bending', 'lumbar_extension', 'lumbar_rotation', 
                            # 'pelvis_list', 'pelvis_rotation', 'pelvis_tilt', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 

                        if(playground.env.model_idx == 1):
                            mujoco.mj_resetData(playground.env.scene.mj_model, playground.env.dummy_data)
                            playground.env.dummy_data.joint('root_x').qpos = playground.env.scene.mj_data.joint('root_x').qpos + 0.8 #1.5 #for revision 1 -> 0.5 (video)
                            playground.env.dummy_data.joint('root_y').qpos = playground.env.scene.mj_data.joint('root_y').qpos
                            playground.env.dummy_data.joint('root_z').qpos = playground.env.scene.mj_data.joint('root_z').qpos + 0.1 #for revision 1 -> 0.1 (image, video)
                            playground.env.dummy_data.joint('root_rot_x').qpos = playground.env.scene.mj_data.joint('root_rot_x').qpos
                            playground.env.dummy_data.joint('root_rot_y').qpos = playground.env.scene.mj_data.joint('root_rot_y').qpos
                            playground.env.dummy_data.joint('root_rot_z').qpos = playground.env.scene.mj_data.joint('root_rot_z').qpos

                            if(playground.env.ver_double_mk1 == True): #double + mk1
                                playground.env.dummy_data.joint('r_hip_y').qpos = playground.env.temp_angle[0]
                                playground.env.dummy_data.joint('l_hip_y').qpos = playground.env.temp_angle[0]

                                playground.env.dummy_data.joint('r_knee_y').qpos = playground.env.temp_angle[1]
                                playground.env.dummy_data.joint('l_knee_y').qpos = playground.env.temp_angle[1]

                                if(True): #mk1
                                    playground.env.dummy_data.joint('r_ankle_y').qpos = playground.env.temp_angle[2]
                                    playground.env.dummy_data.joint('l_ankle_y').qpos = playground.env.temp_angle[2]

                            if(playground.env.ver_double_mk4 == True):
                                if(playground.env.mk4_mode == 0):
                                    playground.env.dummy_data.joint('r_hip_y').qpos = playground.env.temp_angle[0]
                                    playground.env.dummy_data.joint('l_hip_y').qpos = playground.env.temp_angle[0]

                                elif(playground.env.mk4_mode == 1):
                                    playground.env.dummy_data.joint('r_knee_y').qpos = playground.env.temp_angle[0]
                                    playground.env.dummy_data.joint('l_knee_y').qpos = playground.env.temp_angle[0]

                                elif(playground.env.mk4_mode == 2):
                                    playground.env.dummy_data.joint('r_ankle_y').qpos = playground.env.temp_angle[0]
                                    playground.env.dummy_data.joint('l_ankle_y').qpos = playground.env.temp_angle[0]


                            mujoco.mj_step(playground.env.scene.mj_model, playground.env.dummy_data)

                        for g in range(playground.env.scene.mj_model.nbody):
                            if(g == 0 or g == 1):
                                continue
                            current_body = playground.env.dummy_data.body(g)
                            parent_idx = playground.env.scene.mj_model.body(g).parentid[-1] #0

                            #print(g, parent_idx)
                            current_pos = current_body.xpos
                            parent_pos = playground.env.dummy_data.body(parent_idx).xpos

                            #print(current_pos)

                            body_color = np.array([1, 0, 0, 1])
                            if(current_body.name == 'toes_r' or current_body.name == 'toes_l'):
                                body_color = np.array([1, 1, 0, 1])


                            add_visual_capsule(viewer.user_scn, current_pos, parent_pos , 0.01, body_color)


                    if(render_root == True and (playground.env.without_vae_version_goal_target_vel_rot == True or playground.env.without_vae_version_goal_global_vel_rot == True)): #forward vector from quaternion / goal_target_vel_rot #2dim
                        current_root_pos = obs['state'][0][0:3]
                        current_root_quat = obs['state'][0][3:7]
                        qx = current_root_quat[0]
                        qy = current_root_quat[1]
                        qz = current_root_quat[2]
                        qw = current_root_quat[3]

                        vx = 1 - 2 * (qy*qy + qz*qz)
                        vy = 2 * (qx*qy + qw*qz)
                        vz = 2 * (qx*qz - qw*qy)


                        if(playground.env.model_idx == 0):
                            if(True): #current vel, current forward vel (normalized), target vel, 
                                goal_vel = [vx, 0, vz] #vy
                                add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([1, 0, 0, 1]))
        
                                goal_vel = [obs['state'][0][7], 0, obs['state'][0][9]]
                                add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 0, 1]))
        
                                if(True):
                                    goal_vel = np.array([playground.env.target_velocity, 0, playground.env.target_velocity_side])
                                    add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 0, 1, 1]))
                            
                            #root traj
                            #add_visual_capsule(viewer.user_scn, current_root_pos, current_root_pos + np.array([0.001, 0, 0.001]) , 0.02, np.array([0, 0, 1, 1]))

                            #global vel
                            if(True):
                                if(playground.env.without_vae_version_goal_global_vel_rot == True):
                                    goal_vel = np.array([playground.env.global_velocity, 0, playground.env.global_velocity_side])
                                    add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 1, 1]))
                        

                            if(True): #up dir
                                goal_vel = np.array([obs['observation'][320], obs['observation'][321], obs['observation'][322]])
                                #root_pos = np.array([obs['state'][0][0], obs['state'][0][1], obs['state'][0][2]])
                                add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 1, 1]))

                        if(playground.env.model_idx == 1): #ostrich
                            goal_vel = [vx, vy, 0] #vy
                            add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([1, 0, 0, 1]))
    
                            goal_vel = [obs['state'][0][7], obs['state'][0][8], 0]
                            add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 0, 1]))
    
    
                            goal_vel = np.array([playground.env.target_velocity, playground.env.target_velocity_side, 0])
                            add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 0, 1, 1]))
                            

                            if(playground.env.without_vae_version_goal_global_vel_rot == True):
                                goal_vel = np.array([playground.env.global_velocity, playground.env.global_velocity_side, 0])
                                add_visual_arrow(viewer.user_scn, current_root_pos, current_root_pos + goal_vel , 0.02, np.array([0, 1, 1, 1]))


                    #if(need_reset == True): #reset button
                    if(state3 > 0 or need_reset == True):
                    #if(num_step >= 150):
                    #if(state3 > 0 or num_step >= 100):
                        need_reset = False
                        print('env reset')
                        obs = playground.env.reset()[0]
                        num_step = 0

                        is_first = True

                        #viewer.user_scn.ngeom = 0 #mujoco version > 3.0

                        start_root_pos = obs['state'][0][0:3]

                    #self.step_cnt % self.random_count)==0 and self.step_cnt!=0
                    elif((playground.env.step_cnt % playground.env.random_count) == 0 and playground.env.step_cnt != 0 ):
                        
                        print('target reset')
                        is_first = True

                        #viewer.user_scn.ngeom = 0 #mujoco version > 3.0

                        start_root_pos = obs['state'][0][0:3]

                    else:
                        pass

                    after = time.time()

                    with viewer.lock():
                        pass
                        #viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                    # Pick up changes to the physics state, apply perturbations, update options from GUI.
                    #time.sleep(0.001)
                    viewer.sync()

                    # Rudimentary time keeping, will drift relative to wall clock.
                    #time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    time_until_next_step = time_for_render_step - (time.time() - step_start)

                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                    #mujoco version > 3.0
                    viewer.user_scn.ngeom = 0

    #FINISH
    #if args['use'] == 'run':
    #    playground.run()

