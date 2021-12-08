import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import json
import cv2
import time
import imageio

#Helper functions

vec_to_rot_matrix = lambda x: R.as_matrix(R.from_rotvec(x))

rot_matrix_to_vec = lambda y: R.as_rotvec(R.from_matrix(y))

rot_x = lambda phi: np.array([
        [1., 0., 0.],
        [0., np.cos(phi), -np.sin(phi)],
        [0., np.sin(phi), np.cos(phi)]], dtype=np.float32)

def skew_matrix_torch(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def add_noise_to_state(state, noise):
    state = state.numpy()
    rot = state[6:15]
    vec = rot_matrix_to_vec(rot.reshape(3, 3))

    condensed_state = np.concatenate((state[:6], vec, state[15:])) + noise

    rot_noised = vec_to_rot_matrix(condensed_state[6:9])

    return torch.tensor(np.concatenate((condensed_state[:6], rot_noised.reshape(-1), condensed_state[9:])), dtype=torch.float32)

class Agent():
    def __init__(self, x0, cfg) -> None:

        #Initialize simulator
        self.path = cfg['path']
        self.half_res = cfg['half_res']
        self.white_bg = cfg['white_bg']

        self.iter = 0

        #Initialized pose
        self.x0 = x0
        self.x = x0

        self.dt = cfg['dt']
        self.g = cfg['g']
        self.mass = cfg['mass']
        self.I = cfg['I']
        self.invI = torch.inverse(self.I)

        self.states_history = [self.x.clone().cpu().detach().numpy().tolist()]

    def reset(self):
        self.x = self.x0
        return

    def step(self, action, noise=None):
        #DYANMICS FUNCTION

        action = action.reshape(-1)

        newstate = self.drone_dynamics(self.x, action)

        if noise is not None:
            newstate_noise = add_noise_to_state(newstate.cpu().clone().detach(), noise)
        else:
            newstate_noise = newstate

        self.x = newstate_noise

        new_state = newstate_noise.clone().cpu().detach().numpy()

        ### IMPORTANT: ACCOUNT FOR CAMERA ORIENTATION WRT DRONE ORIENTATION
        new_pose = np.zeros((4, 4))
        new_pose[:3, :3] = rot_x(np.pi/2) @ (new_state[6:15]).reshape((3, 3))
        new_pose[:3, 3] = new_state[:3]
        new_pose[3, 3] = 1.

        # Write a transform file and receive an image from Blender
        path_to_pose = self.path + f'/{self.iter}.json'
        self.write_transform(new_pose, path_to_pose)
        path_to_img = self.path + f'/{self.iter}.png'
        img = self.listen_img(path_to_img)
        self.img = img
        self.states_history.append(self.x.clone().cpu().detach().numpy().tolist())
        self.iter += 1

        return new_pose, new_state, img

    def state2image(self, state):
        # Directly update the stored state and receive the image
        self.x = state

        new_state = state.clone().cpu().detach().numpy()

        new_pose = np.zeros((4, 4))
        new_pose[:3, :3] = (new_state[6:15]).reshape((3, 3))
        new_pose[:3, 3] = new_state[:3]
        new_pose[3, 3] = 1.

        # Write a transform file and receive an image from Blender
        path_to_pose = self.path + f'/{self.iter}.json'
        self.write_transform(new_pose, path_to_pose)
        path_to_img = self.path + f'/{self.iter}.png'
        img = self.listen_img(path_to_img)
        self.img = img
        self.states_history.append(self.x.clone().cpu().detach().numpy().tolist())

        return new_pose, new_state, img

    def drone_dynamics(self, state, action):
        #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        # and omega are angular rates in the body frame
        next_state = torch.zeros(18)

        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:15]
        R = R_flat.reshape((3, 3))
        omega = state[15:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        dv = (torch.tensor([0,0,-self.mass*self.g]) + R @ sum_action)/self.mass

        # The angular accelerations
        domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        angle = omega*self.dt
        theta = torch.norm(angle, p=2)
        if theta == 0:
            exp_i = torch.eye(3)
        else:
            exp_i = torch.eye(3)
            angle_norm = angle / theta
            K = skew_matrix_torch(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_state[0:3] = pos + v * self.dt
        next_state[3:6] = v + dv * self.dt

        next_state[6:15] = next_R.reshape(-1)

        next_state[15:] = omega + domega * self.dt

        return next_state

    def write_transform(self, transform, filename):
        with open(filename,"w+") as f:
            json.dump(transform.tolist(), f)
        return

    def listen_img(self, filename):
        while os.path.exists(filename) is False:
            time.sleep(0.01)
        time.sleep(.3)
        img = imageio.imread(filename)
        img = (np.array(img) / 255.0).astype(np.float32)
        if self.half_res is True:
            width = int(img.shape[1]//2)
            height = int(img.shape[0]//2)
            dim = (width, height)
  
            # resize image
            img = cv2.resize(img, dim)

        if self.white_bg is True:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])

        img = (np.array(img) * 255.).astype(np.uint8)
        print('Received updated image')
        return img

    def command_sim_reset(self):
        filename = self.path + '/reset.json'
        content = {}
        with open(filename,"w+") as f:
            json.dump(content, f)
        return

    def save_data(self, filename):
        true_states = {}
        true_states['true_states'] = self.states_history
        with open(filename,"w+") as f:
            json.dump(true_states, f)
        return