import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F
import torchvision
import json
import time
from matplotlib import pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from lietorch import SE3, LieGroupParameter
from scipy.spatial.transform import Rotation as R
import cv2

from nerf import (get_ray_bundle, run_one_iter_of_nerf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mahalanobis(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
    return m

rot_x = lambda phi: torch.tensor([
        [1., 0., 0.],
        [0., torch.cos(phi), -torch.sin(phi)],
        [0., torch.sin(phi), torch.cos(phi)]], dtype=torch.float32)

rot_x_np = lambda phi: np.array([
        [1., 0., 0.],
        [0., np.cos(phi), -np.sin(phi)],
        [0., np.sin(phi), np.cos(phi)]], dtype=np.float32)

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])

def SE3_to_trans_and_quat(data):
    rot = data[:3, :3]
    trans = data[:3, 3]

    r = R.from_matrix(rot)
    quat = r.as_quat()
    return np.concatenate([trans, quat])

def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    #img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #sift = cv2.SIFT_create()
    #keypoints = sift.detect(img, None)

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    keypoints2 = orb.detect(img,None)

    #if DEBUG:
    #    img = cv2.drawKeypoints(img_gray, keypoints, img)
    #keypoints = keypoints + keypoints2
    keypoints = keypoints2

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

class Estimator():
    def __init__(self, filter_cfg, agent, start_state, filter=True) -> None:
        
    # Parameters
        self.batch_size = filter_cfg['batch_size']
        self.kernel_size = filter_cfg['kernel_size']
        self.dil_iter = filter_cfg['dil_iter']

        self.lrate = filter_cfg['lrate']
        self.sampling_strategy = filter_cfg['sampling_strategy']
        self.reject_thresh = filter_cfg['reject_thresh']

        self.agent = agent

        self.is_filter = filter

        #State initial estimate at time t=0
        self.xt = start_state                   #Size 18
        self.sig = 1e-1*torch.eye(start_state.shape[0])
        self.Q = 1e-1*torch.eye(start_state.shape[0])
        #self.sig = filter_cfg['sig0']          #State covariance 18x18
        #self.Q = filter_cfg['Q']              #Process noise covariance
        self.R = filter_cfg['R']               #Measurement covariance
        self.iter = filter_cfg['N_iter']

        #NERF SPECIFIC CONFIGS
        # create meshgrid from the observed image
        self.W, self.H, self.focal = filter_cfg['W'], filter_cfg['H'], filter_cfg['focal']

        #self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.W - 1, self.W), np.linspace(0, self.H - 1, self.H)), -1),
        #                    dtype=int)

        #Storage for plots
        self.pixel_losses = {}
        self.dyn_losses = {}
        self.covariance = []
        self.state_estimates = []
        self.states = {}
        self.predicted_states = []
        self.actions = []

        self.iteration = 0

    def estimate_relative_pose(self, sensor_image, start_state, sig, obs_img_pose=None, obs_img=None, model_coarse=None, model_fine=None,cfg=None,
    encode_position_fn=None, encode_direction_fn=None):

        b_print_comparison_metrics = obs_img_pose is not None
        b_generate_overlaid_images = b_print_comparison_metrics and obs_img is not None

        obs_img_noised = sensor_image
        W_obs = sensor_image.shape[0]
        H_obs = sensor_image.shape[1]

        # find points of interest of the observed image
        POI = find_POI(obs_img_noised, False)  # xy pixel coordinates of points of interest (N x 2)

        ### IF FEATURE DETECTION CANT FIND POINTS, RETURN INITIAL
        if len(POI.shape) == 1:
            self.pixel_losses[f'{self.iteration}'] = []
            self.dyn_losses[f'{self.iteration}'] = []
            self.states[f'{self.iteration}'] = []
            return start_state.clone().detach(), False

        obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)
        obs_img_noised = torch.tensor(obs_img_noised).cuda()

        #sensor_image[POI[:, 1], POI[:, 0]] = [0, 255, 0]

        # create meshgrid from the observed image
        coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W_obs - 1, W_obs), np.linspace(0, H_obs - 1, H_obs)), -1), dtype=int)

        # create sampling mask for interest region sampling strategy
        interest_regions = np.zeros((H_obs, W_obs, ), dtype=np.uint8)
        interest_regions[POI[:,1], POI[:,0]] = 1
        I = self.dil_iter
        interest_regions = cv2.dilate(interest_regions, np.ones((self.kernel_size, self.kernel_size), np.uint8), iterations=I)
        interest_regions = np.array(interest_regions, dtype=bool)
        interest_regions = coords[interest_regions]

        # not_POI contains all points except of POI
        coords = coords.reshape(H_obs * W_obs, 2)
        #not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
        #not_POI = np.array([list(point) for point in not_POI]).astype(int)

        #Break up state into components
        start_trans = start_state[:3].reshape((3, 1))

        ### IMPORTANT: ROTATION MATRIX IS ROTATED BY SOME AMOUNT TO ACCOUNT FOR CAMERA ORIENTATION
        start_rot = rot_x_np(np.pi/2) @ start_state[6:15].reshape((3, 3))
        start_pose = np.concatenate((start_rot, start_trans), axis=1)

        start_vel = torch.tensor(start_state[3:6]).cuda()
        start_omega = torch.tensor(start_state[15:]).cuda()   

        # Create pose transformation model
        start_pose = SE3_to_trans_and_quat(start_pose)

        starting_pose = SE3(torch.from_numpy(start_pose).float().cuda())
        starting_pose = LieGroupParameter(starting_pose).cuda()

        #print('Start pose', start_pose, start_vel, start_omega)

        # Add velocities, omegas, and pose object to optimizer
        if self.is_filter is True:
            optimizer = torch.optim.Adam(params=[starting_pose, start_vel, start_omega], lr=self.lrate, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adam(params=[starting_pose], lr=self.lrate, betas=(0.9, 0.999))

        # calculate angles and translation of the observed image's pose
        if b_print_comparison_metrics:
            phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
            theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
            psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
            translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)

        #Store data
        pix_losses = []
        dyn_losses = []
        states = []

        for k in range(self.iter):
            model_coarse.eval()
            if model_fine:
                model_fine.eval()

            rgb_coarse, rgb_fine = None, None

            # TODO: IMPLEMENT INERF WITH USE_CACHED DATSET!!!

            rand_inds = np.random.choice(interest_regions.shape[0], size=self.batch_size, replace=False)
            batch = interest_regions[rand_inds]

            target_s = obs_img_noised[batch[:, 1], batch[:, 0]]
            #target_s = torch.Tensor(target_s).to(device)

            pose = starting_pose.retr().matrix()[:3, :4]

            ray_origins, ray_directions = get_ray_bundle(self.H, self.W, self.focal, pose)  # (H, W, 3), (H, W, 3)
            #with torch.no_grad():
            #    r_o, r_d = ray_origins, ray_directions

            #print('Ray origins cuda', ray_origins.is_cuda)
            ray_origins = ray_origins[batch[:, 1], batch[:, 0], :]
            ray_directions = ray_directions[batch[:, 1], batch[:, 0], :]

            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                self.H,
                self.W,
                self.focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            #target_ray_values = target_s

            #print(time.time() - then)

            ### OUTLIER REJECTION
            threshold = self.reject_thresh
            with torch.no_grad():
                coarse_sample_loss = torch.sum(torch.abs(rgb_coarse[..., :3] - target_s[..., :3]), 1)/3
                fine_sample_loss = torch.sum(torch.abs(rgb_fine[..., :3] - target_s[..., :3]), 1)/3
                csl = F.relu(-(coarse_sample_loss-threshold))
                fsl = F.relu(-(fine_sample_loss-threshold))
                coarse_ind = torch.nonzero(csl)
                fine_ind = torch.nonzero(fsl)
            ### ---------------- ###
            
            coarse_loss = torch.nn.functional.mse_loss(
                rgb_coarse[coarse_ind, :3], target_s[coarse_ind, :3]
            )
            fine_loss = None
            if rgb_fine is not None:
                fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine[fine_ind, :3], target_s[fine_ind, :3]
                )
            
            loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)

            pix_losses.append(loss.clone().cpu().detach().numpy().tolist())
            #Add dynamics loss
            state = torch.cat((pose[:3, 3], start_vel, (rot_x(torch.tensor(-np.pi/2)) @ pose[:3, :3]).reshape(-1), start_omega), dim=0)
            dyn_loss = mahalanobis(state, torch.tensor(start_state), sig)

            states.append(state.clone().cpu().detach().numpy().tolist())
            dyn_losses.append(dyn_loss.clone().cpu().detach().numpy().tolist())

            if self.is_filter is True:
                loss += dyn_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            new_lrate = self.lrate * (0.8 ** ((k + 1) / 100))
            #new_lrate = extra_arg_dict['lrate'] * np.exp(-(k)/1000)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            # print results periodically
            if b_print_comparison_metrics and ((k + 1) % 20 == 0 or k == 0):
                print('Step: ', k)
                print('Loss: ', loss)

                with torch.no_grad():
                    pose_dummy = starting_pose.retr().matrix().clone().cpu().detach().numpy()
                    # calculate angles and translation of the optimized pose
                    phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                    theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                    psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                    translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                    #translation = pose_dummy[2, 3]
                    # calculate error between optimized and observed pose
                    phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                    theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                    psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                    rot_error = phi_error + theta_error + psi_error
                    translation_error = abs(translation_ref - translation)
                    print('Rotation error: ', rot_error)
                    print('Translation error: ', translation_error)
                    print('Number of rays accepted', len(fine_ind))
                    print('-----------------------------------')
                    
                    '''
                    if (k+1) % 100 == 0:
                        _, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                            self.H,
                            self.W,
                            self.focal,
                            model_coarse,
                            model_fine,
                            r_o,
                            r_d,
                            cfg,
                            mode="validation",
                            encode_position_fn=encode_position_fn,
                            encode_direction_fn=encode_direction_fn,
                        )

                        rgb = rgb_fine.cpu().detach().numpy()
                        f, axarr = plt.subplots(2)
                        axarr[0].imshow(rgb)
                        axarr[1].imshow(sensor_image)
                        plt.show()
                    '''

        print("Done with main relative_pose_estimation loop")
        self.target_s = target_s.detach()
        self.batch = batch
        self.pixel_losses[f'{self.iteration}'] = pix_losses
        self.dyn_losses[f'{self.iteration}'] = dyn_losses
        self.states[f'{self.iteration}'] = states
        return state.clone().detach(), True
        
    def measurement_function(self, state, start_state, sig, model_coarse=None, model_fine=None,cfg=None,
    encode_position_fn=None, encode_direction_fn=None):
        target_s = self.target_s
        batch = self.batch

        # Breaking state into pieces
        rot_mat = rot_x(torch.tensor(np.pi/2)) @ state[6:15].reshape((3, 3))
        trans = state[:3].reshape((3, 1))
        pose_mat = torch.cat((rot_mat, trans), dim=1)

        #Process loss. 
        loss_dyn = mahalanobis(state, torch.tensor(start_state), sig)

        #TODO: CONVERT STATE INTO POSE
        ray_origins, ray_directions = get_ray_bundle(self.H, self.W, self.focal, pose_mat)  # (H, W, 3), (H, W, 3)

        ray_origins = ray_origins[batch[:, 1], batch[:, 0], :]
        ray_directions = ray_directions[batch[:, 1], batch[:, 0], :]

        rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
            self.H,
            self.W,
            self.focal,
            model_coarse,
            model_fine,
            ray_origins,
            ray_directions,
            cfg,
            mode="validation",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )
        target_ray_values = target_s

        
        ### OUTLIER REJECTION
        threshold = self.reject_thresh
        with torch.no_grad():
            coarse_sample_loss = torch.sum(torch.abs(rgb_coarse[..., :3] - target_ray_values[..., :3]), 1)/3
            fine_sample_loss = torch.sum(torch.abs(rgb_fine[..., :3] - target_ray_values[..., :3]), 1)/3
            csl = F.relu(-(coarse_sample_loss-threshold))
            fsl = F.relu(-(fine_sample_loss-threshold))
            coarse_ind = torch.nonzero(csl)
            fine_ind = torch.nonzero(fsl)
        ### ---------------- ###
        
        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[coarse_ind, :3], target_ray_values[coarse_ind, :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[fine_ind, :3], target_ray_values[fine_ind, :3]
            )

        loss_rgb = coarse_loss + (fine_loss if fine_loss is not None else 0.0)

        loss = loss_rgb + loss_dyn

        return loss

    def estimate_state(self, sensor_img, obs_img_pose, action, model_coarse=None, model_fine=None,cfg=None,
        encode_position_fn=None, encode_direction_fn=None):
        # Computes Jacobian w.r.t dynamics are time t-1. Then update state covariance Sig_{t|t-1}.
        # Perform grad. descent on J = measurement loss + process loss
        # Compute state covariance Sig_{t} by hessian at state at time t.

        #with torch.no_grad():
        #Propagated dynamics. x t|t-1
        start_state = self.agent.drone_dynamics(self.xt, action)
        start_state = start_state.cpu().numpy()

        #State estimate at t-1 is self.xt. Find jacobian wrt dynamics
        t1 = time.time()
    
        A = torch.autograd.functional.jacobian(lambda x: self.agent.drone_dynamics(x, action), self.xt)

        #with torch.no_grad():
        t2 = time.time()
        #print('Elapsed time for Jacobian', t2-t1)

        #Propagate covariance
        sig_prop = A @ self.sig @ A.T + self.Q

        #Argmin of total cost. Encapsulate this argmin optimization as a function call
        then = time.time()
        xt, success_flag = self.estimate_relative_pose(sensor_img, start_state, sig_prop, obs_img_pose=obs_img_pose, obs_img=None,
            model_coarse=model_coarse, model_fine=model_fine,cfg=cfg, encode_position_fn=encode_position_fn, encode_direction_fn=encode_direction_fn)
        
        print('Optimization step for filter', time.time()-then)
        #with torch.no_grad():
        #Update state estimate
        self.xt = xt

        #Hessian to get updated covariance
        t3 = time.time()
        
        if self.is_filter is True and success_flag is True:
            hess = torch.autograd.functional.hessian(lambda x: self.measurement_function(x, start_state, sig_prop, model_coarse=model_coarse, 
                model_fine=model_fine,cfg=cfg, encode_position_fn=encode_position_fn, encode_direction_fn=encode_direction_fn), self.xt)

            #with torch.no_grad():
            #Turn covariance into positive definite
            hess_np = hess.clone().cpu().detach().numpy()
            hess = nearestPD(hess_np)

            t4 = time.time()
            print('Elapsed time for hessian', t4-t3)

            #self.sig_det.append(np.linalg.det(sig.cpu().numpy()))

            #Update state covariance
            self.sig = torch.inverse(torch.tensor(hess))

                #print(self.sig)

                #print('Start state', start_state)

        self.actions.append(action.clone().cpu().detach().numpy().tolist())
        self.predicted_states.append(start_state.tolist())
        self.covariance.append(self.sig.clone().cpu().detach().numpy().tolist())
        self.state_estimates.append(self.xt.clone().cpu().detach().numpy().tolist())

        self.iteration += 1
        return self.xt.clone().detach()

    def save_data(self, filename):
        data = {}

        data['pixel_losses'] = self.pixel_losses
        data['dyn_losses'] = self.dyn_losses
        data['covariance'] = self.covariance
        data['state_estimates'] = self.state_estimates
        data['states'] = self.states
        data['predicted_states'] = self.predicted_states
        data['actions'] = self.actions

        with open(filename,"w+") as f:
            json.dump(data, f)
        return
