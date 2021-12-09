import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import math
import shutil
import pathlib

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import argparse
import glob
import torch.nn.functional as F
import torchvision
import yaml
#from torch.utils.tensorboard import SummaryWriter

# Import Helper Classes
from estimator_helpers import Estimator
from agent_helpers import Agent
from quad_plot import System
from quad_helpers import vec_to_rot_matrix
from mpc_utils import extra_config_parser, Renderer
from pose_estimate import rot_psi, rot_theta, rot_phi, trans_t
from nerf import (CfgNode, get_embedding_function,
                  load_blender_data, load_llff_data, models)

DEBUG = False
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nerf_filter = True

####################### MAIN LOOP ##########################################
def simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, model_coarse, model_fine, cfg, encode_position_fn, encode_direction_fn):
    '''We've assumed that by calling this function, the NeRF model has already been created (i.e. create_nerf has been called) such that
    such that calling render() returns a valid RGB, etc tensor.

    How trajectory planning works:

    A good initialization for the sequence of poses is returned by running A*. This is only run once! A trajectory loss is computed, consisting of a collision loss
    (querying densities from the NeRF from x,y,z points) and a trust region loss. The outputs are a sequence of future rollout poses (where the planner wants the agent to be)
    and a control action(s) to update the agent. This algorithm is run MPC style, with the intent that A* yields a good initialization for the trajectory, and subsequent optimizations can just be done by
    performing gradient descent on the trajectory loss whilst having good performance. 

    How state estimation works: 

    Given an image, gradient descent is performed on the NeRF reconstruction loss, optimizing on the estimated pose in SE(3). The exponential map was used to create SE(3) from se(3) in R6 such that 
    the transformation is differentiable. Two sampling schemes exist: (1) random sampling of pixels from the full image H x W, or (2) random sampling from a mask around features detected by ORB/SIFT on the
    observed image (termed interest region sampling by iNeRF). 

    How the whole pipeline works:

    The objective is to path plan from pose P0 at time t = 0 to PT at time t = T. At time t, the agent runs the trajectory planning algorithm, yielding a control action(s) and future desired poses P{t+1:T}.
    The agent takes the control action and also receives an image corresponding to the "real" pose at time t + 1. The state estimator uses P{t+1} as the anchor of the tangential plane and returns P_hat_{t+1} = P @ P{t+1},
    where P in SE(3) are the parameters optimized by the state estimator. P_hat_{t+1} is passed to the trajectory planner as the pose estimate. 

    Args:
        

    '''

    start_state = planner_cfg['start_state']
    end_state = planner_cfg['end_state']

    render_kwargs = {
        'embed_fn': encode_position_fn,
        'embeddirs_fn': encode_direction_fn,
        'chunksize': 1500000,
        'model': model_fine
    }

    if DEBUG == False:
        exp_name = planner_cfg['exp_name']
        for i in range(100):

            renderer = Renderer(render_kwargs)
            
            basefolder = "paths" / pathlib.Path(planner_cfg['exp_name'])
            if basefolder.exists():
                print(basefolder, "already exists!")
                if input("Clear it before continuing? [y/N]:").lower() == "y":
                    shutil.rmtree(basefolder)
            basefolder.mkdir()
            (basefolder / "train_poses").mkdir()
            (basefolder / "train_graph").mkdir()
            (basefolder / "execute_poses").mkdir()
            (basefolder / "execute_graph").mkdir()
            print("created", basefolder)

            traj = System(renderer, start_state, end_state, planner_cfg)

            traj.basefolder = basefolder

            then = time.time()
            traj.a_star_init()
            now = time.time()
            print('A* takes', now - then)

            traj.learn_init()
            print('Initial Path takes', time.time() - now)

            agent = Agent(start_state, agent_cfg)

            filter = Estimator(filter_cfg, agent, start_state)
            #inerf_dynamics = Estimator(filter_cfg, agent, start_state, filter=False)

            true_states = start_state.cpu().detach().numpy()

            #steps = traj.get_actions().shape[0]

            '''
            agent_file = './paths/agent_data.json'
            with open(agent_file,"r") as f:
                meta = json.load(f)
                true_states = meta["true_states"]
                true_states = np.array(true_states)

            true_states = true_states[1:]
            '''
            '''
            action_file = './paths/estimator_data.json'
            with open(action_file,"r") as f:
                data_estimator = json.load(f)
                actions = torch.tensor(data_estimator['actions'])
            '''

            #assert len(true_states) == len(actions)
            
            #steps = actions.shape[0]

            ###FOR EXPERIMENTS, TAKE THE FIRST 5 STEPS
            steps = 10

            noise_std = extra_cfg['mpc_noise_std']
            noise_mean = extra_cfg['mpc_noise_mean']

            for iter in trange(steps):
                if iter < steps - 5:
                    action = traj.get_next_action().clone().detach()
                else:
                    action = traj.get_actions()[iter - steps + 5, :]

                #print(traj.get_actions().shape)

                #action = actions[iter]

                noise = np.random.normal(noise_mean, noise_std)
                true_pose, true_state, gt_img = agent.step(action, noise=noise)
                true_states = np.vstack((true_states, true_state))

                #true_pose, true_state, gt_img = agent.state2image(torch.tensor(true_states[iter]))
                #plt.figure()
                #plt.imsave('paths/true/'+ f'{iter}_gt_img.png', gt_img)
                #plt.close()

                #action = torch.tensor(actions[iter])

                #measured_state = estimator.optimize(start_state, sig, gt_img, true_pose)

                #measured_states.append(measured_state.cpu().detach().numpy().tolist())

                torch.cuda.empty_cache()
                then = time.time()
                state_est = filter.estimate_state(gt_img, true_pose, action,
                    model_coarse=model_coarse, model_fine=model_fine,cfg=cfg, encode_position_fn=encode_position_fn,
                    encode_direction_fn=encode_direction_fn)
                now = time.time()
                print('Estimator takes', now-then)

                #state_est_inerf_dyn = inerf_dynamics.estimate_state(gt_img, true_pose, action,
                #    model_coarse=model_coarse, model_fine=model_fine,cfg=cfg, encode_position_fn=encode_position_fn,
                #    encode_direction_fn=encode_direction_fn)

                then = time.time()
                if iter < steps - 5:
                    traj.update_state(state_est)
                    traj.learn_update(iter)
                now = time.time()
                print('Update planner', now - then)

            #plot_trajectory(traj.get_full_states(), true_states)
            filter.save_data(f'paths/{exp_name}/filter_data_{i}.json')
            #inerf_dynamics.save_data(f'paths/{exp_name}/inerf_dyn_data_{i}.json')
            agent.save_data(f'paths/{exp_name}/agent_data_{i}.json')
            agent.command_sim_reset()
            time.sleep(0.1)

        return

    else:
        ####################################### DEBUGING ENVIRONMENT ####################################################3
        '''
        to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
        renderer = Renderer(hwf, K, chunk, render_kwargs_train)
        
        #Read in poses
        data_path = './paths/stonehenge_every_grad_step/'
        gt_data_path = data_path + 'agent_data.json'
        est_data_path = data_path + 'estimator_data.json'
        gt_img_path = data_path + 'true/'
        with open(gt_data_path,"r") as f:
            meta = json.load(f)
            true_states = meta["true_states"]
            true_states = np.array(true_states)

        with open(est_data_path,"r") as f:
            data_estimator1 = json.load(f)

            pixel_losses1 = []
            dyn_losses1 = []
            rot_errors1 = np.empty((0, 3))
            trans_errors1 = np.empty((0, 3))
            state_estimates1 = data_estimator1['state_estimates']
            covariances1 = data_estimator1['covariance']
            states1 = np.empty((0, 18))
            predicted_states1 = data_estimator1['predicted_states']
            actions1 = data_estimator1['actions']

            for iter, val in enumerate(data_estimator1["iterations"]):
                pixel_losses1 += data_estimator1['pixel_losses'][f'{iter}']
                dyn_losses1 += data_estimator1['dyn_losses'][f'{iter}']

                rot_errors1 = np.vstack((rot_errors1, np.array(data_estimator1['rot_errors'][f'{iter}'])))
                trans_errors1 = np.vstack((trans_errors1, np.array(data_estimator1['trans_errors'][f'{iter}'])))
                states1 = np.vstack((states1, np.array(data_estimator1['states'][f'{iter}'])))

        for it, state_est in enumerate(state_estimates1):
            testsavedir = data_path + f'gif_step{it}'
            os.makedirs(testsavedir, exist_ok=True)

            obs_img = imageio.imread(gt_img_path + f'{it}_gt_img.png')
            obs_img = obs_img[..., :3]
            obs_img = (np.array(obs_img) / 255.).astype(np.float32)

            imgs = []

            with torch.no_grad():
                for iter, state in enumerate(data_estimator1['states'][f'{it}']):
                    if iter < 60:
                        print('Iteration', iter, 'Image Number', it)

                        state = torch.tensor(state)
                        pose = state2pose(state)
                        sim_pose = convert_blender_to_sim_pose(pose.cpu().detach().numpy())
                        rgb = renderer.get_img_from_pose(torch.tensor(sim_pose))
                        rgb = rgb.cpu().detach().numpy()
                        rgb8 = to8b(rgb)
                        ref = to8b(obs_img)

                        print(rgb.shape)
                        print(ref.shape)
                        filename = os.path.join(testsavedir, str(iter)+'.png')
                        #dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                        #imageio.imwrite(filename, dst)
                        #imgs.append(dst)

                        imageio.imwrite(filename, rgb8)
                        imgs.append(rgb8)

            #imageio.mimwrite(os.path.join(testsavedir, 'video.gif'), imgs, fps=8) #quality = 8 for mp4 format
        '''
        pass
    return

####################### END OF MAIN LOOP ##########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )

    parser = extra_config_parser(parser)
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None

    #TODO: Implement CACHED DATASET!
    '''
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        '''
    # Load dataset
    images, poses, render_poses, hwf = None, None, None, None
    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    elif cfg.dataset.type.lower() == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if cfg.dataset.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(images.shape[0])
                if (i not in i_test and i not in i_val)
            ]
        )
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    '''
    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    '''
    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        start_iter = checkpoint["iter"]

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    ### PLANNER CONFIG DETAILS
    # renderer = get_nerf('configs/stonehenge.txt')
    # stonehenge - simple
    #start_pos = torch.tensor([-0.9,-0.9, 0.])
    #start_pos = torch.tensor([0.9,-0.2, 0.2])
    #start_pos   = torch.tensor([-0.31,-0.9, 0.])
    #end_pos   = torch.tensor([-0.2,0.55, 0.3])
    #end_pos   = torch.tensor([-0.3,0.5, 0.4])
    #end_pos   = torch.tensor([-0.55, 0.6, 0.4])
    # start_pos = torch.tensor([-1, 0, 0.2])
    # end_pos   = torch.tensor([ 1, 0, 0.5])

    #playground
    #start_pos = torch.tensor([0.7,-0.2, 0.4])
    #end_pos   = torch.tensor([-0.35,0.55, 0.4])

    #stonehenge
    #start_pos = [0.39, -0.67, 0.2]
    #end_pos = [-0.4, 0.55, 0.16]

    #CHURCH
    #start_pos = torch.tensor([-1.1,-0.8, 0.6])
    #end_pos   = torch.tensor([-1.64,-0.73, 0.59])

    #Violin
    #start_pos = torch.tensor([-0.9,-0.9, 0.2])
    #end_pos   = torch.tensor([0.4,0.75, 0.15])

    #Kings Hall
    #start_pos = torch.tensor([-0.12,-0.65, -0.24])
    #end_pos   = torch.tensor([-.1,0.33, -0.25])

    #start_R = vec_to_rot_matrix( torch.tensor([0.0,0.0, .3]))
    #end_R = vec_to_rot_matrix(torch.tensor([0.,0.0, 0.]))

    start_pos = torch.tensor(cfg_dict['start_pos']).float()
    end_pos = torch.tensor(cfg_dict['end_pos']).float()

    start_R = vec_to_rot_matrix( torch.tensor(cfg_dict['start_R']))
    end_R = vec_to_rot_matrix(torch.tensor(cfg_dict['end_R']))

    ### ASSUME ZERO INITIAL RATES
    init_rates = torch.zeros(3)
    start_state = torch.cat( [start_pos, init_rates, start_R.reshape(-1), init_rates], dim=0 )
    end_state   = torch.cat( [end_pos,   init_rates, end_R.reshape(-1), init_rates], dim=0 )

    ### PLANNER CONFIGS
    planner_cfg = {"T_final": cfg_dict['T_final'],
            "steps": cfg_dict['steps'],
            "lr": cfg_dict['planner_lr'],
            "epochs_init": cfg_dict['epochs_init'],
            "fade_out_epoch": cfg_dict['fade_out_epoch'],
            "fade_out_sharpness": cfg_dict['fade_out_sharpness'],
            "epochs_update": cfg_dict['epochs_update'],
            'start_state': start_state.to(device),
            'end_state': end_state.to(device),
            'exp_name': cfg.experiment.id
            }

    ### AGENT CONFIGS
    agent_cfg = {'dt': planner_cfg["T_final"]/planner_cfg["steps"],
                'mass': cfg_dict['mass'],
                'g': cfg_dict['g'],
                'I': torch.tensor(cfg_dict['I']).float().to(device),
                'path': cfg_dict['path'], 
                'half_res': cfg.dataset.half_res, 
                'white_bg': cfg.nerf.train.white_background}

    ### FILTER CONFIGS
    filter_cfg = {
        'dil_iter': cfg_dict['dil_iter'],
        'batch_size': cfg_dict['batch_size'],
        'kernel_size': cfg_dict['kernel_size'],
        'lrate': cfg_dict['lrate_relative_pose_estimation'],
        'sampling_strategy': cfg_dict['sampling_strategy'],
        'reject_thresh': cfg_dict['reject_thresh'],
        'N_iter': cfg_dict['N_iter'],
        'sig0': torch.tensor(cfg_dict['sig0']).float().to(device),
        'Q': torch.tensor(cfg_dict['Q']).float().to(device),
        'R': torch.tensor(cfg_dict['R']).float().to(device),
        'H': H,
        'W': W,
        'focal': focal
    }

    ### EXTRA CONFIGS
    extra_cfg = {
        'mpc_noise_std': np.array([float(i) for i in cfg_dict['mpc_noise_std']]),
        'mpc_noise_mean': np.array([float(i) for i in cfg_dict['mpc_noise_mean']])
    }

    simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, model_coarse, model_fine, cfg, encode_position_fn, encode_direction_fn)
