import numpy as np
import torch

#Helper Functions
def state2pose(vector):
    pose = torch.zeros((4, 4))
    rot_flat = vector[6:15]
    rot = rot_flat.reshape((3, 3))
    trans = vector[:3]

    pose[:3, :3] = rot
    pose[:3, 3] = trans
    pose[3, 3] = 1.

    return pose

def extra_config_parser(parser):
    '''
    Take in the minimum amount required to load in a blender model and run relative_pose_estimation
    '''

    ###FILTER CONFIGS
    parser.add_argument("--dil_iter", type=int, default=3,
                        help='Number of iterations of dilation process')
    parser.add_argument("--kernel_size", type=int, default=5,
                        help='Kernel size for dilation')
    parser.add_argument("--batch_size", type=int, default=512,
                        help='Number of sampled rays per gradient step')
    parser.add_argument("--lrate_relative_pose_estimation", type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument("--sampling_strategy", type=str, default='interest_regions',
                        help='options: random / interest_point / interest_regions')
    parser.add_argument("--reject_thresh", type=float, default=0.6,
                        help='Rejection threshold on per pixel rgb loss')
    parser.add_argument("--N_iter", type=int, default=300,
                        help='Iterations to run optimization per time step')  
    
    parser.add_argument("--sig0", type=list, default=None,
                        help='Initial covariance estimate')  
    parser.add_argument("--Q", type=list, default=None,
                        help='Process noise covariance')  
    parser.add_argument("--R", type=list, default=None,
                        help='Measurement noise covariance')  

    # parameters to define initial pose
    parser.add_argument("--delta_psi", type=float, default=0.0,
                        help='Rotate camera around x axis')
    parser.add_argument("--delta_phi", type=float, default=0.0,
                        help='Rotate camera around z axis')
    parser.add_argument("--delta_theta", type=float, default=0.0,
                        help='Rotate camera around y axis')
    parser.add_argument("--delta_t", type=float, default=0.0,
                        help='translation of camera (negative = zoom in)')
    # apply noise to observed image
    parser.add_argument("--noise", type=str, default='None',
                        help='options: gauss / salt / pepper / sp / poisson')
    parser.add_argument("--sigma", type=float, default=0.01,
                        help='var = sigma^2 of applied noise (variance = std)')
    parser.add_argument("--amount", type=float, default=0.05,
                        help='proportion of image pixels to replace with noise (used in ‘salt’, ‘pepper’, and ‘s&p)')
    parser.add_argument("--delta_brightness", type=float, default=0.0,
                        help='reduce/increase brightness of the observed image, value is in [-1...1]')

    ### PLANNER CONFIGS
    parser.add_argument("--T_final", type=float, default=2.,
                        help='Final time in seconds')
    parser.add_argument("--steps", type=int, default=20,
                        help='Number of time steps')    
    parser.add_argument("--planner_lr", type=float, default=0.01,
                        help='Planner learning rate')    
    parser.add_argument("--epochs_init", type=int, default=2500,
                        help='Initial number of iterations to run planner')    
    parser.add_argument("--fade_out_epoch", type=int, default=0,
                        help='TODO: MICHAL FILL THIS IN')   
    parser.add_argument("--fade_out_sharpness", type=int, default=10,
                        help='TODO: MICHALL FILL THIS IN')  
    parser.add_argument("--epochs_update", type=int, default=250,
                        help='Number of iterations to run planner at each time step')  

    ### AGENT CONFIGS
    parser.add_argument("--mass", type=float, default=1.,
                        help='Mass of agent')  
    parser.add_argument("--g", type=float, default=10.,
                        help='Gravitational constant')  
    parser.add_argument("--I", type=list, default=None,
                        help='Inertia matrix') 
    parser.add_argument("--path", type=str, default=None,
                        help='path to folder interfacing with sim (Blender)')   

    ###MPC CONFIGS
    parser.add_argument("--start_pos", type=list, default=None,
                        help='Starting position')   
    parser.add_argument("--end_pos", type=list, default=None,
                        help='Ending position')   
    parser.add_argument("--start_R", type=list, default=None,
                        help='Starting orientation')   
    parser.add_argument("--end_R", type=list, default=None,
                        help='Ending orientation')   
    parser.add_argument("--mpc_noise_mean", type=list, default=None,
                        help='List of Gaussian noise means to be injected into simulation')   
    parser.add_argument("--mpc_noise_std", type=list, default=None,
                        help='List of Gaussian noise std to be injected into simulation')   

    return parser
