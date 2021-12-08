import numpy as np
import torch
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked
from nerf.nerf_helpers import get_minibatches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--batch_size", type=int, default=256,
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

class Renderer():
    def __init__(self, render_kwargs):

        self.embed_fn = render_kwargs['embed_fn']
        self.embeddirs_fn = render_kwargs['embeddirs_fn']
        self.chunksize = render_kwargs['chunksize']
        self.network_fn = render_kwargs['model']

    def eval_model(self, pts, viewdirs=torch.tensor([[1., 1., 1.]]).to(device)):
        pts_flat = pts.reshape((-1, pts.shape[-1]))
        embedded = self.embed_fn(pts_flat)
        if self.embeddirs_fn is not None:
            input_dirs = viewdirs.expand(pts.shape)
            input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat((embedded, embedded_dirs), dim=-1)

        batches = get_minibatches(embedded, chunksize=self.chunksize)
        preds = [self.network_fn(batch) for batch in batches]
        radiance_field = torch.cat(preds, dim=0)
        radiance_field = radiance_field.reshape(
            list(pts.shape[:-1]) + [radiance_field.shape[-1]]
        )
        return torch.sigmoid(radiance_field[..., 3]-1)

    def get_density_from_pt(self, pts: TensorType[1, 'N_points', 3], viewdirs=torch.tensor([[1., 1., 1.]])) -> TensorType['N_points']:

        "[N_rays, N_samples, 3] input for pt ([1, N_points, 3]) in this case. View_dir does not matter, but must be given to network. Returns density of size N_points)"

        run_fn = self.network_fn if self.network_fine is None else self.network_fine
        #raw = run_network(pts, fn=run_fn)
        raw = self.network_query_fn(pts, viewdirs, run_fn)

        #Make sure differential densities are non-negative
        # density = F.relu(raw[..., 3])
        density = torch.sigmoid(raw[..., 3] - 1)

        return density.reshape(-1)

    @typechecked
    def get_density(self, points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
        out_shape = points.shape[:-1]
        points = points.reshape(1, -1, 3)

        # +z in nerf is -y in blender
        # +y in nerf is +z in blender
        # +x in nerf is +x in blender
        #mapping = torch.tensor([[1, 0, 0],
        #                        [0, 0, 1],
        #                        [0,-1, 0]], dtype=torch.float)

        #points = points @ mapping.T
        points = points.to(device)

        output = self.eval_model(points)
        return output.reshape(*out_shape)