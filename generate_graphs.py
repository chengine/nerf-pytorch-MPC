import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from scipy.spatial.transform import Rotation
import matplotlib.ticker as ticker
import os

def convert_to_rot_vector(input):
    vect = np.zeros((len(input), 3))
    for it, val in enumerate(input):
        rot_matrix = val[6:15].reshape((3, 3))
        rot = Rotation.from_matrix(rot_matrix)
        vect[it, :] = rot.as_rotvec()

    return vect

def find_angle_and_dist(gt, states, predicted):
    #input1 is the GT at every time step
    #input2 is a dictionary across time steps of the estimate at every gradient step

    angles = []
    trans = []
    input1 = gt[1:]

    for iter, (GT_state, predict) in enumerate(zip(input1, predicted)):
        if len(states[f'{iter}']) != 0:
            states_at_time_step = states[f'{iter}']
        else:
            states_at_time_step = np.tile(predict, (300, 1))
        GT_rot = GT_state[6:15].reshape((3, 3))
        GT_trans = GT_state[:3]
        for row in states_at_time_step:
            row_rot = np.array(row[6:15]).reshape((3, 3))
            row_trans = np.array(row[:3])
            angles.append(find_angle_diff(GT_rot, row_rot))

            trans.append(np.linalg.norm(row_trans-GT_trans, ord=2))
    return angles, trans

def find_angle_diff(input1, input2):
    rotation = input1 @ input2.T
    rot = Rotation.from_matrix(rotation)
    rot_angle = rot.as_rotvec()

    return np.linalg.norm(rot_angle, ord=2)

def find_l2_norm(input1, input2, predicted):
    input1 = input1[1:]
    vel_error = []
    omega_error = []

    for iter, (GT_state, predict) in enumerate(zip(input1, predicted)):
        if len(input2[f'{iter}']) != 0:
            states_at_time_step = input2[f'{iter}']
        else:
            states_at_time_step = np.tile(predict, (300, 1))
        GT_vel = GT_state[3:6]
        GT_omega = GT_state[15:]

        for row in states_at_time_step:
            row_vel = np.array(row[3:6])
            row_omega = np.array(row[15:])
            vel_error.append(np.linalg.norm(GT_vel - row_vel, ord=2))

            omega_error.append(np.linalg.norm(GT_omega-row_omega, ord=2))

    return vel_error, omega_error

#Name of folder
base_dir = 'paths/'
exp_name = 'stonehenge_data/'
agent_name = 'agent_data_'
filter_name = 'filter_data_'
inerf_dyn_name = 'inerf_dyn_data_'

save_dir = base_dir + exp_name + 'plots/'

agent_data = []
filter_data = []
inerf_dyn_data = []

iter = 0
while True:
    #Load GT Agent
    agent_file = base_dir + exp_name + agent_name + f'{iter}.json'
    filter_file = base_dir + exp_name + filter_name + f'{iter}.json'
    inerf_dyn_file = base_dir + exp_name + inerf_dyn_name + f'{iter}.json'

    if os.path.exists(agent_file) is True:
        agent_dict = {}
        with open(agent_file,"r") as f:
            meta = json.load(f)
            true_states = meta["true_states"]
            true_states = np.array(true_states)
        agent_dict['GT'] = true_states

        filter_dict = {}
        #Load Filter
        with open(filter_file,"r") as f:
            meta = json.load(f)

            # STORED AS LISTS
            state_est = np.array(meta['state_estimates'])
            covariance = np.array(meta['covariance'])
            predicted_state = np.array(meta['predicted_states'])
            action = np.array(meta['actions'])

            # STORED AS DICTIONARIES WITH KEY IS ITERATION NUMBER
            states = meta['states']
            ploss = meta['pixel_losses']
            dloss = meta['dyn_losses']
        filter_dict['ploss'] = ploss
        filter_dict['dloss'] = dloss
        filter_dict['estimate'] = state_est
        filter_dict['predicted'] = predicted_state
        filter_dict['actions'] = action
        filter_dict['cov'] = covariance
        filter_dict['states'] = states

        inerf_dyn_dict = {}
        #Load Inerf Dynamical
        with open(inerf_dyn_file,"r") as f:
            meta = json.load(f)

            # STORED AS LISTS
            state_est = np.array(meta['state_estimates'])
            covariance = np.array(meta['covariance'])
            predicted_state = np.array(meta['predicted_states'])
            action = np.array(meta['actions'])

            # STORED AS DICTIONARIES WITH KEY IS ITERATION NUMBER
            states = meta['states']
            ploss = meta['pixel_losses']
            dloss = meta['dyn_losses']
        inerf_dyn_dict['ploss'] = ploss
        inerf_dyn_dict['dloss'] = dloss
        inerf_dyn_dict['estimate'] = state_est
        inerf_dyn_dict['predicted'] = predicted_state
        inerf_dyn_dict['actions'] = action
        inerf_dyn_dict['cov'] = covariance
        inerf_dyn_dict['states'] = states

        agent_data.append(agent_dict)
        filter_data.append(filter_dict)
        inerf_dyn_data.append(inerf_dyn_dict)

        iter += 1
    else:
        break

print('Successfully read in data')

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

'''
#Plot Losses
fig, ax = plt.subplots(2, figsize=(12, 12), dpi=80)
ax[0].plot(list(range(len(pixel_losses))), pixel_losses)
ax[1].plot(list(range(len(dyn_losses))), dyn_losses)
ax[0].set_title('Photometric Loss')
ax[1].set_title('Process Loss')

ax[0].set_xticks(np.arange(0, len(pixel_losses), 16))
for ind, label in enumerate(ax[0].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[1].set_xticks(np.arange(0, len(pixel_losses), 16))
for ind, label in enumerate(ax[1].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)
ax[0].grid()
ax[1].grid()
fig.savefig(pathname+'losses.png')
'''

'''
#Plot errors
rot_errors = np.array(rot_errors).reshape((-1, 3))

trans_errors = np.array(trans_errors).reshape((-1, 3))

fig, ax = plt.subplots(2, figsize=(12, 12), dpi=80)
ax[0].plot(list(range(len(rot_errors))), rot_errors[:, 0])
ax[0].plot(list(range(len(rot_errors))), rot_errors[:, 1])
ax[0].plot(list(range(len(rot_errors))), rot_errors[:, 2])

ax[1].plot(list(range(len(trans_errors))), trans_errors[:, 0])
ax[1].plot(list(range(len(trans_errors))), trans_errors[:, 1])
ax[1].plot(list(range(len(trans_errors))), trans_errors[:, 2])

ax[0].set_title('Rotational Error')
ax[1].set_title('Translational Error')


ax[0].set_xticks(np.arange(0, len(rot_errors), 16))
for ind, label in enumerate(ax[0].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[1].set_xticks(np.arange(0, len(trans_errors), 16))
for ind, label in enumerate(ax[1].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)
ax[0].grid()
ax[1].grid()
fig.savefig(pathname+'errors.png')
'''
'''
#Plot velocity and angular rates
vel1, omega1 = find_l2_norm(true_states, data_estimator1['states'])
vel2, omega2 = find_l2_norm(true_states, data_estimator2['states'])
vel3, omega3 = find_l2_norm(true_states, data_estimator3['states'])

fig, ax = plt.subplots(2, figsize=(10, 10), dpi=80)
ax[0].plot(list(range(len(vel1))), vel1, label='Filter')
ax[0].plot(list(range(len(vel2))), vel2, label='Filter w/o process loss')
ax[0].plot(list(range(len(vel3))), vel3, label='iNeRF')

ax[1].plot(list(range(len(omega1))), omega1, label='Filter')
ax[1].plot(list(range(len(omega2))), omega2, label='Filter w/o process loss')
ax[1].plot(list(range(len(omega3))), omega3, label='iNeRF')

ax[0].set_title('Velocity Error')
ax[1].set_title(r'$\omega$ Error')

ax[0].set_xticks(np.arange(0, len(vel1), 16))
for ind, label in enumerate(ax[0].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[1].set_xticks(np.arange(0, len(omega1), 16))
for ind, label in enumerate(ax[1].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/16))
ax[0].xaxis.set_major_formatter(ticks)
ax[1].xaxis.set_major_formatter(ticks)
#ax[0].grid()
#ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[0].set_ylim([0, 0.05])
ax[0].set_xlim([0, len(omega1)])
ax[1].set_ylim([0, 0.1])
ax[1].set_xlim([0, len(omega1)])
#ax[0].set_xlabel('Trajectory Time')
ax[1].set_xlabel('Trajectory Time')
fig.savefig('./paths/rates.png')
'''

a_error_fil = []
p_error_fil = []
a_error_dyn = []
p_error_dyn = []
v_fil = []
o_fil = []
v_dyn = []
o_dyn = []

for i, (ag, fil, dyn) in enumerate(zip(agent_data, filter_data, inerf_dyn_data)):
    true_states = ag['GT']

    states_fil = fil['states']
    predict_fil = fil['predicted']

    states_dyn = dyn['states']
    predict_dyn = dyn['predicted']

    #Plot errors as single value
    ang_error_filter, pos_error_filter = find_angle_and_dist(true_states, states_fil, predict_fil)
    ang_error_inerf_dyn, pos_error_inerf_dyn = find_angle_and_dist(true_states, states_dyn, predict_dyn)
    #angle_error3, pos_errors3 = find_angle_and_dist(true_states, data_estimator3['states'])

    vel_fil, omega_fil = find_l2_norm(true_states, states_fil, predict_fil)
    vel_dyn, omega_dyn = find_l2_norm(true_states, states_dyn, predict_dyn)
    #vel3, omega3 = find_l2_norm(true_states, data_estimator3['states'])

    a_error_fil.append(ang_error_filter)
    a_error_dyn.append(ang_error_inerf_dyn)
    p_error_fil.append(pos_error_filter)
    p_error_dyn.append(pos_error_inerf_dyn)
    v_fil.append(vel_fil)
    v_dyn.append(vel_dyn)
    o_fil.append(omega_fil)
    o_dyn .append(omega_dyn)

ang_error_filter = np.mean(np.array(a_error_fil), axis=0)
ang_error_inerf_dyn = np.mean(np.array(a_error_dyn), axis=0)
pos_error_filter = np.mean(np.array(p_error_fil), axis=0)
pos_error_inerf_dyn = np.mean(np.array(p_error_dyn), axis=0)
vel_fil = np.mean(np.array(v_fil), axis=0)
vel_dyn = np.mean(np.array(v_dyn), axis=0)
omega_fil = np.mean(np.array(o_fil), axis=0)
omega_dyn = np.mean(np.array(o_dyn), axis=0)

ang_error_filter_std = np.std(np.array(a_error_fil), axis=0)
ang_error_inerf_dyn_std = np.std(np.array(a_error_dyn), axis=0)
pos_error_filter_std = np.std(np.array(p_error_fil), axis=0)
pos_error_inerf_dyn_std = np.std(np.array(p_error_dyn), axis=0)
vel_fil_std = np.std(np.array(v_fil), axis=0)
vel_dyn_std = np.std(np.array(v_dyn), axis=0)
omega_fil_std = np.std(np.array(o_fil), axis=0)
omega_dyn_std = np.std(np.array(o_dyn), axis=0)

ang_error_filter_pos = ang_error_filter + ang_error_filter_std
ang_error_inerf_dyn_pos = ang_error_inerf_dyn + ang_error_inerf_dyn_std
pos_error_filter_pos = pos_error_filter + pos_error_filter_std
pos_error_inerf_dyn_pos = pos_error_inerf_dyn + pos_error_inerf_dyn_std
vel_fil_pos = vel_fil + vel_fil_std
vel_dyn_pos = vel_dyn + vel_dyn_std
omega_fil_pos = omega_fil + omega_fil_std
omega_dyn_pos = omega_dyn + omega_dyn_std

ang_error_filter_neg = ang_error_filter - ang_error_filter_std
ang_error_inerf_dyn_neg = ang_error_inerf_dyn - ang_error_inerf_dyn_std
pos_error_filter_neg = pos_error_filter - pos_error_filter_std
pos_error_inerf_dyn_neg = pos_error_inerf_dyn - pos_error_inerf_dyn_std
vel_fil_neg = vel_fil - vel_fil_std
vel_dyn_neg = vel_dyn - vel_dyn_std
omega_fil_neg = omega_fil - omega_fil_std
omega_dyn_neg = omega_dyn - omega_dyn_std

fig, ax = plt.subplots(2, 2, figsize=(45, 30), dpi=200)
# ANGULAR ERRORS
ax[0, 0].plot(list(range(len(ang_error_filter))), ang_error_filter, label='Filter', color='red', linewidth=5)
ax[0, 0].plot(list(range(len(ang_error_inerf_dyn))), ang_error_inerf_dyn, label='INeRF W/ Dynamics',  color='green', linewidth=5)
#ax[0].plot(list(range(len(angle_error1))), angle_error3, label='iNeRF')

#ax[0].plot(list(range(len(ang_error_filter_pos))), ang_error_filter_pos, label='Filter Pos')
#ax[0].plot(list(range(len(ang_error_inerf_dyn_pos))), ang_error_inerf_dyn_pos, label='INeRF W/ Dynamics Pos')

#ax[0].plot(list(range(len(ang_error_filter_neg))), ang_error_filter_neg, label='Filter Neg')
#ax[0].plot(list(range(len(ang_error_inerf_dyn_neg))), ang_error_inerf_dyn_neg, label='INeRF W/ Dynamics Neg')

ax[0, 0].fill_between(list(range(len(ang_error_filter_pos))), ang_error_filter_pos, ang_error_filter_neg, color='red',
                 alpha=0.4)
ax[0, 0].fill_between(list(range(len(ang_error_inerf_dyn_pos))), ang_error_inerf_dyn_pos, ang_error_inerf_dyn_neg, color='green',
                 alpha=0.3)

# POSITIONAL ERRORS
ax[0, 1].plot(list(range(len(pos_error_filter))), pos_error_filter, color='red', linewidth=5)
ax[0, 1].plot(list(range(len(pos_error_inerf_dyn))), pos_error_inerf_dyn, color='green', linewidth=5)
#ax[1].plot(list(range(len(pos_errors1))), pos_errors3, label='iNeRF')

#ax[1].plot(list(range(len(pos_error_filter_pos))), pos_error_filter_pos, label='Filter Pos')
#ax[1].plot(list(range(len(pos_error_inerf_dyn_pos))), pos_error_inerf_dyn_pos, label='INeRF W/ Dynamics Pos')

#ax[1].plot(list(range(len(pos_error_filter_neg))), pos_error_filter_neg, label='Filter Neg')
#ax[1].plot(list(range(len(pos_error_inerf_dyn_neg))), pos_error_inerf_dyn_neg, label='INeRF W/ Dynamics Neg')

ax[0, 1].fill_between(list(range(len(pos_error_filter_pos))), pos_error_filter_pos, pos_error_filter_neg, color='red',
                 alpha=0.4)
ax[0, 1].fill_between(list(range(len(pos_error_inerf_dyn_pos))), pos_error_inerf_dyn_pos, pos_error_inerf_dyn_neg, color='green',
                 alpha=0.3)

# VELOCITY ERRORS
ax[1, 1].plot(list(range(len(vel_fil))), vel_fil, color='red', linewidth=5)
ax[1, 1].plot(list(range(len(vel_dyn))), vel_dyn, color='green', linewidth=5)

#ax[2].plot(list(range(len(vel_fil_pos))), vel_fil_pos, label='Filter Pos')
#ax[2].plot(list(range(len(vel_dyn_pos))), vel_dyn_pos, label='INeRF W/ Dynamics Pos')

#ax[2].plot(list(range(len(vel_fil_neg))), vel_fil_neg, label='Filter Neg')
#ax[2].plot(list(range(len(vel_dyn_neg))), vel_dyn_neg, label='INeRF W/ Dynamics Neg')

ax[1, 1].fill_between(list(range(len(vel_fil_pos))), vel_fil_pos, vel_fil_neg, color='red',
                 alpha=0.4)
ax[1, 1].fill_between(list(range(len(vel_dyn_pos))), vel_dyn_pos, vel_dyn_neg, color='green',
                 alpha=0.3)

# OMEGA ERRORS
ax[1, 0].plot(list(range(len(omega_fil))), omega_fil, color='red', linewidth=5)
ax[1, 0].plot(list(range(len(omega_dyn))), omega_dyn, color='green', linewidth=5)

#ax[3].plot(list(range(len(omega_fil_pos))), omega_fil_pos, label='Filter Pos')
#ax[3].plot(list(range(len(omega_dyn_pos))), omega_dyn_pos, label='INeRF W/ Dynamics Pos')

#ax[3].plot(list(range(len(omega_fil_neg))), omega_fil_neg, label='Filter Neg')
#ax[3].plot(list(range(len(omega_dyn_neg))), omega_dyn_pos, label='INeRF W/ Dynamics Neg')

ax[1, 0].fill_between(list(range(len(omega_fil_pos))), omega_fil_pos, omega_fil_neg, color='red',
                 alpha=0.4)
ax[1, 0].fill_between(list(range(len(omega_dyn_pos))), omega_dyn_pos, omega_dyn_neg, color='green',
                 alpha=0.3)

ax[1, 1].set_ylabel('Velocity Error', fontsize=70)
ax[1, 0].set_ylabel(r'$\omega$ Error', fontsize=70)

ax[0, 0].set_ylabel('Rotational Error', fontsize=70)
ax[0, 1].set_ylabel('Translational Error', fontsize=70)

ax[0, 0].set_xticks(np.arange(0, len(ang_error_filter), 300))
for ind, label in enumerate(ax[0, 0].get_xticklabels()):
    if ind % 1 != 0:
        label.set_visible(False)

ax[0, 1].set_xticks(np.arange(0, len(ang_error_filter), 300))
for ind, label in enumerate(ax[0, 1].get_xticklabels()):
    if ind % 1 != 0:
        label.set_visible(False)

ax[1, 0].set_xticks(np.arange(0, len(ang_error_filter), 300))
for ind, label in enumerate(ax[1, 0].get_xticklabels()):
    if ind % 1 != 0:
        label.set_visible(False)

ax[1, 1].set_xticks(np.arange(0, len(ang_error_filter), 300))
for ind, label in enumerate(ax[1, 1].get_xticklabels()):
    if ind % 1 != 0:
        label.set_visible(False)

ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/300))
ax[0, 0].xaxis.set_major_formatter(ticks)
ax[0, 1].xaxis.set_major_formatter(ticks)
ax[1, 0].xaxis.set_major_formatter(ticks)
ax[1, 1].xaxis.set_major_formatter(ticks)

ax[0, 0].tick_params(axis='both', which='major', labelsize=50)
ax[0, 1].tick_params(axis='both', which='major', labelsize=50)
ax[1, 0].tick_params(axis='both', which='major', labelsize=50)
ax[1, 1].tick_params(axis='both', which='major', labelsize=50)

#bbox_to_anchor=(0.5, 1.05)
leg = fig.legend(loc='upper center',
        ncol=5, fancybox=True, shadow=True, fontsize=80)
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(8.0)

#ax[0].set_ylim([0, 0.05])
#ax[0].set_xlim([0, len(ang_error_filter)])
#ax[1].set_ylim([0, 0.1])
#ax[1].set_xlim([0, len(ang_error_filter)])
ax[1, 0].set_xlabel('Time Step', fontsize=70)
ax[1, 1].set_xlabel('Time Step', fontsize=70)

fig.savefig(save_dir + f'errors_stat.pdf', format = 'pdf', dpi=300)

#print('Iteration', i)

'''
#Plot state evolution
states = np.array(states).reshape((-1, 18))
states_rot = convert_to_rot_vector(states)
true_states_rot = convert_to_rot_vector(true_states)

fig, ax = plt.subplots(3, 2, figsize=(16, 16), dpi=80)
fig.suptitle('State Estimates')
ax[0, 0].step(16*np.array(list(range(len(true_states[:, 0])))), true_states[:, 0], label='Ground Truth')
ax[0, 0].plot(list(range(len(states[:, 0]))), states[:, 0], label='Estimate')
ax[0, 0].legend()

ax[1, 0].step(16*np.array(list(range(len(true_states[:, 0])))), true_states[:, 1], label='Ground Truth')
ax[1, 0].plot(list(range(len(states[:, 0]))), states[:, 1], label='Estimate')
ax[1, 0].legend()

ax[2, 0].step(16*np.array(list(range(len(true_states[:, 0])))), true_states[:, 2], label='Ground Truth')
ax[2, 0].plot(list(range(len(states[:, 0]))), states[:, 2], label='Estimate')
ax[2, 0].legend()

ax[0, 1].step(16*np.array(list(range(len(true_states_rot[:, 0])))), true_states_rot[:, 0], label='Ground Truth')
ax[0, 1].plot(list(range(len(states_rot[:, 0]))), states_rot[:, 0], label='Estimate')
ax[0, 1].legend()

ax[1, 1].step(16*np.array(list(range(len(true_states_rot[:, 0])))), true_states_rot[:, 1], label='Ground Truth')
ax[1, 1].plot(list(range(len(states_rot[:, 0]))), states_rot[:, 1], label='Estimate')
ax[1, 1].legend()

ax[2, 1].step(16*np.array(list(range(len(true_states_rot[:, 0])))), true_states_rot[:, 2], label='Ground Truth')
ax[2, 1].plot(list(range(len(states_rot[:, 0]))), states_rot[:, 2], label='Estimate')
ax[2, 1].legend()

ax[0, 0].set_title('X')
ax[1, 0].set_title('Y')
ax[2, 0].set_title('Z')

ax[0, 1].set_title(r'$\theta$')
ax[1, 1].set_title(r'$\phi$')
ax[2, 1].set_title(r'$\psi$')


ax[0, 0].set_xticks(np.arange(0, len(rot_errors), 16))
for ind, label in enumerate(ax[0, 0].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[1, 0].set_xticks(np.arange(0, len(trans_errors), 16))
for ind, label in enumerate(ax[1, 0].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[2, 0].set_xticks(np.arange(0, len(rot_errors), 16))
for ind, label in enumerate(ax[2, 0].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[0, 1].set_xticks(np.arange(0, len(trans_errors), 16))
for ind, label in enumerate(ax[0, 1].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[1, 1].set_xticks(np.arange(0, len(rot_errors), 16))
for ind, label in enumerate(ax[1, 1].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[2, 1].set_xticks(np.arange(0, len(trans_errors), 16))
for ind, label in enumerate(ax[2, 1].get_xticklabels()):
    if ind % 5 != 0:
        label.set_visible(False)

ax[0, 0].grid()
ax[1, 0].grid()
ax[2, 0].grid()

ax[0, 1].grid()
ax[1, 1].grid()
ax[2, 1].grid()

fig.savefig(pathname+'state_estimates.png')
'''


#Calculate statistics
'''
print('Filter trans error', np.mean(np.linalg.norm(np.array(state_estimates1)[:, :3] - true_states[1:, :3], ord=2, axis=1)))
print('INeRF dynamics trans error', np.mean(np.linalg.norm(np.array(state_estimates2)[:, :3] - true_states[1:, :3], ord=2, axis=1)))
print('INeRF trans error', np.mean(np.linalg.norm(np.array(state_estimates3)[:, :3] - true_states[1:, :3], ord=2, axis=1)))

ang_err1 = []
ang_err2 = []
ang_err3 = []
for a, val in enumerate(state_estimates1):
    ang_err1.append(find_angle_diff(true_states[1+a, 6:15].reshape((3, 3)), np.array(state_estimates1)[a, 6:15].reshape((3, 3))))
    ang_err2.append(find_angle_diff(true_states[1+a, 6:15].reshape((3, 3)), np.array(state_estimates2)[a, 6:15].reshape((3, 3))))
    ang_err3.append(find_angle_diff(true_states[1+a, 6:15].reshape((3, 3)), np.array(state_estimates3)[a, 6:15].reshape((3, 3))))

print('Filter rot error', np.mean(ang_err1))
print('INeRF dynamics rot error', np.mean(ang_err2))
print('INeRF rot error', np.mean(ang_err3))
'''