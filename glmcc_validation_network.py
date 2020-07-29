# -*- coding: utf-8 -*-
#
# glmcc_validation_network.py
#
# Author: Anno Kurth
# Contact: a.kurth@fz-juelich.de
#
# Before running create directory 'data' in WD


import numpy as np
import os
import nest
import csv

abspath = os.path.abspath(__file__)
path, tail = os.path.split(abspath)
data_dir = os.path.join(path, 'data')

nest.ResetKernel()

'''
Simulation setup
'''
N_mpi = 1  # number of mpi processes
N_cpus = 8  # number of CPUs per mpi process

N_vps = N_mpi*N_cpus

'''
Experimental Setup
Network parameters changing for different experimental setups corresponding to
avergage incoming excitatory connections of 10, 20, 50, 100, 200
'''
setup = [
    {'avg_inp_ex' : 10,  # average number of incoming excitatory connections
     'scal_poi_ex' : 1.1,  # scaling rate Poisson-input for excitatory neurons
     'scal_poi_in' : 1.2,  # sclaing rate Poisson-input for inhibitory neurons
     'I_e_ex' : -0.32,  # external current injected to excitatory neurons
     'I_e_in' : -0.32},  # etxernal current injected to inhibitory neurons
    {'avg_inp_ex' : 20,
     'scal_poi_ex' : 1.1,
     'scal_poi_in' : 1.2,
     'I_e_ex' : -0.28,
     'I_e_in' : -0.28},
    {'avg_inp_ex' : 50,
     'scal_poi_ex' : 1.1,
     'scal_poi_in' : 1.2,
     'I_e_ex' : -0.19,
     'I_e_in' : -0.19},
    {'avg_inp_ex' : 100,
     'scal_poi_ex' : 1.1,
     'scal_poi_in' : 1.2,
     'I_e_ex' : -0.06,
     'I_e_in' : -0.06},
    {'avg_inp_ex' : 200,
     'scal_poi_ex' : 1.1,
     'scal_poi_in' : 1.3,
     'I_e_ex' : 0.31,
     'I_e_in' : 0.3}
    ]


setup_number = 3  # number defining given experimental setup

'''
Assigning simulation parameters
'''
dt = 0.1  # time resolution in ms
simtime = 10. # 1000.*60*90  # simtime in ms
delay = 2.


'''
Definition of network parameters
'''
model = 'iaf_psc_exp'  # neuron model
N_neurons = int(1e4)  # total number of neurons
NI = N_neurons//5  # number of inhibitory neurons
NE = N_neurons - NI  # number of excitatory neurons
N_rec = 100  # total number neurons to be recorded from
NI_rec = N_rec//5  # number of inhibitory neurons recorded from
NE_rec = N_rec - NI_rec  # number of excitatory neurons recorded from
g = 7.5  # ratio of inhibitory weight/excitatory weight
avg_inp_ex = setup[setup_number]['avg_inp_ex']  # average input from excitatory neurons
scal_poi_ex = setup[setup_number]['scal_poi_ex']  # scaling rate Poisson-input
scal_poi_in = setup[setup_number]['scal_poi_in']  # scaling rate Poissin-input
I_e_ex = setup[setup_number]['I_e_ex']  # external current to exc-neurons
I_e_in = setup[setup_number]['I_e_in']  # external current to inh-neurons


'''
Definition of single neuron parameters
See Zaytsev, Morrison, Deger, Reconstruction of recurrent synaptic connectivity
of thousands of neurons from simulated spiking activity,
J Comput Neurosci (2015) 39:77-103, DOI 10.1007/s10827-015-0565-5
'''
neuron_params = {'C_m': 0.45,  # membrane capacitance in pF
                 'tau_m': 20.0,  # membrane time constant in ms
                 'tau_syn_ex': 0.5,  # synpatic time constant for EPSC in ms
                 'tau_syn_in': 0.5,  # synpatic time constant for IPSC in ms
                 't_ref': 2.0,  # refractory period in ms
                 'E_L': 0.0,  # resting potential in mV
                 'V_reset': 0.0,  # spike after-potential in mV
                 'V_m': 0.0,  # initial membrane potential in mV
                 'V_th': 20.0}  # threshold in mV


'''
Definition of connectivity parameters
'''
CE = int(np.round(avg_inp_ex*N_neurons/NE))  # indegree of excitatory connections
CI = 2*CE  # indegree of inhibitory connections

J_ex = 2.
J_in = -g*J_ex

T_start = 100.  # start of recording in ms

'''
Definition of external rate
'''
nu_th = neuron_params['V_th']/(J_ex*CE*neuron_params['tau_m'])
nu_th = 1000.*nu_th*CE

nu_ex = scal_poi_ex*nu_th  # rate of Poisson noise to excitatory neurons
nu_in = scal_poi_in*nu_th  # rate of Poisson noise to inhibitory neurons


'''
Configuration of simulation kernel
'''
nest.SetKernelStatus({'print_time': False,
                      'resolution': dt,
                      'total_num_virtual_procs': N_vps,
                      'overwrite_files': True})

msd = 123456
nest.SetKernelStatus({'rng_seeds': range(msd+N_vps+1, msd+2*N_vps+1)})

'''
Configuration of chosen model for simulation
'''
nest.SetDefaults(model, neuron_params)
nest.CopyModel('poisson_generator', 'noise_ex', {'rate': nu_ex})
nest.CopyModel('poisson_generator', 'noise_in', {'rate': nu_in})

'''
Creation of nodes
'''
nodes_ex = nest.Create(model, NE)
nodes_in = nest.Create(model, NI)
nodes_rec = nodes_ex[:NE_rec] + nodes_in[:NI_rec]
noise_ex = nest.Create('noise_ex')
noise_in = nest.Create('noise_in')
espikes = nest.Create('spike_detector')
ispikes = nest.Create('spike_detector')
spikes = nest.Create('spike_detector')


'''
Configuration of neurons
'''
nest.SetStatus(nodes_ex, [{'I_e': I_e_ex}])
nest.SetStatus(nodes_in, [{'I_e': I_e_in}])


'''
Configuration of spike detectors
'''
nest.SetStatus(espikes, [{'label': os.path.join(data_dir, 'spikes_ex'),
                          'withtime': True,
                          'withgid': True,
                          'to_file': True}])
nest.SetStatus(ispikes, [{'label': os.path.join(data_dir, 'spikes_in'),
                          'withtime': True,
                          'withgid': True,
                          'to_file': True}])


'''
Connecting
'''
nest.CopyModel('static_synapse', 'excitatory', {'weight': J_ex, 'delay': delay})


syn_params_ex = {'model': 'excitatory'}

conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}


nest.Connect(noise_ex, nodes_ex, syn_spec=syn_params_ex)
nest.Connect(noise_in, nodes_in, syn_spec=syn_params_ex)


nest.Connect(nodes_ex, nodes_ex+nodes_in, conn_params_ex,
             {'model': 'static_synapse', 'weight': J_ex, 'delay': delay})
nest.Connect(nodes_in, nodes_ex+nodes_in, conn_params_in,
             {'model': 'static_synapse', 'weight': J_in, 'delay': delay})


nest.Connect(nodes_ex[0:NE_rec], espikes)
nest.Connect(nodes_in[0:NI_rec], ispikes)

'''
Simulation of the network
'''
nest.Simulate(simtime)

conns = nest.GetStatus(nest.GetConnections(nodes_rec, nodes_rec),
                       keys=['target', 'source', 'weight'])
with open(os.path.join(data_dir, 'connections.csv'), 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['target', 'source', 'weight'])
    for row in conns:
        csv_out.writerow(row)
