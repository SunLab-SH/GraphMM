'''
    Author: Chenxi Wang (chenxi.wang@salilab.org)
    Date: 2022-04-20
'''

import numpy as np
from GraphMetamodel.utils import *
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import random
from GraphMetamodel.SequentialImportanceSampling_v3 import effective_n, resample


class MetaModel:

    def __init__(self, coupling_graph):

        self.coupling_graph = coupling_graph
        self.n_model = len(self.coupling_graph.model_idx)
        self.meta_state = []
        for key in self.coupling_graph.model_idx:
            self.meta_state += self.coupling_graph.model_idx[key].state
        self.meta_n_state = len(self.meta_state) + 1
        print(self.meta_n_state)

        self.total_time_list = [self.coupling_graph.model_idx[key].total_time for key in self.coupling_graph.model_idx]
        self.dt_list = [self.coupling_graph.model_idx[key].dt for key in self.coupling_graph.model_idx]
        self.min_dt = min(self.dt_list)
        self.max_total_time = max(self.total_time_list)
        self.max_n_step = len(np.arange(0, self.max_total_time, self.min_dt, dtype=float))
        self.upd_coupling_var = []

        # ======= parameters in the coupling graph =========
        for num, key in enumerate(self.coupling_graph.models):
            
            model_a = self.coupling_graph.models[key][0]
            model_b = self.coupling_graph.models[key][1]
                
            model_a.con_var_idx += [self.coupling_graph.connect_idx[key][0]]
            model_a.con_omega += [[item[0] for item in self.coupling_graph.omega]]
            model_a.con_unit_weight += [self.coupling_graph.unit_weight[num][0]]
            
            model_b.con_var_idx += [self.coupling_graph.connect_idx[key][1]]
            model_b.con_omega += [[item[1] for item in self.coupling_graph.omega]]
            model_b.con_unit_weight += [self.coupling_graph.unit_weight[num][1]]

        self.noise_model_type = 'time-variant'

    
    def _get_initial_meta_state(self):

        coupling_var_state_mean_t0 = [c[0,0] for c in self.coupling_graph.coupling_variable] 
        coupling_var_state_std_t0 = [c[0,1] for c in self.coupling_graph.coupling_variable] 

        model_var_state_mean_t0, model_var_state_std_t0 = [], []
        for key in self.coupling_graph.model_idx:
            s_model = self.coupling_graph.model_idx[key]
            if isinstance(s_model.initial, list):
                model_var_state_mean_t0 += s_model.initial
            else:
                model_var_state_mean_t0 += s_model.initial.tolist()
            model_var_state_std_t0 += np.diag(s_model.initial_noise).tolist()
           
        Meta_mean_t0 = np.array(coupling_var_state_mean_t0 + model_var_state_mean_t0)
        Meta_cov_t0 = np.diag(np.array(coupling_var_state_std_t0 + model_var_state_std_t0))**2

        return Meta_mean_t0, Meta_cov_t0


    def _fx_metamodel(self, x_ts, dt, ts, omega_ts):

        xout = x_ts.copy()
        units = [1., 1.]

        xout[0] = self.coupling_graph.coupling_variable[0][ts, 0]
        xout[1:3] = self.coupling_graph.models['a_b'][0].fx(x_ts[1:3], self.min_dt)
        xout[2] = ((1-omega_ts[0])*xout[2]*units[0] + omega_ts[0]*xout[0]) / units[0]
        xout[3:] = self.coupling_graph.models['a_b'][1].fx(x_ts[3:], self.min_dt)
        xout[-2] = ((1-omega_ts[1])*xout[-2]*units[1] + omega_ts[1]*xout[0]) / units[1]

        return xout


    def particle_sampling(self, particles, particle_weights, ts, omega_ts):
        
        n_particles = len(particles)
        new_particles = np.zeros_like(particles)
        
        for i in range(n_particles):
            new_particles[i] = self._fx_metamodel(particles[i], self.min_dt, ts, omega_ts)
        
        # Update weights based on likelihood
        likelihoods = np.ones(n_particles)
        particle_weights *= likelihoods
        particle_weights /= np.sum(particle_weights)  # Normalize weights
        
        # Resample if effective N is too low
        if effective_n(particle_weights) < n_particles / 2:
            indices = resample(particle_weights)
            new_particles = new_particles[indices]
            particle_weights = np.ones(n_particles) / n_particles
        
        return new_particles, particle_weights


    def inference(self, test_omega, n_particles=1000, filepath=None, verbose=1):

        if verbose==1:
            print('******** Metamodel info ********')
            for i,key in enumerate(self.coupling_graph.model_idx):
                model = self.coupling_graph.model_idx[key]
                print('==========================')
                print('model_{}_name: {}'.format(i+1, model.modelname))
                print('total_time: {} {}'.format(model.total_time, model.unit))
                print('time_step: {} {}'.format(model.dt, model.unit))
                print('==========================')
            print('******** Metamodel info ********')

        if filepath is not None:
            output = open(filepath, 'w')
            print(*['coupler','coupler']*self.coupling_graph.n_coupling_var+list(np.repeat(self.meta_state, 2)), file=output, sep=',')
        
        if verbose==1:
            print('-------- Run metamodel ---------') 

        x_ts, P_ts = self._get_initial_meta_state()
        particles = np.random.multivariate_normal(x_ts, P_ts, n_particles)
        particle_weights = np.ones(n_particles) / n_particles

        for ts in range(self.max_n_step):
                
            omega_ts = test_omega
            particles, particle_weights = self.particle_sampling(particles, particle_weights, ts, omega_ts)
            
            Meta_mean_ts = np.average(particles, weights=particle_weights, axis=0)
            Meta_std_ts = np.sqrt(np.average((particles - Meta_mean_ts)**2, weights=particle_weights, axis=0))

            if filepath is not None:
                print(*list(chain.from_iterable(zip(Meta_mean_ts, Meta_std_ts))), file=output, sep=',')
                
            if verbose==1:
                time.sleep(1e-20)
                process_bar(ts+1, self.max_n_step)
        
        if filepath is not None:
            output.close()

        print('\n-------- Finished ----------')
