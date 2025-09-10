# %%

import numpy as np
import matplotlib.pyplot as plt
from GraphMetamodel.MultiScaleInference_PF import MetaModel as PFMetaModel
from GraphMetamodel.MultiScaleInference_new import MetaModel as UKFMetaModel
from GraphMetamodel.DefineCouplingGraph_new import coupling_graph
from Surrogate_model_a import run_surrogate_model_a
from Surrogate_model_b import run_surrogate_model_b


# Run surrogate model
surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1,
                                        transition_cov_scale=0.01, emission_cov_scale=10)

surrogate_b = run_surrogate_model_b(method='MultiScale', mean_scale=1,
                                    transition_cov_scale=0.01, emission_cov_scale=1)

# Load surrogate model states
surrogate_a_states = np.genfromtxt('./results/surrogate_model_a.csv', delimiter=',', skip_header=1).reshape(-1,2,2)
surrogate_b_states = np.genfromtxt('./results/surrogate_model_b.csv', delimiter=',', skip_header=1).reshape(-1,3,2)

# Define a single coupling graph for both PF and UKF
Coupling_graph = coupling_graph(models = {'a_b':(surrogate_a, surrogate_b)}, 
                                connect_var = {'a_b':('$I_{islet}^a\ [pg/islet]$', 
                                                      '$I_{cell}^b [pg/islet]$')}, 
                                unit_weights = [[1,1]],  
                                model_states = {'a':surrogate_a_states, 
                                                'b':surrogate_b_states},
                                timescale = {'a_b':[1, 5]},
                                w_phi=1, w_omega=1, w_epsilon=1)

Coupling_graph.get_coupling_graph_multi_scale(p=0.5)

# Initialize PF and UKF metamodels with the same coupling graph
pf_metamodel = PFMetaModel(Coupling_graph)
ukf_metamodel = UKFMetaModel(Coupling_graph)

# Set test parameters
test_omega = [0.8, 0.2]  # Example test parameters

# Run inference
n_particles = 10000  # Number of particles for PF
# pf_metamodel.inference(n_particles=n_particles, test_omega=test_omega, filepath='./results/pf_results.csv')
ukf_metamodel.inference(test_omega=test_omega, filepath='./results/ukf_results_joint.csv')



# %%

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from InputModel.Groundtruth import groundtruth
import numpy as np

def plot_comparison(pf_results, ukf_results, surrogate_a, gt_data):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    variables = ['D^B', 'S^B \ [pg/islet/min]', '\gamma^C', 'S^C \ [pg/cell/min]', 'G^C [mM]']
    indices = [2, 4, 6, 8, 10]

    for i, (ax, var) in enumerate(zip(axs.ravel(), variables)):
        total = 7  # min

        if i < 2:
            ax = axs[0, i]
        else:
            ax = axs[1, i - 2]

        setup_axis(ax, var, surrogate_a.unit)
        
        # Calculate cutoff points based on data length
        ukf_cutoff = int(len(ukf_results) * 7/8)  # For 7 out of 8 minutes
        pf_cutoff = int(len(pf_results) * 7/8)
        gt_cutoff = int(len(gt_data) * 7/8)
        
        plot_results(ax, total, ukf_results[:ukf_cutoff], indices[i], 'C0', 'GraphMM_UKF', '-')
        plot_results(ax, total, pf_results[:pf_cutoff], indices[i], 'C1', 'GraphMM_PF', '-')
        plot_ground_truth(ax, total, gt_data[:gt_cutoff], i)

    axs[0, -1].set_visible(False)
    plt.legend(prop={'size': 20, 'family': 'Arial Narrow'}, frameon=False)
    plt.tight_layout()
    plt.savefig('./pf_ukf_comparison_all_variables.png', dpi=600, bbox_inches='tight')
    plt.show()

def setup_axis(ax, var, unit):
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.f'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(25)
        label.set_fontfamily('Arial Narrow')

    ax.set_xlabel(f'Time [{surrogate_a.unit}]', fontsize=25, fontname='Arial Narrow')
    ax.set_ylabel(f'${var}$', fontsize=25, fontname='Arial Narrow')
    ax.legend(loc='best', prop={'size': 25, 'family': 'Arial Narrow'}, frameon=False)

def plot_results(ax, sim_time, results, index, color, label, linestyle):
    sim_time = np.linspace(0, sim_time, len(results))  
    ax.plot(sim_time, results[:, index], color=color, label=label, linewidth=2, linestyle=linestyle)
    ax.fill_between(sim_time, 
                    results[:, index] - results[:, index+1], 
                    results[:, index] + results[:, index+1],
                    color=color, alpha=0.1)

def plot_ground_truth(ax, sim_time, gt_data, index):
    sim_time = np.linspace(0, sim_time, len(gt_data))  
    ax.plot(sim_time, gt_data[:, index], color='gray', label='Groundtruth', linewidth=2, linestyle='--')

def add_vertical_line(ax, y_value):
    ymin = (y_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.axvline(x=1, ymax=ymin, color='gray', linestyle='--')

# Load results
pf_results = np.genfromtxt('./results/pf_results.csv', delimiter=',', skip_header=1)
ukf_results = np.genfromtxt('./results/ukf_results.csv', delimiter=',', skip_header=1)
gt_data = np.array([groundtruth[:, 0], groundtruth[:, 2], groundtruth[:, 1], groundtruth[:, 2], groundtruth[:, 3]]).transpose()

# Plot comparison
plot_comparison(pf_results, ukf_results, surrogate_a, gt_data)

# # Compare performance for all variables
# for i, var in enumerate(variables):
#     pf_error = np.mean((pf_results[:, pf_indices[i]] - ukf_results[:, ukf_indices[i]])**2)
#     print(f"Mean Squared Error between PF and UKF for {var}: {pf_error:.6f}")






# %%

# for n_particles in [10, 100, 1000, 10000]:
#     pf_metamodel.inference(n_particles=n_particles, test_omega=test_omega, filepath=f'./results/pf_results_{n_particles}.csv')



# %%


# # plot and compare PF with different numbers of particles

# # Plot and compare PF with different numbers of particles
# fig, axs = plt.subplots(3, 2, figsize=(15, 18))
# fig.suptitle('Comparison of PF with different particle numbers', fontsize=16)

# variables = ['Coupling Variable', 'I_islet^a', 'D^a', 'I_cell^b', 'G^b', 'S^b']
# colors = plt.cm.viridis(np.linspace(0, 1, 4))  # Color map for different particle numbers

# for i, (ax, var) in enumerate(zip(axs.flatten(), variables)):
#     for j, n_particles in enumerate([10, 100, 1000, 10000]):
#         pf_results = np.genfromtxt(f'./results/pf_results_{n_particles}.csv', delimiter=',', skip_header=1)
#         pf_sim_time = np.linspace(0, surrogate_a.total_time, len(pf_results))
        
#         ax.plot(pf_sim_time, pf_results[:, i*2], color=colors[j], label=f'PF ({n_particles} particles)', linewidth=1.8)
#         ax.fill_between(pf_sim_time, 
#                         pf_results[:, i*2] - pf_results[:, i*2+1], 
#                         pf_results[:, i*2] + pf_results[:, i*2+1],
#                         color=colors[j], alpha=0.1)
    
#     ax.set_title(var)
#     ax.set_xlabel('Time [{}]'.format(surrogate_a.unit))
#     ax.set_ylabel('Value')
#     ax.legend(loc='best', prop={'size': 8, 'family': 'Arial Narrow'}, frameon=False)
#     ax.tick_params(labelsize=10)
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')

# plt.tight_layout()
# plt.savefig('./pf_comparison_particle_numbers.png', dpi=600, bbox_inches='tight')
# plt.show()

# # Compare performance for all variables
# ukf_results = np.genfromtxt('./results/ukf_results.csv', delimiter=',', skip_header=1)
# for n_particles in [10, 100, 1000, 10000]:
#     pf_results = np.genfromtxt(f'./results/pf_results_{n_particles}.csv', delimiter=',', skip_header=1)
#     for i, var in enumerate(variables):
#         pf_error = np.mean((pf_results[:, i*2] - ukf_results[:, i*2])**2)
#         print(f"Mean Squared Error between PF ({n_particles} particles) and UKF for {var}: {pf_error:.6f}")
#     print()


# %%

