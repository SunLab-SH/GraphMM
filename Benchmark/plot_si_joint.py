# %%

from InputModel.Groundtruth import *
from InputModel.Subsystem import *
from GraphMetamodel.SurrogateModel_new import *
from GraphMetamodel.DefineCouplingGraph_new import *
from InputModel.Subsystem import *
from InputModel.Groundtruth import *
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker 
from Surrogate_model_a import *
from Surrogate_model_b import *
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'Arial Narrow'

# Load data

surrogate_a = run_surrogate_model_a(method='MultiScale', mean_scale=1, 
                                    transition_cov_scale=0.01, emission_cov_scale=1)    

surrogate_b = run_surrogate_model_b(method='MultiScale', mean_scale=1, 
                                    transition_cov_scale=0.01, emission_cov_scale=1)
                                    

surrogate_a_mean = np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',', skip_header=1).reshape(-1,2,2)[:, :, 0]
surrogate_a_std =  np.genfromtxt('./results/surrogate_model_a_new.csv', delimiter=',', skip_header=1).reshape(-1,2,2)[:, :, 1]


# %%

gt_data = np.array([groundtruth[:, 2], groundtruth[:, 0]]).transpose()


# Create a separate figure for the legend
fig_legend = plt.figure(figsize=(10, 1))
ax_legend = fig_legend.add_subplot(111)

# Create dummy plots for legend entries
dummy_lines = [
    plt.Line2D([0], [0], color='C1', linestyle='--', linewidth=2, label='Input model'),
    plt.Line2D([0], [0], color='C0', linewidth=2, label='Surrogate model'),
    plt.Line2D([0], [0], color='red', linewidth=2, label='Updated surrogate model'),
    plt.Line2D([0], [0], color='black', linewidth=2, label='Groundtruth')
]

# Add the legend to the separate figure
legend = ax_legend.legend(handles=dummy_lines, loc='center', ncol=1, prop={'size': 16, 'family': 'Arial Narrow'}, frameon=False)

# Remove axis from the legend figure
ax_legend.axis('off')

# Save the legend as a separate image
plt.tight_layout()
plt.savefig('./legend.png', dpi=600, bbox_inches='tight')
plt.close(fig_legend)


# %%

gt_data = np.array([groundtruth[:, 1], groundtruth[:, 2], groundtruth[:, 3]]).transpose()



# from Surrogate_model_a import *
from scipy.stats import multivariate_normal

surrogate_a_data = np.genfromtxt('./results/surrogate_model_a_joint.csv', delimiter=',', skip_header=1)
surrogate_a_mean = surrogate_a_data[::3]
surrogate_a_cov = np.zeros((160, 2, 2))
for i in range(2):  
    surrogate_a_cov[:, i, :] = surrogate_a_data[i+1::3]


fig = plt.figure(figsize=(4.5, 4))
ax = fig.add_subplot(1, 1, 1)
ts = 80

# Generate grid of points
x, y = np.meshgrid(np.linspace(10, 23, 100), np.linspace(15, 40, 100))
pos = np.dstack((x, y))

# Input model distribution
rv_input = multivariate_normal([input_a[ts, 0], input_a[ts, 1]], 
                               [[input_a_std[ts, 0]**2, 0], 
                                [0, input_a_std[ts, 1]**2]])
z_input = rv_input.pdf(pos)

# Surrogate model distribution
rv_surrogate = multivariate_normal([surrogate_a_mean[ts, 0], surrogate_a_mean[ts, 1]], 
                                   surrogate_a_cov[ts, :, :])

z_surrogate = rv_surrogate.pdf(pos)

# Plot contours
plt.contour(x, y, z_input, colors='#5f89b1', linestyles='dashed')
plt.contour(x, y, z_surrogate, colors='#D93F49')
plt.plot([], [], color='#5f89b1', linestyle='--', label='Input model')
plt.plot([], [], color='#D93F49', label='Surrogate model')
# Set up axes
ax.tick_params(labelsize=20, direction='in')
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.f'))

ax.yaxis.set_major_locator(MultipleLocator(6))
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.d'))

ax.tick_params(axis='x', which='both', length=0, pad=10)
ax.tick_params(axis='y', which='both', length=0, pad=10)

# Labels and title
plt.title('Body model joint distribution', fontsize=20, fontname='Arial Narrow', pad=15)
plt.xlabel(r'$S^{B}\ $[pg/islet/min]', fontsize=20, fontname='Arial Narrow', labelpad=5)
plt.ylabel(r'$D^{B}$', fontsize=20, fontname='Arial Narrow', labelpad=5)
plt.legend(loc='best', prop={'size': 16, 'family': 'Arial Narrow'}, frameon=False)

# Save and show
plt.savefig('./surrogate_input_model_a_joint.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
