import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl

# ===================== Part 1: Data Extraction =====================

filepath = '/home/GraphMM_153/results/Metamodel_VE_IHC_ISK_600s_nohub/'
os.makedirs(filepath + 'extract_files/', exist_ok=True)

# Build the column index mapping from the CSV header
model_var_prefix = (
    'coupler,coupler,coupler,coupler,I.VE,I.VE,V.VE,V.VE,F.VE,F.VE,R.VE,R.VE,'
    'D.VE,D.VE,D_IR.VE,D_IR.VE,gamma.VE,gamma.VE,rho.VE,rho.VE,ISR.VE,ISR.VE'
)
model_var = [model_var_prefix]
for prefix in ['V', 'n', 's', 'Ca']:
    for j in range(153):
        model_var.append(f'{prefix}{j}.IHC,{prefix}{j}.IHC')
model_var += [
    'mem_V.ISK,mem_V.ISK,Ca_md.ISK,Ca_md.ISK,Ca_ic.ISK,Ca_ic.ISK,'
    'N1.ISK,N1.ISK,N2.ISK,N2.ISK,N3.ISK,N3.ISK,N4.ISK,N4.ISK,'
    'N5.ISK,N5.ISK,N6.ISK,N6.ISK,NF.ISK,NF.ISK,NR.ISK,NR.ISK,'
    'SE.ISK,SE.ISK'
]
model_var = ','.join(model_var)
model_vars = model_var.split(',')

for i in range(153):
    meta_filename = f"Metamodel_VE_IHC_ISK_600s_nohub_{i}.csv"

    # Indices for VE variables
    ve_cell_D = model_vars.index('D.VE')
    ve_cell_R = model_vars.index('R.VE')
    ve_cell_DIR = model_vars.index('D_IR.VE')
    ve_cell_f = model_vars.index('F.VE')
    ve_cell_isr = model_vars.index('ISR.VE')

    # Indices for IHC variables (cell i)
    ihc_cell_v = model_vars.index(f'V{i}.IHC')
    ihc_cell_n = model_vars.index(f'n{i}.IHC')
    ihc_cell_s = model_vars.index(f's{i}.IHC')
    ihc_cell_ca = model_vars.index(f'Ca{i}.IHC')

    # Indices for ISK variables
    isk_cell_v = model_vars.index('mem_V.ISK')
    isk_cell_ca = model_vars.index('Ca_md.ISK')
    isk_cell_f = model_vars.index('NR.ISK')
    isk_cell_ca_ic = model_vars.index('Ca_ic.ISK')

    try:
        meta_result = np.genfromtxt(
            filepath + meta_filename,
            delimiter=',',
            skip_header=1,
            max_rows=100000,
            usecols=(
                0, 1, 2, 3,
                ve_cell_f, ve_cell_f + 1,
                ve_cell_isr, ve_cell_isr + 1,
                ihc_cell_v, ihc_cell_v + 1,
                ihc_cell_n, ihc_cell_n + 1,
                ihc_cell_s, ihc_cell_s + 1,
                ihc_cell_ca, ihc_cell_ca + 1,
                isk_cell_v, isk_cell_v + 1,
                isk_cell_ca, isk_cell_ca + 1,
                isk_cell_f, isk_cell_f + 1,
                ve_cell_R, ve_cell_R + 1,
                ve_cell_D, ve_cell_D + 1,
                ve_cell_DIR, ve_cell_DIR + 1,
                isk_cell_ca_ic, isk_cell_ca_ic + 1
            )
        ).reshape(-1, 30)

        print(f"Cell {i}: loaded shape {meta_result.shape}")
        np.savetxt(filepath + 'extract_files/' + meta_filename, meta_result)

    except Exception as e:
        print(f"Failed to process {meta_filename}: {e}")

# ===================== Part 2: Plotting =====================

# Font and style settings
font = {'family': 'Arial narrow', 'size': 20}
COLOR = '#202020'
mpl.rc('font', **font)
mpl.rc('xtick', labelsize=20)
mpl.rc('ytick', labelsize=20)
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

# Input / output folders
input_folder = '/home/GraphMM_153/results/Metamodel_VE_IHC_ISK_600s_nohub/extract_files/'
output_folder = './plots_metamodel/'
os.makedirs(output_folder, exist_ok=True)

# Time axis (100000 points, from 0 to 10)
time = np.linspace(0, 10, 100000)

# List of variable names to plot (13 total)
variables = [
    "coupler1", "coupler2", "F.VE", "ISR.VE", "V.IHC", "n.IHC", "s.IHC", "Ca.IHC",
    "mem_V.ISK", "Ca_md.ISK", "NR.ISK", "R.VE", "D.VE", "D_IR.VE", "Ca_ic.ISK"
]
num_vars = len(variables)

# Create subplot grid
ncols = 4
nrows = int(np.ceil(num_vars / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
axs = axs.flatten()

# Loop over all 153 cells and overlay their traces
for cell_id in range(153):
    filepath = os.path.join(input_folder, f'Metamodel_VE_IHC_ISK_600s_nohub_{cell_id}.csv')
    if not os.path.exists(filepath):
        print(f"[!] Skipping missing file: {filepath}")
        continue

    try:
        data = np.genfromtxt(filepath, delimiter=' ').reshape(-1, 15, 2)
    except Exception as e:
        print(f"[!] Failed to read {filepath}: {e}")
        continue

    data_mean = data[:, :, 0]  # only mean values (first column of each pair)

    # Plot each variable into its corresponding subplot
    for idx in range(num_vars):
        ax = axs[idx]
        y = data_mean[:, idx]
        ax.plot(time, y, color='C0', alpha=0.3, linewidth=1)

# Customize subplots
for idx in range(num_vars):
    ax = axs[idx]
    ax.set_title(variables[idx], fontsize=18)
    ax.set_xlabel('Time [s]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(25))

# Hide any unused subplots
for j in range(num_vars, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'all_meta_overlay_153cells.png'), dpi=600, bbox_inches='tight')
plt.close()
print("[✓] All plots generated successfully.")