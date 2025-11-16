# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:18:30 2025

This code runs a Monte-Carlo simulation in order to explore the probability 
distribution in the Meier-Saupe model. The model is an extension of the Ising 
model to nematic systems. The nemodes have head-tail symmetry, such that a
180° degree rotation leads to states that are identical and indistinguishable.

The Meier-Saupe model uses the angle subtended between nemodes to determine the 
strength of their interaction with each other. The potential is different in 
2D and 3D. The model typically uses a nearest-neighbour interaction assumption,
with periodic boundary conditions such that the nth particle of an n-by-n 
lattice interacts with the 1st particle.

In order to model the nemodes, we define a randomised array of values for the
initial angles phi_i. The interaction potential depends on the angle theta, 
given by the difference in the phi values for the adjacent nemodes. A given
set of values is called a configuration. Each step calculates the random 
adjustment to a given phi, and the energy difference to the previous 
mini-configuration. Following the Monte Carlo criteron this pass is accepted
or rejected with a certain probability dependening on the energy difference.

After n^2 steps, a pass has been completed, and the entire configuration's
energy is recorded and logged. This randomised optimisation continues until the
standard deviation of the last m configurations reaches below a given value - 
that is to say, an equilibrium is reached.


@author: Aarnav Panda 
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
plt.close('all')  # clean start
from scipy.special import eval_legendre
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import time
import imageio
import os, shutil, stat
import cProfile, pstats


# Note:
# 2D_potential = -eps * cos(2*theta)
# 3D_potential = -eps * eval_legendre(2, np.cos(theta))

# GLOBALS
n = 30 # size of array
d = 2 # dimension of system
eps = 1 # to start off with
T = 0.5 # Kelvin
beta = 1/(T)
interval_amplitude = 3   # should be lower than 6.28 really
max_runtime = 36000
snapshot_interval = 100  # 50 usually makes sense
small_snapshot_interval = 10
BINS = 72
SAVE_FIGS = False
SAVE_HIST = False
CREATE_GIF = False
CREATE_GIF_HIST = False
FILE_STRING = f"rev_simulation_{T:.2f}K_{n}by{n}_amp{interval_amplitude}"
ramp_folder = f"ramp_data_{n}by{n}_rev"
base_dir = os.path.join(os.getcwd(), ramp_folder)
eq_step_estimate = n**2
# Step 1: find the energy for the whole configuration.        
# OLD: Nested C code in np.roll
# ============================================================================
# def total_energy_old(phi_array):
#     shifts = [(0,1),(0,-1),(1,0),(-1,0)]
#     total_energy = 0.0
#     for dx, dy in shifts:
#         nn_thetas = np.roll(phi_array, (dx,dy), axis=(0,1)) - phi_array
#         total_energy += -eps * np.cos(2*nn_thetas).sum()
#     return 0.5 * total_energy  # each bond counted twice
# =============================================================================
@njit # NEW: Accelerated loops with numba
def total_energy(phi_array):
    tot = 0.0
    for i in range(n):
        for j in range(n):
            phi_array_0 = phi_array[i, j]
            ip = (i + 1) % n #p for plus (1)
            im = (i - 1) % n #m for minus (1)
            jp = (j + 1) % n
            jm = (j - 1) % n

            tot += -eps * (
                np.cos(2*(phi_array[ip, j] - phi_array_0)) +
                np.cos(2*(phi_array[im, j] - phi_array_0)) +
                np.cos(2*(phi_array[i, jp] - phi_array_0)) +
                np.cos(2*(phi_array[i, jm] - phi_array_0))
            )
    return 0.5 * tot


#TODO: Make it so that running the code creates a folder and 
# saves the end graphs and gif into that folder
# folder should be dynamically named: simulation_{T}K_{n}by{n}_amp{interval_amplitude}
 
# Take a snapshot of histograms to make a gif for them too
# Parámetro de orden 

#Step 2: Make the iterative change with the criterion and compare
@njit
def mc_pass(phi_array, beta):
    """
    Metropolis sweep over all n×n sites in the array.
    Propose new, adjusted phi:
    - accept change if dE < 0 (i.e. more stable)
    - /accept change with proability exp(-βΔE) if dE > 0.
    """
    
    for i in range(n):
        for j in range(n): #this does 1 step per value of j for given i
            old_phi = phi_array[i,j]
            eta = np.random.uniform(-1,1) 
            new_phi = old_phi + eta*interval_amplitude

            # fetch the 4 periodic neighbours by direct indexing
            ip = (i+1) % n
            im = (i-1) % n
            jp = (j+1) % n
            jm = (j-1) % n
            
        
            # local energy before and after
            E_old = -eps * (np.cos(2*(phi_array[ip,j] - old_phi)) +
                            np.cos(2*(phi_array[im,j] - old_phi)) +
                            np.cos(2*(phi_array[i,jp] - old_phi)) +
                            np.cos(2*(phi_array[i,jm] - old_phi)))
            # Yes it's ugly but slightly faster.
            E_new = -eps * (np.cos(2*(phi_array[ip,j] - new_phi)) +
                            np.cos(2*(phi_array[im,j] - new_phi)) +
                            np.cos(2*(phi_array[i,jp] - new_phi)) +
                            np.cos(2*(phi_array[i,jm] - new_phi)))
            dE = E_new - E_old

            # Metropolis criterion
            if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
                phi_array[i,j] = new_phi
    return np.mod(phi_array, 2*np.pi)

def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree.
    If the removal fails because the file is read-only,
    change its permission and try again.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)
    
def plot_hist(flat_angles,angle_plotting_vals,S,step):
    
    # Q = angle_plotting_vals[-1]
    
    # 1) Histogram + density
    hist, edges = np.histogram(
        flat_angles,
        bins=BINS,
        range=(0, 2*np.pi),
        density=True
    )
    centres = 0.5 * (edges[:-1] + edges[1:]) # extracts avg phi for each bin.
    delta = 2*np.pi / BINS # interval between bins

    # 2) Find the peak values for phi
    sigma=1
    hist_smooth = gaussian_filter1d(hist, sigma=sigma) # sigma = 2 always sensible for 2pi range
    peaks, props = find_peaks(
    hist_smooth,
    height=0.5,       # e.g. only peaks >0.1
    distance=10,      # at least 5 bins apart (with a sensible bin number, this always works)
    prominence=0.05   # remove tiny bumps
)
    peak_angles  = centres[peaks]
    peak_values  = hist_smooth[peaks]
    #This is for display purposes
    
    # 3) Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.rc('font', size=14)  
    ax.bar(centres, hist, width=delta, color='dodgerblue', edgecolor='k')
    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Probability density")
    title = f"Angle configuration, T* = {T:.2f}, S={S:3.2f}"
    ax.plot(centres, hist_smooth, '-k', lw=2, label=f"smoothed (σ={sigma})", alpha=0.6)
    ax.plot(peak_angles, peak_values, 'ro', ms=8, label="peaks")    
    ax.set_title(title)
    ax.set_xlim(0,2*np.pi)
    ax.set_ylim(0,1)
    ax.grid(True)
    ax.text(0,-0.13,f"Step = {step}")
    Wistia = plt.get_cmap("RdBu", len(angle_plotting_vals))
    colour_dict = {i: Wistia(i) for i in range(len(angle_plotting_vals))}
    for i in range(len(angle_plotting_vals)):
        ax.arrow(3, 0.8, 
                 1.5*np.cos(np.angle(angle_plotting_vals[-i])/2), 
                 0.2*np.sin(np.angle(angle_plotting_vals[-i])/2),
             head_width=0.04,
             head_length=0.05,
             fc=colour_dict[i], ec=colour_dict[i],
             alpha=1-(i*0.1), # alpha_q later
             length_includes_head=True) # this arrow shows the director
    
    if SAVE_HIST == True:
        plt.savefig(f"frames_hist/angle_histogram_{step:03d}.png",
                    bbox_inches='tight', dpi=80,
                    pil_kwargs={"compress_level": 1})
    #plt.show()
    print(f"{step} steps completed (histogram) ")
    plt.close(fig)
    return 0
    
def calculate_order_param(phi_array):
    flat_angles = phi_array.flatten()
    Q = np.mean(np.exp(2j*flat_angles))
    S = np.abs(Q) # same as taking the sum of cos(2phi)*f(phi)
    return Q,S

def main(phi_init=None):
    print("Current run: " + FILE_STRING)
    start_time = time.time()
    if os.path.isdir("frames"):
        shutil.rmtree("frames",onerror=remove_readonly)
        print("frames folder deleted!")
    if os.path.isdir("frames_hist"):
        shutil.rmtree("frames_hist",onerror=remove_readonly)
        print("frames_hist folder deleted!")
    os.makedirs("frames", exist_ok=True) # Create directory for snapshots of each config
    os.makedirs("frames_hist", exist_ok=True) # Create directory for snapshots of each config
    os.makedirs(FILE_STRING, exist_ok=True) # Create directory for snapshots of each config

    step = 0
    if phi_init is None:
        phi_array = np.random.uniform(-np.pi, np.pi, (n,n))
    else:
        phi_array = phi_init.copy()
    energies = []
    order_param =[]
    Q,S = calculate_order_param(phi_array)
    order_param.append(S)
    angle_plotting_vals = []
    fig, ax = plt.subplots(figsize=(6,6))  # adjust size as needed
    im = ax.imshow(phi_array, cmap='RdBu', vmin=0, vmax=np.pi)
    cbar = fig.colorbar(im)
    cbar.set_label("Angle(rad)")
    # optimisations
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_autoscale_on(False)
    
    while True:
        phi_array = mc_pass(phi_array, beta)
        
        config_energy = total_energy(phi_array)
        energies.append(config_energy)
        Q,S = calculate_order_param(phi_array)
        # same as taking the sum of cos(2phi)*f(phi)
        order_param.append(S)
            

        # Snapshot
        if step < 100 and step % small_snapshot_interval == 0 and SAVE_FIGS == True:
            im.set_data(np.mod(phi_array, np.pi))
            fig.canvas.draw()
            plt.savefig(f"frames/snapshot_{step:03d}.png", bbox_inches='tight', dpi=80,
                    pil_kwargs={"compress_level": 1})
            print(f"{step} figure")
        elif step >= 100 and step % snapshot_interval == 0 and SAVE_FIGS == True:
            im.set_data(np.mod(phi_array, np.pi))
            fig.canvas.draw()
            plt.savefig(f"frames/snapshot_{step:03d}.png", bbox_inches='tight', dpi=80,
                    pil_kwargs={"compress_level": 1})
            print(f"{step} steps completed (figure)")
        # Histogram snapshots            
        
        if step < 100 and step % small_snapshot_interval == 0 and SAVE_HIST == True:
            flat_angles = phi_array.flatten()
            angle_plotting_vals.append(Q)
            if len(angle_plotting_vals) > 5:
                angle_plotting_vals.pop(0) # removes oldest element of rolling window
            plot_hist(flat_angles, angle_plotting_vals, S, step)
        elif step >= 100 and step % snapshot_interval == 0 and SAVE_HIST == True:
            flat_angles = phi_array.flatten()
            angle_plotting_vals.append(Q)
            if len(angle_plotting_vals) > 5:
                angle_plotting_vals.pop(0) # removes oldest element of rolling window
            plot_hist(flat_angles, angle_plotting_vals, S, step)

        if time.time() - start_time > max_runtime:
            print(f"Timed out at step {step}")
            im.set_data(phi_array)
            plt.savefig(f"frames/snapshot_{step:03d}.png", bbox_inches='tight', dpi=80,
                    pil_kwargs={"compress_level": 1})
            end_frame = f"frames/snapshot_{step:03d}.png"
            flat_angles = phi_array.flatten()
            Q,S = calculate_order_param(phi_array)
            angle_plotting_vals.append(Q)
            plot_hist(flat_angles, angle_plotting_vals, S, step)
            end_frame_hist = f"frames_hist/snapshot_{step:03d}.png"
            break
        
        if step == 3*eq_step_estimate:
            im.set_data(phi_array)
            plt.savefig(f"frames/snapshot_{step:03d}.png", bbox_inches='tight', dpi=80,
                    pil_kwargs={"compress_level": 1})
            end_frame = f"frames/snapshot_{step:03d}.png"
            if SAVE_HIST==True:
                flat_angles = phi_array.flatten()
                Q,S = calculate_order_param(phi_array)
                angle_plotting_vals.append(Q)
                plot_hist(flat_angles, angle_plotting_vals, S, step)
                end_frame_hist = f"frames_hist/snapshot_{step:03d}.png"
            break
        step += 1

    plt.close(fig)

    fig, (axE, axS) = plt.subplots(
        1, 2, 
        figsize=(16, 5), 
        sharex=True)
    
    # leave room at top for the legend, and at bottom for the avg-text
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.3)
    
    # --- Energy panel ---
    h1, = axE.plot(energies, color='k', lw=1, alpha=0.8, label='Energy')
    v1 = axE.axvline(eq_step_estimate, color='red', ls='--', alpha=0.7, label='Equilibrium ~reached')
    avgE = np.mean(energies[eq_step_estimate:])
    axE.hlines(avgE, eq_step_estimate, step,
                    colors='red', linestyles='-', lw=1)
    axE.hlines(avgE, 0, eq_step_estimate,
                    colors='red', linestyles='--', lw=0.6)    
    axE.set_title(f"Energy Relaxation, $T^*={T:.2f}$")
    axE.set_xlabel("Monte Carlo step")
    axE.set_ylabel("Total Energy")
    axE.grid(True)
    axE.set_xlim(left=0)
    
    # text underneath (in axis coords)
    axE.text(
        0.5, -0.25,
        rf"$\langle E\rangle = {avgE:.3f}$",
        transform=axE.transAxes,
        ha='center', va='top'
    )
    
    # --- Order-parameter panel ---
    h2, = axS.plot(order_param, color='orange', lw=1, alpha=0.8, label='Order param')
    axS.axvline(eq_step_estimate, color='red', ls='--', alpha=0.7)  # no new label
    avgS = np.mean(order_param[eq_step_estimate:])
    axS.hlines(avgS, eq_step_estimate, step,
                    colors='red', linestyles='-', lw=1)
    axS.hlines(avgS, 0, eq_step_estimate,
                    colors='red', linestyles='--', lw=0.6)    
    axS.set_title(f"Order Parameter, $S$,  $T^*={T:.2f}$")
    axS.set_xlabel("Monte Carlo step")
    axS.set_ylabel("Order Parameter")
    axS.grid(True)
    axS.set_ylim(0,1)
    axS.set_xlim(left=0)
    
    axS.text(
        0.5, -0.25,
        rf"$\langle S\rangle = {avgS:.3f}$",
        transform=axS.transAxes,
        ha='center', va='top'
    )
    
    # --- Figure-level legend ---
    # collect handles + labels from both axes
    handles = [h1, v1, h2]
    labels  = [h.get_label() for h in handles]
    
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False
    )
    # save and show
    plt.savefig(f"{FILE_STRING}/energy_and_order.png", bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(os.getcwd())
     # Create GIF
    if CREATE_GIF == True:
        with imageio.get_writer(FILE_STRING+"/"+f"simulation_{T:.2f}K_{n}by{n}_amp{interval_amplitude}.gif", 
                                mode="I", duration=0.2) as writer: 
            for snap in range(0, 100, small_snapshot_interval):
                filename = f"frames/snapshot_{snap:03d}.png"
                image = imageio.v2.imread(filename)
                writer.append_data(image)
            for snap in range(100, step, snapshot_interval):
                filename = f"frames/snapshot_{snap:03d}.png"
                image = imageio.v2.imread(filename)
                writer.append_data(image)

    if CREATE_GIF_HIST == True:
        with imageio.get_writer(FILE_STRING + "/"+f"hist_simulation_{T:.2f}K_{n}by{n}_amp{interval_amplitude}.gif", 
                                mode="I", duration=0.2) as writer: 
            for snap in range(0, 100, small_snapshot_interval):
                filename = f"frames_hist/angle_histogram_{snap:03d}.png"
                image = imageio.v2.imread(filename)
                writer.append_data(image)
            for snap in range(0, step, snapshot_interval):
                filename = f"frames_hist/angle_histogram_{snap:03d}.png"
                image = imageio.v2.imread(filename)
                writer.append_data(image)
            if os.path.exists(end_frame_hist):
                filename = end_frame_hist
                image = imageio.v2.imread(filename)
                writer.append_data(image)
    return phi_array, energies, order_param

def ramp_sweep(T_start=1.2, T_end=0.4, dT=0.2):
    """
    Sweep T downward from T_start to T_end (inclusive) in steps of dT.
    At each T:
      • Warm‐start from the last phi (None for the first T)
      • Run main(phi_init) to completion → (phi, energies, order_param)
      • Once len(energies) > eq_step_estimate, compute
          E_avg = mean(energies[eq_step_estimate:])
          S_avg = mean(order_param[eq_step_estimate:])
      • Otherwise set E_avg/S_avg = nan
    Returns list of (T, E_avg, S_avg).
    """

   
    # 1) Make a single top-level data folder

    if os.path.isdir(ramp_folder):
        ask = input("Do you want to delete the existing ramp data?(Y/N):")
        if ask == "Y":
            shutil.rmtree(ramp_folder,onerror=remove_readonly)
            print("ramp_data folder deleted!")
        else:
            return 0 
    os.makedirs(base_dir, exist_ok=True)

    Ts = np.arange(T_start, T_end - 1e-9, -dT)
    summary = []
    phi = None

    for Tval in Ts:
        print(f"\n=== Running at T* = {Tval:.2f} ===")
        sim_name = f"/simulation_{Tval:.2f}K_{n}by{n}_amp{interval_amplitude}"
        # Build per‐Temperature simulations directory
        sim_dir = os.path.join(base_dir, sim_name)
        os.makedirs(sim_dir, exist_ok=True)
        eq_step_estimate = int(n**2)
        
        # update globals so your existing main() picks up the new temperature
        globals()['T']    = Tval
        globals()['beta'] = 1.0 / Tval
        globals()['FILE_STRING'] = ramp_folder+sim_name
        globals()['sim_name'] = sim_name   
        globals()['eq_step_estimate'] = eq_step_estimate
        # run your simulation (phi_init=None on the first pass)
        phi, energies, order_param = main(phi)
        
        # 6) Save the trajectories (no nested ramp_data folder)
        np.save(os.path.join(sim_dir, "energies.npy"), energies)
        np.save(os.path.join(sim_dir, "order_param.npy"), order_param)

        # compute averages *only* after eq_step_estimate burn-in
        if len(energies) > eq_step_estimate:
            E_avg = np.mean(energies[eq_step_estimate:])
            S_avg = np.mean(order_param[eq_step_estimate:])
        else:
            E_avg = 0
            S_avg = 0

        summary.append((Tval, E_avg, S_avg))

    return summary

if __name__ == "__main__":
# =============================================================================
#     profile_file = "profile.out"
#     cProfile.run('main()', filename=profile_file)
# 
#     # Then analyze
#     p = pstats.Stats(profile_file)
#     #p.strip_dirs().sort_stats('tottime')
#     #p.print_stats(20)
# =============================================================================
# =============================================================================
#     profile_file = "profile.out"
#     cProfile.run('main()', filename=profile_file)
#     
# =============================================================================
    stats = ramp_sweep(T_start=0.2, T_end=2, dT=-0.2)
    
    # unpack for plotting
    Ts   = [s[0] for s in stats]
    Eavs = [s[1] for s in stats]
    Savs = [s[2] for s in stats]

    fig, ax = plt.subplots()
    ax.plot(Ts, Eavs, 'o-', label=r'$\langle E\rangle$', color = 'dodgerblue')
    ax.plot(Ts, Savs, 's-', alpha=0, label=r'$\langle S\rangle$', color = 'orange')
    ax.tick_params(axis='y', labelcolor='dodgerblue')
    ax2 = ax.twinx()
    ax2.plot(Ts, Savs, 's-', color='orange')
    ax2.set_ylabel(r'Average $S$')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.set_xlabel('T*')
    ax.set_ylabel(r'Average $E$')
    ax.legend(loc='upper left')
    ax.invert_xaxis()   # T decreasing left→right
    ax.set_title(f"{n}by{n} Average $S$,$E$ with decreasing $T^*$")
    fig.savefig(os.path.join(base_dir, "avg_vs_T.png"), bbox_inches='tight')
    plt.show()



# Add GUI? Probably not the best for this type of program?
