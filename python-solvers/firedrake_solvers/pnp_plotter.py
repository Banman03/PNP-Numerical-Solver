import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.animation as animation
import numpy as np
import matplotlib.tri as tri
import os
import shutil

def prepare_results_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created directory: {folder_name}")
    else:
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"Cleared existing contents of: {folder_name}")

def plot_solutions(U_prev, z_vals, mode, num_steps, dt, t, results_dir, n):
    fig, axes = plt.subplots(1, n+1, figsize=(5*(n+1), 4))

    # Plot c0
    tripcolor(U_prev.sub(0), axes=axes[0], cmap='viridis')
    axes[0].set_title(f'c0 (z={z_vals[0]:+d}) at t={t:.3f}')
    plt.colorbar(axes[0].collections[0], ax=axes[0])

    # Plot c1
    tripcolor(U_prev.sub(1), axes=axes[1], cmap='plasma')
    axes[1].set_title(f'c1 (z={z_vals[1]:+d}) at t={t:.3f}')
    plt.colorbar(axes[1].collections[0], ax=axes[1])
    
    # Plot c2
    tripcolor(U_prev.sub(2), axes=axes[2], cmap='plasma')
    axes[1].set_title(f'c2 (z={z_vals[2]:+d}) at t={t:.3f}')
    plt.colorbar(axes[2].collections[0], ax=axes[2])
    
    # Plot c3
    tripcolor(U_prev.sub(3), axes=axes[3], cmap='plasma')
    axes[1].set_title(f'c3 (z={z_vals[3]:+d}) at t={t:.3f}')
    plt.colorbar(axes[3].collections[0], ax=axes[3])

    # Plot phi
    tripcolor(U_prev.sub(4), axes=axes[4], cmap='coolwarm')
    axes[2].set_title(f'phi at t={t:.3f}')
    plt.colorbar(axes[4].collections[0], ax=axes[4])

    bv_status = "with BV" if mode == 1 else "with Robin" if mode == 2 else "Normal"
    fig.suptitle(f'PNP Solution ({bv_status}): {num_steps} steps, dt={dt}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save to the specific folder
    output_path = os.path.join(results_dir, 'pnp_solution_init.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved final state visualization to {output_path}")
    plt.show()
    
def create_animations(snapshots, mode, mesh, results_dir):
    print(f"\nGenerating animations to {results_dir}...")
    coords = mesh.coordinates.dat.data_ro
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1])

    def create_animation_sub(data_list, title, cmap, filename):
        fig_anim, ax = plt.subplots(figsize=(6, 5))
        vmin, vmax = np.min(data_list), np.max(data_list)
        tpc = ax.tripcolor(triangulation, data_list[0], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(tpc, ax=ax, label=title)

        def update(frame):
            ax.clear()
            tpc = ax.tripcolor(triangulation, data_list[frame], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} | t = {snapshots['t'][frame]:.10f}")
            return tpc,

        anim = animation.FuncAnimation(fig_anim, update, frames=len(data_list), interval=100)

        # Full path for the animation
        full_path = os.path.join(results_dir, filename)

        try:
            anim.save(full_path + '.mp4', writer='ffmpeg', fps=10, dpi=150)
            print(f"  Saved {full_path}.mp4")
        except:
            anim.save(full_path + '.gif', writer='pillow', fps=10)
            print(f"  Saved {full_path}.gif")

        plt.close(fig_anim)

    create_animation_sub(snapshots['c0'], 'c0', 'viridis', f'c0_anim_{mode}')
    create_animation_sub(snapshots['c1'], 'c1', 'plasma', f'c1_anim_{mode}')
    create_animation_sub(snapshots['c2'], 'c2', 'plasma', f'c2_anim_{mode}')
    create_animation_sub(snapshots['c3'], 'c3', 'plasma', f'c3_anim_{mode}')
    create_animation_sub(snapshots['phi'], 'phi', 'coolwarm', f'phi_anim_{mode}')