import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.animation as animation
import numpy as np
import matplotlib.tri as tri

def plot_solutions(U_prev, z_vals, mode, num_steps, dt, t):
    # Create figure with subplots for each field
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot concentration of species 0
    tripcolor(U_prev.sub(0), axes=axes[0], cmap='viridis')
    axes[0].set_title(f'Concentration c₀ (z={z_vals[0]:+d}) at t={t:.3f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='c₀')

    # Plot concentration of species 1
    tripcolor(U_prev.sub(1), axes=axes[1], cmap='plasma')
    axes[1].set_title(f'Concentration c₁ (z={z_vals[1]:+d}) at t={t:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='c₁')

    # Plot electric potential
    tripcolor(U_prev.sub(2), axes=axes[2], cmap='coolwarm')
    axes[2].set_title(f'Electric Potential phi at t={t:.3f}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='phi')

    bv_status = "with BV" if mode == 1 else "with Robin" if mode == 2 else "Normal"
    fig.suptitle(f'PNP Solution ({bv_status}): {num_steps} time steps, dt={dt}', fontsize=14, y=1.02)
    plt.tight_layout()
    output_png = f'pnp_solution_init.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Saved final state visualization to {output_png}")
    plt.show()
    
def create_animations(snapshots, mode, mesh):
    print(f"\nGenerating animations from {len(snapshots['t'])} snapshots...")

    # Get mesh coordinates for plotting
    coords = mesh.coordinates.dat.data_ro
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Create triangulation for plotting
    triangulation = tri.Triangulation(x_coords, y_coords)

    def create_animation_sub(data_list, title, cmap, filename):
        """Create animation for a single field"""
        fig_anim, ax = plt.subplots(figsize=(6, 5))

        # Determine global colorbar limits
        vmin = min(np.min(d) for d in data_list)
        vmax = max(np.max(d) for d in data_list)

        # Initial plot
        tpc = ax.tripcolor(triangulation, data_list[0], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.colorbar(tpc, ax=ax, label=title)

        def update(frame):
            ax.clear()
            tpc = ax.tripcolor(triangulation, data_list[frame], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.set_title(title)
            time_text = ax.text(0.02, 0.95, f't = {snapshots["t"][frame]:.3f}',
                            transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            return tpc, time_text

        anim = animation.FuncAnimation(fig_anim, update, frames=len(data_list),
                                    interval=100, blit=False, repeat=True)

        # Try to save as MP4, fall back to GIF if ffmpeg not available
        try:
            anim.save(filename + '.mp4', writer='ffmpeg', fps=10, dpi=150)
            print(f"  Saved {filename}.mp4")
        except:
            try:
                anim.save(filename + '.gif', writer='pillow', fps=10)
                print(f"  Saved {filename}.gif (ffmpeg not available)")
            except Exception as e:
                print(f"  Warning: Could not save animation for {title}: {e}")

        plt.close(fig_anim)

    # Create animations for each field
    create_animation_sub(snapshots['c0'], 'Concentration c₀', 'viridis', f'c0_animation{mode}')
    create_animation_sub(snapshots['c1'], 'Concentration c₁', 'plasma', f'c1_animation{mode}')
    create_animation_sub(snapshots['phi'], 'Electric Potential phi', 'coolwarm', f'phi_animation{mode}')

    print("Animation generation complete!")