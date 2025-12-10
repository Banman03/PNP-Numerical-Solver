#!/usr/bin/env python3
"""
Mesh refinement convergence study for PNP solver using MMS.

This script:
1. Runs pnp_solver_mms.py with increasing mesh refinement
2. Collects error data (L2 and H1 norms)
3. Plots convergence rates on log-log plots
4. Verifies theoretical convergence rates

Expected convergence rates:
- Linear elements (order=1): L2 error ~ O(h^2), H1 error ~ O(h)
- Quadratic elements (order=2): L2 error ~ O(h^3), H1 error ~ O(h^2)
"""

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def run_mms_solver(nx, ny, order, dt=1e-3, t_end=0.1):
    """
    Run the MMS solver with specified parameters.

    Args:
        nx, ny: Number of elements in each direction
        order: Polynomial order (1 or 2)
        dt: Time step
        t_end: End time

    Returns:
        Dictionary with error data
    """
    cmd = [
        'python3', 'pnp_solver_mms.py',
        '--nx', str(nx),
        '--ny', str(ny),
        '--order', str(order),
        '--dt', str(dt),
        '--t_end', str(t_end)
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("  ✓ Completed")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed with return code {e.returncode}")
        print(f"  stderr: {e.stderr}")
        return None

    # Read the generated CSV file
    csv_file = f"mms_errors_nx{nx}_ny{ny}_order{order}.csv"
    try:
        df = pd.read_csv(csv_file)
        return df.iloc[0].to_dict()
    except FileNotFoundError:
        print(f"  ✗ Output file {csv_file} not found")
        return None


def compute_convergence_rate(h_values, error_values):
    """
    Compute convergence rate from log-log fit.

    Convergence rate p is computed from: error ~ C * h^p
    Using linear regression on log(error) vs log(h)

    Returns:
        Slope (convergence rate)
    """
    log_h = np.log(h_values)
    log_error = np.log(error_values)

    # Linear fit: log(error) = log(C) + p * log(h)
    coeffs = np.polyfit(log_h, log_error, 1)
    slope = coeffs[0]

    return slope


def plot_convergence(results_dict, title, filename):
    """
    Create convergence plots for spatial refinement study.

    Args:
        results_dict: Dictionary with keys as (order, error_type) and values as dataframes
        title: Plot title
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot L2 errors
    ax = axes[0]
    for order in [1, 2]:
        if (order, 'L2') in results_dict:
            df = results_dict[(order, 'L2')]
            h = df['h_avg'].values

            # Plot each species
            for species in ['c0', 'c1', 'phi']:
                col_name = f'L2_{species}'
                if col_name in df.columns:
                    error = df[col_name].values
                    rate = compute_convergence_rate(h, error)
                    label = f'{species} (order {order}, rate={rate:.2f})'
                    ax.loglog(h, error, 'o-', label=label, markersize=8)

    ax.set_xlabel('Element size h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('L2 Norm Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Plot H1 errors
    ax = axes[1]
    for order in [1, 2]:
        if (order, 'H1') in results_dict:
            df = results_dict[(order, 'H1')]
            h = df['h_avg'].values

            for species in ['c0', 'c1', 'phi']:
                col_name = f'H1_{species}'
                if col_name in df.columns:
                    error = df[col_name].values
                    rate = compute_convergence_rate(h, error)
                    label = f'{species} (order {order}, rate={rate:.2f})'
                    ax.loglog(h, error, 's-', label=label, markersize=8)

    ax.set_xlabel('Element size h', fontsize=12)
    ax.set_ylabel('H1 Error', fontsize=12)
    ax.set_title('H1 Norm Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Plot DOFs vs error for efficiency comparison
    ax = axes[2]
    for order in [1, 2]:
        if (order, 'L2') in results_dict:
            df = results_dict[(order, 'L2')]
            dofs = df['DOFs'].values

            # Just plot phi for clarity
            error = df['L2_phi'].values
            ax.loglog(dofs, error, 'o-', label=f'phi L2 (order {order})', markersize=8)

    ax.set_xlabel('Total DOFs', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Error vs DOFs', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Convergence plot saved to: {filename}")
    plt.close()


def main():
    """Run spatial convergence study."""

    print("="*70)
    print("PNP SOLVER - SPATIAL CONVERGENCE STUDY")
    print("="*70)
    print("\nThis will run the MMS solver with increasing mesh refinement")
    print("and verify that the numerical solution converges at the expected rate.\n")

    # Refinement levels: start with 4x4, then 8x8, 16x16, 32x32
    refinement_levels = [4, 8, 16, 32]

    # Test both linear (order=1) and quadratic (order=2) elements
    orders_to_test = [1, 2]

    # Use short time for convergence study (spatial errors dominate)
    dt = 1e-4
    t_end = 0.01

    # Storage for results
    all_results = {}

    for order in orders_to_test:
        print(f"\n{'='*70}")
        print(f"Testing order {order} elements")
        print(f"{'='*70}\n")

        results = []

        for nx in refinement_levels:
            ny = nx  # Square mesh
            print(f"\n--- Mesh: {nx}x{ny}, order={order} ---")

            error_data = run_mms_solver(nx, ny, order, dt, t_end)

            if error_data is not None:
                results.append(error_data)
            else:
                print(f"Skipping mesh {nx}x{ny} due to error")

        if results:
            df = pd.DataFrame(results)
            all_results[(order, 'L2')] = df
            all_results[(order, 'H1')] = df

            # Print summary table
            print(f"\n{'='*70}")
            print(f"Summary for order {order} elements:")
            print(f"{'='*70}")
            print("\nMesh refinement data:")
            print(df[['h_avg', 'DOFs', 'L2_c0', 'L2_phi', 'H1_c0', 'H1_phi']].to_string(index=False))

            # Compute and print convergence rates
            if len(df) >= 2:
                print(f"\nConvergence rates for order {order}:")
                print(f"  Expected L2 rate: {order+1:.1f}")
                print(f"  Expected H1 rate: {order:.1f}")

                h = df['h_avg'].values

                rate_L2_c0 = compute_convergence_rate(h, df['L2_c0'].values)
                rate_L2_phi = compute_convergence_rate(h, df['L2_phi'].values)
                rate_H1_c0 = compute_convergence_rate(h, df['H1_c0'].values)
                rate_H1_phi = compute_convergence_rate(h, df['H1_phi'].values)

                print(f"  Actual L2 rate (c0): {rate_L2_c0:.2f}")
                print(f"  Actual L2 rate (phi): {rate_L2_phi:.2f}")
                print(f"  Actual H1 rate (c0): {rate_H1_c0:.2f}")
                print(f"  Actual H1 rate (phi): {rate_H1_phi:.2f}")

    # Create convergence plots
    if all_results:
        plot_convergence(all_results,
                        'PNP Solver Spatial Convergence Study',
                        'pnp_convergence_study.png')

    print("\n" + "="*70)
    print("CONVERGENCE STUDY COMPLETE")
    print("="*70)
    print("\nInterpretation:")
    print("- Convergence rate should match theoretical prediction")
    print("- Order 1 elements: L2 ~ O(h²), H1 ~ O(h)")
    print("- Order 2 elements: L2 ~ O(h³), H1 ~ O(h²)")
    print("\nIf rates are lower than expected, check:")
    print("- Time step is small enough (temporal error negligible)")
    print("- Manufactured solution is smooth enough")
    print("- Boundary conditions are applied correctly")
    print("="*70)


if __name__ == '__main__':
    main()
