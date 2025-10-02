# PNP-Numerical-Solver

## Installing the Necessary Dependencies for the Python Interface
To remain consistent with package/dependency management best practices, either a Conda environment or a Docker container should be used to manage dependencies for FEniCS/Dolfinx.

To create/activate the conda environment, run:

`conda create -n <environment-name>`

`conda activate <environment-name>`

`conda install -c conda-forge fenics-dolfinx mpich pyvista matplotlib cycler`

You can then easily deactivate/reactivate it with:

`conda deactivate`

`conda activate <environment-name>`


## Useful links
- `dolfinx repo`: https://github.com/FEniCS/dolfinx
- `dolfinx python tutorials`: https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos.html
- `dolfin-adjoint`: https://github.com/dolfin-adjoint/dolfin-adjoint
- `Paper detailing the PNP equations with maufactured solutions with convergence results`: https://arxiv.org/pdf/2105.01163
- `Paper detailing the PNP model`: https://pmc.ncbi.nlm.nih.gov/articles/PMC3122111/