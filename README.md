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