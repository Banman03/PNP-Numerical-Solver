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

## Using Firedrake
Because dolfinx does not support dolfin-adjoint, I am exploring the use of Firedrake as an alternative.

This is because there exists an alternative algorithmic differentiation tool, Pyadjoint, that is built on Firedrake.

To create/activate the conda environment, run the same commands as above in order to activate or deactivate it.

Here is a link to detailed installation instructions: `https://www.firedrakeproject.org/install.html`

### In Docker
This is the easiest method (in my opinion) of running Firedrake.

To pull, the image, run `docker pull firedrakeproject/firedrake:latest`

Once pulled, run the image using `docker run -it firedrakeproject/firedrake:latest`. (This runs the container with root privileges, so be careful with any commands you run while the container has elevated privileges)


## Relevant Repositories
- `dolfin-adjoint repo`: https://github.com/dolfin-adjoint/dolfin-adjoint
- `dolfinx repo`: https://github.com/FEniCS/dolfinx (this contains both the Python and C++ interfaces; I am going to try solving with the C++ interface as it is more verbose)

## Relevant Code Examples
- `dolfinx python tutorials`: https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos.html

## Relevant Papers
- `Paper detailing the PNP equations with maufactured solutions with convergence results`: https://arxiv.org/pdf/2105.01163
- `Paper detailing the PNP model`: https://pmc.ncbi.nlm.nih.gov/articles/PMC3122111/
- `Paper detailing reasons why Firedrake might be preferred over FEniCSx`: https://www.osti.gov/servlets/purl/2370161
- `Paper detailing automatic adjoints in Firedrake (automatic disjoint methods are not supported in FEniCSx)`: https://joss.theoj.org/papers/10.21105/joss.01292