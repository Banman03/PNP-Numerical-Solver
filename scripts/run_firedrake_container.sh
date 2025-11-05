# xhost +local:root

# docker run -it --rm \
#     -v /home/banman03/Euclid/AdaLovelace/pp/PNP-Numerical-Solver:/home/firedrake/shared \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     --device /dev/dri \
#     firedrakeproject/firedrake:latest \
#     /bin/bash

docker run -it --rm \
    -v /home/banman03/Euclid/AdaLovelace/pp/PNP-Numerical-Solver:/home/firedrake/shared \
    -e DISPLAY="" \
    firedrakeproject/firedrake:latest \
    /bin/bash
