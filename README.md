# Shape optimization for convex shapes
This software solves shape optimization problems with PDE constraints and convexity constraints in two spatial dimensions.
The algorithm is explained in detail in the preprint https://arxiv.org/abs/1810.10735.

To run the software, you need python3 (with mathplotlib, scipy), paraview (with python support), [FEniCS](http://fenicsproject.org) (tested with v2018.1), and [OSQP](https://osqp.org/). OSQP can be installed by `pip3 install osqp`.

The examples from the preprint can be run by executing `run_all.sh`.
