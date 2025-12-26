# MolGrad

MolGrad is a minimalist C++/CUDA library for automatic differentiation and tensor computation, designed for biomolecular and structural biology research.

## Motivation

Modern deep learning frameworks abstract away critical implementation details
such as memory layout, kernel fusion, and gradient graph construction.
While this is ideal for rapid experimentation, it obscures how these systems work
at a low level and limits their adaptability to non-standard scientific domains.

MolGrad is an educational and research-oriented project aimed at:

- Understanding automatic differentiation at the graph level
- Implementing tensor operations with explicit memory control
- Exploring differentiable computation for 3D biomolecular structures


## Project Status

MolGrad is a long-term personal research project.
It is not intended as a production-ready deep learning framework,
but as a platform for experimentation, learning, and scientific exploration.