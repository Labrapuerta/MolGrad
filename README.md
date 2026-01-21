# MolGrad

MolGrad is a minimalist C++/CUDA library for automatic differentiation and tensor computation, designed for biomolecular and structural biology research.

The project focuses on low-level implementation details such as memory layout, tensor slicing, and explicit gradient graph construction, providing a foundation for understanding and experimenting with differentiable programming in scientific domains.

## Motivation

Modern deep learning frameworks abstract away critical implementation details
such as memory layout, kernel fusion, and gradient graph construction.
While this is ideal for rapid experimentation, it obscures how these systems work
at a low level and limits their adaptability to non-standard scientific domains.

MolGrad is an educational and research-oriented project aimed at:

- Understanding automatic differentiation at the graph level
- Implementing tensor operations with explicit memory control
- Exploring differentiable computation for 3D biomolecular structures
- CUDA acceleration of tensor operations and low level kernels


## Project Status

MolGrad is a long-term personal research project.
It is not intended as a production-ready deep learning framework,
but as a platform for experimentation, learning, and scientific exploration.

- [x] Tensor implementation with storage, slices. 
- [x] Memory management and layout (contiguous, strides, clone).
- [] Basic tensor operations (add, multiply, matmul) in CPU. 
- [] Python bindings via Pybind11
- [] CPU and CUDA tensor operations
- [] Automatic differentiation engine with backward graph construction
- [] Basic neural network layers and loss functions
- [] Examples in biomolecular structure prediction and analysis