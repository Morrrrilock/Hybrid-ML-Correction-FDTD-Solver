# Hybrid-ML-Correction-FDTD-Solver
A research-oriented framework for improving 3D FDTD acoustic simulations through machine-learning-based correction models.

# Overview

This repository explores the integration of machine learning into finite-difference time-domain (FDTD) solvers for acoustic wave propagation.

The long-term goal of this research is to develop a data-driven correction framework in which machine learning models are trained using psychoacoustic experimental measurements and subsequently embedded into FDTD simulations to improve prediction accuracy.

In realistic acoustic environments, discrepancies often exist between numerical simulations and human auditory perception. Traditional FDTD methods solve the governing wave equation accurately from a numerical perspective, but they do not explicitly account for perceptual effects revealed by psychoacoustic experiments.

The ideal workflow is therefore:
```text
Psychoacoustic Experiments
            ↓
Measured Acoustic Responses
            ↓
Machine Learning Training
            ↓
Learned Correction Model
            ↓
Embedded into FDTD Solver
            ↓
Perceptually Improved Acoustic Simulation
```

However, psychoacoustic datasets suitable for this purpose are currently unavailable.

As an initial proof-of-concept study, analytical solutions are used as surrogate ground-truth data to train neural networks that learn the discrepancy between standard FDTD solutions and reference solutions.

The repository contains two generations of correction models:

- Local source-point correction.
- Full 3D pressure-field correction.

These studies demonstrate the feasibility of embedding machine learning directly into numerical wave solvers.

# Method 1: Local Source-Point Correction

A lightweight neural network is trained to learn the discrepancy between:

Standard FDTD prediction
Analytical solution

at the source point.

The model receives:

- Current pressure
- Previous pressure
- Source frequency
- Source amplitude
- Simulation time

and predicts a correction term，which is added directly to the source-point update.

This strategy improves agreement with the analytical solution locally while preserving the original FDTD framework.

# Method 2: Full 3D Pressure-Field Correction

The method extends the correction mechanism from a single point to the entire three-dimensional pressure field. Instead of predicting a scalar correction value, a 3D convolutional encoder–decoder network learns a spatial correction field. Inputs include: 
- Current pressure field
- Previous pressure field -
- Source frequency
- Source strength
- Simulation time

The network predicts a spatial correction field, denoted as Δp(x, y, z, t), for every grid point in the computational domain.

The predicted correction field is then added to the FDTD solution:

p_corrected = p_FDTD + Δp

where:
- p_FDTD is the pressure field obtained from the standard FDTD solver.
- Δp is the neural-network-predicted correction field.
- p_corrected is the corrected pressure field.




