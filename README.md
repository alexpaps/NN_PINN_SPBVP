# Singularly Perturbed ODEs & Physics-Informed Neural Networks (PINNs)

## Overview

This repository contains a collection of implementations and experiments on **singularly perturbed ordinary differential equations (ODEs)** using **Physics-Informed Neural Networks (PINNs)** and asymptotic-informed neural architectures.

The main goal is to investigate how neural networks behave in regimes with boundary layers and small perturbation parameters (ε ≪ 1), and to compare classical PINN formulations with structured asymptotic and multiscale approaches.

---

## Mathematical Problems Studied

### 1. Smooth ODE
We consider the simple problem:
$y'(x) = x, \quad x \in [0,1], \quad y(0)=0$
with exact solution:
$y(x) = \frac{x^2}{2}$

This serves as a baseline case for standard neural network approximation.

---

### 2. Singularly Perturbed Boundary Value Problem
We study:
$-\varepsilon^2 u''(x) + u(x) = 1, \quad x \in [0,1], \quad u(0)=u(1)=0$

with exact solution:
$u(x) = \frac{1 - e^{-x/\varepsilon} + e^{-(1-x)/\varepsilon}}{1 + e^{-1/\varepsilon}}$

This problem exhibits **boundary layers at x = 0 and x = 1** for small ε.

---

## Methods Implemented

This repository explores multiple neural and asymptotic strategies:

### Case 1 — Strong Boundary Enforcement
Boundary conditions are enforced via:
$u_{NN}(x) = x(1-x) y_{NN}(x)$

---

### Case 2 — Weak Boundary Conditions
Boundary conditions are added as penalty terms in the loss function.

---

### Case 3 — Transfer Learning in ε
Training is performed sequentially:
$\varepsilon = 1 \rightarrow 0.1 \rightarrow 0.01$

---

### Case 4 — Boundary Layer Decomposition
Explicit modeling of boundary layers:
$u_{NN}(x) = y_{NN}(x) + B e^{-x/\varepsilon} + C e^{-(1-x)/\varepsilon}$

---

### Case 5 — Smooth Component Decomposition
Outer solution approximation:
$u_S(x) = \frac{f(x)}{b(x)}$

Neural network learns residual corrections.

---

### Case 6 — Combined Decomposition
Combination of smooth + boundary layer structure + neural correction.

---

### Case 7 — Variational / Ritz PINN Formulation
The solution is obtained by minimizing:
$E(u) = \frac{1}{2}\int_0^1 (\varepsilon^2 |u'|^2 + b(x)u^2)\,dx - \int_0^1 f(x)u\,dx$

Boundary conditions are strongly enforced and ε-continuation is used.

---

### Case 8 — Unified Asymptotic-Informed PINN Ansatz

A multiscale neural architecture combining:

- asymptotic outer solution
- explicit boundary layers
- neural correction networks

$u(x;\varepsilon) = u_{asym}(x;\varepsilon) + \sigma(\alpha) u_{ms}(x;\varepsilon)$

This approach separates:
- deterministic asymptotic structure
- learned corrections
- multiscale residual effects

---

## Repository Structure
