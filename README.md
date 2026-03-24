# Double Inverted Pendulum

![Python](https://img.shields.io/badge/Python-3.11.9-blue) 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Research-green)
![Visualization](https://img.shields.io/badge/Data-Visualization-red)
![Data Processing](https://img.shields.io/badge/Data-Preprocessing-purple)
![Control](https://img.shields.io/badge/Field-Control%20Systems-orange)
![Time Series](https://img.shields.io/badge/Analysis-Time%20Series-yellow)
![Nonlinear Systems](https://img.shields.io/badge/Dynamics-Nonlinear-black)



A project dedicated to the analysis, preprocessing, and modeling of the Double Inverted Pendulum system for control, prediction and intelligent control applications.

---

## The Double Inverted Pendulum Problem

The Double Inverted Pendulum consists of two rigid links connected in series, where the first link is attached to a base and the second link is attached to the end of the first. Both links must be balanced in an upright unstable equilibrium.

Unlike the simple inverted pendulum, the double inverted pendulum presents strong nonlinear behavior and chaotic dynamics, making the system significantly more challenging to analyze and control.

Thus, small variations in initial conditions or disturbances can lead to large differences in system behavior, which makes accurate modeling and control strategies essential.
In the below figure we know what 


## Characteristics of the System

The Double Inverted Pendulum presents several important properties:

* **Nonlinear dynamics**
* **Strong coupling between states**
* **Highly unstable equilibrium**
* **Chaotic behavior**
* **High sensitivity to disturbances**

Because of these characteristics, classical linear control techniques are often insufficient when applied directly to the full system dynamics.

---

# Motivation

The main motivation for studying the double inverted pendulum is its importance as a benchmark problem for advanced control techniques, including:

* Nonlinear Control
* Optimal Control
* Adaptive Control
* Reinforcement Learning
* Artificial Intelligence for Control Systems

It is also widely used in research involving:

* Robotics
* Dynamical systems
* Control theory
* Machine learning applied to physical systems

---

# Project Objectives

This project focuses on the following tasks:

* Acquisition of experimental or simulated data from the system
* **Preprocessing and cleaning of time series data**
* Analysis of angular positions and torques
* Mathematical modeling of the system dynamics
* Preparation of datasets for control algorithm development

---

# Repository Structure

```
Double-Inverted-Pendulum
│
├── Inverted Pendulum/
│   └── Final Versions/
│       ├── Codes
│       ├── Data
│       ├── Data Processed
│       ├── Graphs
│       ├── Models
│       │
│       ├── Project_1/
│       │   ├── H_1
│       │   ├── H_2
│       │   ├── H_3
│       │   ├── H_4
│       │   └── H_5
│       │
│       ├── Project_2/
│       │   ├── H_1
│       │   ├── H_2
│       │   ├── H_3
│       │   ├── H_4
│       │   └── H_5
│       │
│       ├── Project_3/
│       │   ├── H_1
│       │   ├── H_2
│       │   ├── H_3
│       │   ├── H_4
│       │   └── H_5
│       │
│       └── Project_Final/
│           ├── H_1
│           ├── H_2
│           ├── H_3
│           ├── H_4
│           └── H_5
|
├── README.md
|
```

---



## Data-Centric Framework for the Double Inverted Pendulum

---

# Part I — State Variables and Formal System Representation

To perform data acquisition, the system must be simulated in a controlled environment, specifically a high-fidelity physics simulation. For this purpose, the MuJoCo Python library, developed by Google Research, was employed. This approach enables the collection of data corresponding to the variables defined below.
In practice, the Double Inverted Pendulum is treated strictly as a data-generating nonlinear system, without requiring an explicit analytical solution of its equations of motion. Therefore, the system is defined in terms of its observable state variables:

$$
\mathbf{x}(t) =
\begin{bmatrix}
\theta_1(t) \
\theta_2(t) \
\dot{\theta}_1(t) \
\dot{\theta}_2(t)
\end{bmatrix}
$$

where:

* $\theta_1, \theta_2$ — angular positions
* $\dot{\theta}_1, \dot{\theta}_2$ — angular velocities

Additionally, the system provides **intrinsic dynamic torques**:

$$
\boldsymbol{\tau}^{dyn}(t) =
\begin{bmatrix}
\tau_1(t) \
\tau_2(t)
\end{bmatrix}
$$

and energy-related quantities:

$$
E(t) = T(t) + V(t)
$$

Thus at each time step, the system is sampled as:

$$
\mathcal{D}_t =
\begin{bmatrix}
\theta_1 & \theta_2 &
\dot{\theta}_1 & \dot{\theta}_2 &
\tau_1 & \tau_2 &
T & V
\end{bmatrix}
$$

The dataset is therefore a discrete time series:

$$
\mathcal{D} = { \mathcal{D}*t }*{t=1}^{N}
$$


Due to the inherently chaotic nature of the system under study, multiple simulations were conducted within the MuJoCo environment to ensure a sufficiently large and representative dataset. Owing to the system’s sensitivity to initial conditions, each simulation produces distinct trajectories in terms of angular positions, torques, and other relevant variables.

Therefore, performing multiple simulations constitutes the most appropriate approach for data analysis, as it enables the capture of a broader range of possible system behaviors and improves the robustness of subsequent exploratory data analysis (EDA).

From a data standpoint, the system exhibits:

* Nonlinearity in state transitions
* Strong coupling between variables
* High sensitivity to initial conditions
* Non-stationary temporal patterns

Thus, multiple simulation episodes are required:

$$
\mathcal{D} = \bigcup_{k=1}^{K} \mathcal{D}^{(k)}
$$

where each episode $k$ corresponds to a different initial condition.

## Observation: Dynamical Representation and Implicit State Reconstruction

In this project, the Double Inverted Pendulum is treated strictly as a data-generating nonlinear system, where the analysis is conducted based on observable variables rather than explicit analytical formulations. Although angular velocity is not directly included in the dataset, the system’s dynamical behavior remains fully accessible through its temporal structure.

Even in the absence of explicit measurements of $\dot{\theta}$, the time series inherently preserves dynamical information. This is due to the fact that variations between consecutive samples encode motion, allowing derivatives to be approximated through finite differences:

$$
\dot{\theta}(t) \approx \frac{\theta(t+1) - \theta(t)}{\Delta t}
$$

Thus, velocity information is implicitly embedded in the evolution of the signal over time. As a consequence, sequences such as $x(t), x(t+1), x(t+2), \dots$ contain sufficient information to approximate local dynamics, since temporal differences act as estimators of derivatives and multi-step dependencies reflect higher-order behavior.

To further exploit this property, delay embedding techniques are employed. By constructing vectors composed of time-delayed observations, it becomes possible to reconstruct the system’s phase space from a single observable:

$$
\mathbf{y}(t) =
\begin{bmatrix}
x(t) \\
x(t+\tau) \\
x(t+2\tau)
\end{bmatrix}
$$

This transformation enables an implicit reconstruction of the system’s state space. In practice, it allows the recovery of the underlying dynamics typically described by $(\theta, \dot{\theta})$, even though only position-related measurements are explicitly available.

Consequently, the modeling paradigm shifts from an explicit state-space representation to an implicit dynamical reconstruction based solely on temporal correlations. This approach is particularly suitable for nonlinear and chaotic systems, where analytical solutions are often intractable or unnecessary for data-driven analysis.

It is important to note that this reconstruction is indirect and depends on appropriate choices of embedding parameters, such as time delay ($\tau$) and embedding dimension, as well as on data quality factors including sampling rate and noise.

From a practical standpoint, the dataset used in this project does not explicitly include angular velocities; however, it preserves their informational content through temporal dependencies. This justifies the adoption of delay embedding, autocorrelation analysis, and phase-space reconstruction as fundamental tools for exploratory data analysis (EDA) and nonlinear system characterization.


---

# Part II — Tidy Data Transformation and Trigonometric Embedding

## Motivation

Angular variables suffer from discontinuities:

$$
\theta \equiv \theta + 2\pi
$$

This creates artificial jumps in raw datasets. To address this, the transformation:

$$
\theta \rightarrow (\sin\theta, \cos\theta)
$$

is applied.

$$
\sin^2{\theta} + \cos^2{\theta} = 1
$$

This embeds angular states into a continuous manifold $S^1$.

---

## Tidy Data Structure

The dataset is reorganized into a tidy format, where:

- each row corresponds to a single observation (time step)
- each column represents a distinct variable

This structure ensures consistency, interpretability, and compatibility with data analysis and machine learning workflows. In particular, it enables:

- efficient vectorized operations
- straightforward statistical analysis
- direct integration with visualization and EDA tools
- compatibility with learning algorithms that assume tabular input

Additionally, representing angular variables using sine and cosine transformations avoids discontinuities at $2\pi$ and preserves the geometric structure of the state space.

The resulting dataset is defined as:

$$
X_{\text{tidy}} =
\begin{bmatrix}
\sin(\theta_1) & \cos(\theta_1) & \sin(\theta_2) & \cos(\theta_2) & \tau_1 & \tau_2
\end{bmatrix}
$$

Tidy data tabulation example whitout angular velocities:

$$
X_{\text{original}} =
\begin{bmatrix}
\text{episode} & t & \theta_1 & \theta_2 & \omega_1 & \omega_2 & \tau_1 & \tau_2 & E_k & E_p \\
0 & 0.002 & -0.191800 & -0.064606 & -0.553969 & 0.388218 & 2.022249 & 0.619492 & 0.228798 & 0.0 \\
0 & 0.004 & -0.192915 & -0.063828 & -0.560330 & 0.389365 & 2.031079 & 0.620264 & 0.232788 & 0.0 \\
0 & 0.006 & -0.194042 & -0.063048 & -0.566779 & 0.390707 & 2.040025 & 0.621060 & 0.236945 & 0.0 \\
0 & 0.008 & -0.195182 & -0.062265 & -0.573315 & 0.392243 & 2.049086 & 0.621881 & 0.241272 & 0.0 \\
0 & 0.010 & -0.196335 & -0.061479 & -0.579940 & 0.393976 & 2.058263 & 0.622725 & 0.245773 & 0.0 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}
$$

To complete variables analyses with angular velocity we have the Tidy below.

$$
X_{\text{tidy}} =
\begin{bmatrix}
\sin(\theta_1) & \cos(\theta_1) & \sin(\theta_2) & \cos(\theta_2) & \omega_1 & \omega_2 & \tau_1 & \tau_2
\end{bmatrix}
$$

$$
X_{\text{tidy}} =
\begin{bmatrix}
\text{episode} & t & \sin\theta_1 & \cos\theta_1 & \sin\theta_2 & \cos\theta_2 & \omega_1 & \omega_2 & \tau_1 & \tau_2 \\
0 & 0.002 & -0.190627 & 0.981663 & -0.064561 & 0.997914 & -0.553969 & 0.388218 & 2.022249 & 0.619492 \\
0 & 0.004 & -0.191720 & 0.981450 & -0.063785 & 0.997964 & -0.560330 & 0.389365 & 2.031079 & 0.620264 \\
0 & 0.006 & -0.192826 & 0.981233 & -0.063006 & 0.998013 & -0.566779 & 0.390707 & 2.040025 & 0.621060 \\
0 & 0.008 & -0.193945 & 0.981012 & -0.062225 & 0.998062 & -0.573315 & 0.392243 & 2.049086 & 0.621881 \\
0 & 0.010 & -0.195076 & 0.980788 & -0.061440 & 0.998111 & -0.579940 & 0.393976 & 2.058263 & 0.622725 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
\end{bmatrix}
$$

This representation provides a continuous and numerically stable encoding of angular states, while maintaining a format suitable for both exploratory data analysis (EDA) and data-driven modeling.

---

## Importance for Computational Analysis

This transformation enables:

### 1. Geometric Consistency

Eliminates discontinuities in angular representation

### 2. Numerical Stability

Avoids abrupt transitions near $\pm\pi$

### 3. Compatibility with Learning Algorithms

Provides a continuous embedding suitable for:

* regression
* clustering
* neural networks

### 4. Enhanced Visualization

Allows phase-space representation without singularities

---

# Part III — Exploratory Data Analysis (EDA) and Nonlinear Dynamics Exploration

# Nonlinear Dynamical Analysis of a Double Inverted Pendulum

## 1. Overview

This work presents a comprehensive nonlinear analysis pipeline applied to a **double inverted pendulum system**, focusing on:

- Stability assessment via **Lyapunov functions**
- Estimation of **Lyapunov exponents**
- Phase space reconstruction using **Takens' embedding theorem**
- Temporal dependency analysis through **Mutual Information**
- Characterization of chaotic behavior via **Finite-Time Lyapunov Exponents (FTLE)**

The methodology operates over multiple simulation episodes, extracting both local and global dynamical properties.

---

## 2. Data Structure and Preprocessing

The dataset consists of time-series measurements of the system states:

- Angular representations:
  - $\sin(\theta_1)$, $\cos(\theta_1)$
  - $\sin(\theta_2)$, $\cos(\theta_2)$
- Angular velocities:
  - $\omega_1$, $\omega_2$
- Time: $t$

### 2.1 Angle Reconstruction

To recover the angular variable:

$$
\theta_1 = \text{atan2}(\sin(\theta_1), \cos(\theta_1))
$$

A normalization step is applied:

$$
\theta_1^{norm} = \frac{\theta_1 - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are the mean and standard deviation, respectively.

---

## 3. Lyapunov-Based Energy Analysis

A candidate **Lyapunov function** is defined as:

$$
V(t) = \frac{1}{2}(\omega_1^2 + \omega_2^2) + k_1 (1 - \cos(\theta_1)) + k_2 (1 - \cos(\theta_2))
$$

This formulation combines:

- Kinetic energy
- Potential energy-like terms derived from angular displacement

### 3.1 Time Derivative

The temporal derivative is approximated numerically:

$$
\dot{V}(t) = \frac{dV}{dt}
$$

This allows assessing system stability:
- $\dot{V}(t) < 0$ → stable behavior
- $\dot{V}(t) > 0$ → energy growth / instability

---

## 4. Mutual Information and Time Delay Selection

To reconstruct the phase space, an appropriate time delay $\tau$ is required.

### 4.1 Mutual Information

The mutual information between $x(t)$ and $x(t+\tau)$ is computed:

$$
I(\tau) = \sum_{i,j} P_{ij} \log \left( \frac{P_{ij}}{P_i P_j} \right)
$$

The optimal delay $\tau$ is selected as:

- The **first local minimum**, or
- The **global minimum**, if no local minimum is found

---

## 5. Phase Space Reconstruction (Takens Embedding)

Using Takens' theorem, the system is embedded in a higher-dimensional space:

$$
\mathbf{x}(t) =
\begin{bmatrix}
x(t) \\
x(t+\tau) \\
x(t+2\tau)
\end{bmatrix}
$$

This enables reconstruction of the **attractor geometry** from scalar observations.

---

## 6. State Space Representation

The full system state is defined as:

$$
\mathbf{s}(t) =
\begin{bmatrix}
\cos(\theta_1), \sin(\theta_1),
\cos(\theta_2), \sin(\theta_2),
\omega_1, \omega_2
\end{bmatrix}
$$

This representation ensures:

- Continuity
- Preservation of angular periodicity
- Suitability for nonlinear analysis

---

## 7. Lyapunov Exponent Estimation

### 7.1 Trajectory Divergence

For nearby trajectories:

$$
d(k) = \| \mathbf{s}_i(k) - \mathbf{s}_j(k) \|
$$

The logarithmic divergence is accumulated:

$$
\ln d(k)
$$

---

### 7.2 Finite-Time Lyapunov Exponent (FTLE)

The FTLE is defined as:

$$
\lambda(k) = \frac{\ln d(k)}{k}
$$

This provides a time-dependent estimate of system sensitivity.

---

## 8. Linear Region Identification

To estimate the **largest Lyapunov exponent**, a linear fit is applied:

$$
\ln d(k) \approx \lambda k + b
$$

The optimal region is selected by maximizing the coefficient of determination:

$$
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
$$

The slope $\lambda$ corresponds to the **Lyapunov exponent**.

---

## 9. Interpretation of Results

- $\lambda > 0$ → Chaotic dynamics (sensitive dependence on initial conditions)
- $\lambda = 0$ → Marginal stability
- $\lambda < 0$ → Stable system

---

## 10. Output Structure

The pipeline generates:

- Lyapunov function plots: $V(t)$ and $\dot{V}(t)$
- Mutual Information curves
- Takens attractors (3D)
- FTLE evolution
- Log-divergence plots with linear fit

Each result is organized per episode in structured directories.

---

## 11. Global Statistical Analysis

Across all episodes:

$$
\bar{\lambda} = \frac{1}{N} \sum_{i=1}^{N} \lambda_i
$$

$$
\sigma_\lambda = \sqrt{\frac{1}{N} \sum (\lambda_i - \bar{\lambda})^2}
$$

These metrics provide a **global characterization of system stability**.

---

## 12. Conclusion

This pipeline integrates:

- Nonlinear dynamics
- Information theory
- Stability theory
- Chaos analysis

to provide a robust framework for analyzing and characterizing the behavior of a double inverted pendulum system.

The approach is particularly suitable for:

- Reinforcement Learning validation
- Control system design
- Detection of chaotic regimes

---

## Final Remarks

This project adopts a **data-driven perspective**, where:

* the system is treated as a **black-box nonlinear generator**
* emphasis is placed on **data structure, transformation, and analysis**
* no explicit solution of the governing equations is required

This approach is particularly suited for:

* machine learning
* system identification
* nonlinear time series analysis
* data-driven control strategies


# Future Work

Future developments of this project may include:

* Implementation of **state-space models**
* Development of **advanced controllers**
* Machine learning approaches for system identification
* Reinforcement learning for stabilization
* Real-time control implementation

---

# References

The double inverted pendulum has been widely studied in the literature and is considered one of the most challenging benchmark systems in nonlinear control theory.
