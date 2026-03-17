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

To complete variables analyses with angular velocity we have the Tidy below.

$$
X_{\text{tidy}} =
\begin{bmatrix}
\sin(\theta_1) & \cos(\theta_1) & \sin(\theta_2) & \cos(\theta_2) & \omega_1 & \omega_2 & \tau_1 & \tau_2
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

## 1. Correlation Matrix

The correlation coefficient is defined as:

$$
\rho_{ij} =
\frac{\mathrm{Cov}(X_i, X_j)}{\sigma_i \sigma_j}
$$

Used to identify **linear dependencies** between:

* angular components
* trigonometric representations
* dynamic torques

---

## 2. Phase Space Visualization

The projection:

$$
(\sin\theta_1, \cos\theta_1)
$$

represents motion on a circular manifold, enabling:

* detection of periodicity
* identification of irregular trajectories

---

## 3. Density Estimation

The joint distribution:

$$
p(\sin\theta_1, \sin\theta_2)
$$

is approximated via 2D histograms, revealing:

* frequently visited regions
* attractor structures
* state concentration zones

---

## 4. Autocorrelation Function

The discrete autocorrelation is given by:

$$
R(k) =
\frac{\sum_{t=1}^{N-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}
{\sum_{t=1}^{N} (x_t - \bar{x})^2}
$$

This quantifies:

* temporal dependence
* memory effects
* characteristic time scales

---

## 5. Cross-Correlation Between States

$$
R_{xy}(k) = \sum_t x(t)y(t+k)
$$

Used to evaluate:

* coupling between $\tau_1$ and $\tau_2$
* interaction between system components

---

## 6. Delay Embedding (Takens Reconstruction)

The time-delay embedding is defined as:

$$
\mathbf{y}(t) =
\begin{bmatrix}
x(t) \
x(t+\tau) \
x(t+2\tau)
\end{bmatrix}
$$

genui{"math_block_widget_always_prefetch_v2":{"content":"\mathbf{y}(t) = \begin{bmatrix} x(t) \\ x(t+\tau) \\ x(t+2\tau) \end{bmatrix}"}}

This reconstructs the system’s attractor from a single observable.

---

## 7. Dynamic Mapping

The discrete mapping:

$$
x(t+1) = F(x(t))
$$

genui{"math_block_widget_always_prefetch_v2":{"content":"x(t+1) = F(x(t))"}}

provides a direct visualization of:

* determinism vs randomness
* nonlinear structure
* chaotic signatures

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
