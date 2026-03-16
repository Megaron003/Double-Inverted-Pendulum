# Double Inverted Pendulum

![Python](https://img.shields.io/badge/Python-3.11.9-blue) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

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
