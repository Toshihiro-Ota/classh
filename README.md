# Phases in class H

This repo provides a Python implementation of the phase-diagram solver presented in the paper "Phases in a class of associative memories via hidden neurons", [arXiv:2609.10976](https://arxiv.org/abs/2609.10976).

## Abstract

Associative memory in the Hopfield network is attractor dynamics in a disordered many-body system, and higher-order and exponential extensions turn its retrieval update into softmax attention.
The polynomial and exponential regimes have been analyzed by different methods, with no common architecture in which to ask what fixes the storage scale.
In this paper we study the bipartite architecture of Krotov and Hopfield, which we call the class $\mathcal{H}$, whose model is fixed by a Lagrangian for each layer, taking the hidden neurons as the order parameter of retrieval.
At polynomial load the replica method yields the replica-symmetric phase diagrams and closed-form capacities, and the crosstalk moment is common to Ising and spherical visible neurons, so their differences come from the visible entropy.
With a softmax hidden layer the load is exponential, and a copy representation maps the thermodynamics onto random-energy-model counting, with paramagnetic, condensed, and frozen phases.
Heating destabilizes retrieval by quantized reassignments of attention, and typical Gaussian patterns remain metastable at every load.
The regimes differ in their crosstalk statistics, central-limit at polynomial load and large-deviation at exponential load, and the class $\mathcal{H}$ splits retrieval into two roles, the visible Lagrangian fixing stability and the hidden one the storage scale, two axes that may also guide the design of new Lagrangians.

## Usage

### Requirements

We use the basic Python libraries. The required packages can be installed with:

```bash
pip install -r requirements.txt
```

### Example

To compute the phase diagram for a model, e.g. Model A, run the following:

```bash
python phase_a.py
```

The phase diagrams shown in the paper are obtained by running the command above with the default settings.

## Citation

If you use this code or find our work useful, please cite the accompanying paper:

```bibtex
@article{ota2026phases,
    title     = {Phases in a class of associative memories via hidden neurons},
    author    = {Ota, Toshihiro and Taki, Masato},
    journal   = {arXiv preprint arXiv:2609.10976},
    year      = {2026}
}
```
