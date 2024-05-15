# Detecting Influence Structures in Multi-Agent Reinforcement Learning

This repository contains the code for the randomly generated environment used in the paper "Detecting Influence Structures in Multi-Agent Reinforcement Learning" by Fabian R. Pieroth, Katherine Fitch, and Lenz Belzner. The paper is accepted to the International Conference on Machine Learning (ICML), 2024.

The policy learning algorithm is Algorithm 1, from "Fully decentralized multi-agent reinforcement learning with networked agents" by Kaiqing Zhang, Yang Zhuoran, Liu Han, Zhang Tong, and Tamar Basar, published at International Conference on Machine Learning (ICML), 2018.

## Setup

Note: These setup instructions assume a linux-based OS and uses python 3.8.10 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)

`sudo apt-get install virtualenv`

Create a virtual environment with virtual env (you can also choose your own name)

`virtualenv networked-agents-exp`

You can specify the python version for the virtual environment via the --python flag. Note that this version already needs to be installed on the system (e.g. `virtualenv --python=python3 networked-agents-exp` uses the standard python3 version from the system).

activate the environment with

`source ./networked-agents-exp/bin/activate`

Install all requirements

`pip install -r requirements.txt`

## Run the experiments

Create a folder 'local_logs' via

`mkdir local_logs`

The runner.py file contains the different scenarios reported in the paper. Uncomment the ones you want to run and exectute

`python runner.py`

## Evaluating the experiments

Adapt the paths in logger/local_log_management.py to point at the experiments you want to evaluate. Additionally, you need to adapt the plot title, metric names, etc.

Then run

`python logger/local_log_management.py`

