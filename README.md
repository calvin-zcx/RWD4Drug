# RWD-4-AD-Drugs
Using RWDs to generate AD drug repurposing hypotheses

## Test Systems
Windows 10 PC, 16 GB memory, 500 GB hard disk,  6 GB NVIDIA GeForce GTX 1060 GPU
Linux Ubuntu 18.04.2 LTS server, 62 GB memory, 500 GB hard disk, 11 GB GeForce RTX 2080 Ti GPU, and 16 CPU cores. 
Python environment install and activation
Notes: recommend using tmux at the terminal to run all the following commands
git clone https://github.com/calvin-zcx/RWD4Drug.git
cd RWD4Drug/
conda env create -f environment.yml   
conda activate rwd4drug

## Code Structure
1. ipreprocess- EHR preorpcessing  
3. iptw - High-throughput screening of AD by machine learning-based propensity-score reweighting method

shell commands are located in each package to run different functions