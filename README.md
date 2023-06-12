# Branching via Cutting Plane Selection 

If this software was used for academic purposes, please cite our paper with the below information:

`@article{turner2023branching,
  title={Branching via Cutting Plane Selection: Improving Hybrid Branching},
  author={Turner, Mark and Berthold, Timo and Besan{\c{c}}on, Mathieu and Koch, Thorsten},
  journal={arXiv preprint arXiv:2306.06050},
  year={2023}
}`

If the instances from ```generate_snd_lib_instances``` were used, then please cite with below information:

`@misc{zenodo_sndlib,
  author={Turner, Mark and Berthold, Timo and Besançon, Mathieu and Koch, Thorsten},
  title={{SNDlib-MIPs: A new set of homogeneous MILP instances}},
  month=jun,
  year=2023,
  publisher={Zenodo},
  doi={10.5281/zenodo.8021237},
  url={https://doi.org/10.5281/zenodo.8021237}
}`

`@article{sndlib,
  title={SNDlib 1.0—Survivable network design library},
  author={Orlowski, Sebastian and Wess{\"a}ly, Roland and Pi{\'o}ro, Michal and Tomaszewski, Artur},
  journal={Networks: An International Journal},
  volume={55},
  number={3},
  pages={276--286},
  year={2010},
  publisher={Wiley Online Library}
}`

## Install Guide
Requirements: Python 3.9 / Debian 11 (Also tested with Python 3.8 and 3.11 and Ubuntu 20.04).
We use SLURM https://slurm.schedmd.com/overview.html as a job manager. 
All calls go through a central function however, and in theory SLURM could be replaced by python's 
default multiprocessing package.

Run the bash script init_venv. If you don't use bash, configure the shebang (first line of the script) 
to be your shell interpreter.

`./init_venv`

After installing the virtual environment, make sure to always activate it with the script printed beneath. 
This is so your python path is appended and files at different directory levels can import from each other.

`source ./set_venv`

Now go and install SCIP from (https://www.scipopt.org/index.php#download / https://github.com/scipopt/scip)
For Ubuntu / Debian, the .sh installer is the easiest choice if you don't want to configure it yourself). 
The latest release version of SCIP is required for this project, or the development branch `mt/branch_gomory`.

You can test if SCIP is installed by locating /bin/scip and calling it through the command line. 
SCIP should hopefully open.

One then needs to install PySCIPOpt https://github.com/scipopt/PySCIPOpt. 
I would recommend following the INSTALL.md guide. Make sure to have set your environment variable pointing to SCIP! 
You can test if this has been properly installed by running one of the tests, or by trying to import Model. 
This research was done using PySCIPOpt 4.2.0. 

How to run the software
We use Nohup https://en.wikipedia.org/wiki/Nohup to run all of our jobs and to capture output 
of the main function calls. It also allows jobs to be started through a SHH connection. 
Using this is not necessary however, so feel free to use whatever software you prefer. 
An example call to redirect output to nohup/nohup.out and to run the process in the background would be

`nohup python dir/example.py > nohup/nohup.out &`

## Instance Guide

- Download instances (and potentially solutions) that you want to perform experiments on

We downloaded the MIPLIB 2017 benchmark set from https://miplib.zib.de/
We then created a script for downloading the best known solution for every instance. To run the script:

`python scripts/download_miplib_solutions.py instance_dir solution_dir`

It is then possible to check the solutions you've downloaded and if they are feasible for SCIP using

`python scripts/check_miplib_solutions.py instance_dir solution_dir`

Note: This repository is not constrained to MIPLIB2017 instances. One can use any instance set they want. 
Another example instance set, for which we obtained preliminary results, is: `insert here`. 

## Run Guide

- Run experiments using various branching rules

All global user settings are stored in `parameters.py`. For example, the random seeds used, which branching rules,
and the time / memory limits.

After setting the desired parameters, please make sure to edit `run_python_slurm_job` in `utilities.py`.
This is to make sure that your individual user information is there and it runs on your system.

All runs are initiated through the main function in `Slurm/main.py`. An example run is:

`nohup python Slurm/main.py instance_dir solution_dir results_dir outfiles_dir True`

- instance_dir: Where your instances are stored
- solution_dir: Where your .sol files are stored (If you don't want to provide solutions set as None)
- results_dir: Where all results files should be stored. They will automatically be sorted when stored.
- outfiles_dir: Where all log files of SCIP runs should be stored.
- print_stats: Whether you want additional statistic files stored of your runs (recommended)

This script will iterate through all instance, permutation seed, random seed, and branching rule combination.
It will store the results of all runs, from which the user can then analyse the results.

#### Thanks for Reading!! I hope this code helps you with your research. Please feel free to send any issues.