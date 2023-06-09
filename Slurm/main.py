#! /usr/bin/env python
import os
import argparse
import time
from utilities import is_dir, is_file, str_to_bool, run_python_slurm_job, get_filename, remove_slurm_files
import parameters


def run_instances(instance_dir, solution_dir, results_dir, outfile_dir, print_stats):
    """
    The main function for issuing all individual solve calls.
    Args:
        instance_dir (dir): The directory containing all instances
        solution_dir (dir): The directory containing all solution files
        results_dir (dir): The directory where all result files will be dumped
        outfile_dir (dir): The directory where all out files will be dumped
        print_stats (bool): Whether a stats file should be output from each solve call
    Returns:
        Nothing at all. This is the main function call and it will just produce all appropriate run files
    """

    # Get all instance files
    instance_files = sorted(os.listdir(instance_dir))
    instances = [instance_file.split('.mps')[0] for instance_file in instance_files]

    # Initialise dummy values for all gomory branching rule options
    efficacy = 1.0
    obj_parallelism = 0.0

    # Iterate over all branching options
    for branch_option in parameters.BRANCHING_OPTIONS:

        # Get the avg gmi efficacy weight from the branching option name
        if '_' in branch_option and 'relpscost' in branch_option:
            weight = branch_option.split('_')[1]
            avg_gmi_eff_weight = int(weight) / (10 ** len(weight))
        else:
            avg_gmi_eff_weight = 0.0

        # Create directory specific results
        branch_outfile_dir = os.path.join(outfile_dir, branch_option)
        branch_results_dir = os.path.join(results_dir, branch_option)
        for branch_dir in [branch_outfile_dir, branch_results_dir]:
            if os.path.isdir(branch_dir):
                remove_slurm_files(branch_dir)
            else:
                os.mkdir(branch_dir)

        # Iterate over all random seed and permutation seed combinations
        for permutation_seed in parameters.PERMUTATION_SEEDS:
            # Initialise a list of all slurm_ids
            slurm_job_ids = []
            for rand_seed in parameters.RANDOM_SEEDS:
                # Say what runs are being started
                print('Starting runs for branching rule {} p-seed {} and r-seed {}!'.format(
                    branch_option, permutation_seed, rand_seed), flush=True)
                # Iterate over all instances
                for i, instance in enumerate(instances):
                    # Create the instance path from the file and directory combination
                    instance_path = os.path.join(instance_dir, instance_files[i])
                    # If there are provided solutions, then check if they exists them and add them
                    if solution_dir is not None:
                        solution_path = os.path.join(solution_dir, instance + '.sol.gz')
                        assert os.path.isfile(solution_path)
                    else:
                        solution_path = 'None'
                    # Call the solve_instance.py run and append the returned slurm job id
                    ji = run_python_slurm_job(python_file='Slurm/solve_instance.py',
                                              job_name='{}--{}--{}--{}'.format(instance, branch_option, rand_seed,
                                                                               permutation_seed),
                                              outfile=os.path.join(branch_outfile_dir, '%j__{}__{}__{}__{}.out'.format(
                                                  instance, branch_option, rand_seed, permutation_seed)),
                                              time_limit=17500,
                                              arg_list=[branch_results_dir, instance_path, instance, rand_seed,
                                                        permutation_seed, parameters.TIME_LIMIT, print_stats,
                                                        solution_path, branch_option, efficacy, obj_parallelism,
                                                        avg_gmi_eff_weight]
                                              )
                    slurm_job_ids.append(ji)
                    time.sleep(0.01)
            # Now submit the checker job that has dependencies slurm_job_ids
            safety_file_root = os.path.join(branch_outfile_dir, '{}'.format(permutation_seed))
            _ = run_python_slurm_job(python_file='Slurm/safety_check.py',
                                     job_name='cleaner',
                                     outfile=safety_file_root + '.out',
                                     time_limit=10,
                                     arg_list=[safety_file_root + '.txt'],
                                     dependencies=slurm_job_ids)
            # Put the program to sleep until all of slurm jobs are complete
            time.sleep(10)
            while not os.path.isfile(safety_file_root + '.txt'):
                time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=is_dir)
    parser.add_argument('solution_dir', type=str)
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('outfile_dir', type=is_dir)
    parser.add_argument('print_stats', type=str_to_bool)
    args = parser.parse_args()

    # Change the solution directory to None if not provided
    if args.solution_dir != 'None':
        assert os.path.isdir(args.solution_dir)
    else:
        args.solution_dir = None

    run_instances(args.instance_dir, args.solution_dir, args.results_dir, args.outfile_dir, args.print_stats)
