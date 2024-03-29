#! /usr/bin/env python
import os
import argparse
import yaml
from utilities import is_dir, is_file, build_scip_model, get_filename, str_to_bool
import parameters


def run_instance(results_dir, instance_path, instance, rand_seed, permutation_seed, time_limit, print_stats,
                 solution_path=None, branch_option='gomory', efficacy=1.0, obj_parallelism=0.0,
                 avg_gmi_eff_weight=0.0, avg_instead_of_last_gmi=False):
    """
    The call to solve a single instance. A model will be created and an instance file (and potentially solution file)
    loaded in. Appropriate settings as defined by this function call are then set and the model solved.
    After the model is solved, found infeasible, or some time limit hit, information is extracted and put into
    a yml file. All calls to solve an instance should go through this function and 'run_python_slurm_job' in
    utilities.py.
    Args:
        results_dir: The directory in which all all result files will be stored
        instance_path: The path to the MIP instance
        instance: The instance base name of the MIP file
        rand_seed: The random seed which will be used to shift all SCIP randomisation
        permutation_seed: The random seed which will be used to permute the problem before solving
        time_limit: The time limit, if it exists for our SCIP instance (in seconds).
        print_stats: Whether the .stats file from the run should be printed or not
        solution_path: The path to the solution file which will be loaded
        branch_option: The type of branching rule used. Please see parameters.py for a complete list
        efficacy: Efficacy weight
        obj_parallelism: Objective parallelism weight
        avg_gmi_eff_weight: The weight given to gmiavgeffweight in the relpscost branching rule
        avg_instead_of_last_gmi: Whether average efficacy of computed GMI cuts should be used instead of just the last computed cut
    Returns:
        Nothing. All results from this run should be output to a file in results_dir.
        The results should contain all information about the run, (e.g. solve_time, dual_bound etc)
    """

    # Make sure the input is of the right type
    assert type(time_limit) == int and time_limit > 0
    assert is_dir(results_dir)
    assert is_file(instance_path)
    assert instance == os.path.split(instance_path)[-1].split('.mps')[0]
    assert type(rand_seed) == int and rand_seed >= 0
    assert isinstance(print_stats, bool)
    if solution_path is not None:
        assert is_file(solution_path) and instance == os.path.split(solution_path)[-1].split('.sol')[0]
    assert branch_option in parameters.BRANCHING_OPTIONS

    # Set the time limit if None is provided.
    time_limit = None if time_limit < 0 else time_limit
    node_lim = -1

    # Build the actual SCIP model from the information now
    scip = build_scip_model(instance_path, node_lim, rand_seed, True, True, True, True,
                            permutation_seed, time_limit=time_limit, sol_path=solution_path,
                            branch_option=branch_option, efficacy=efficacy,
                            obj_parallelism=obj_parallelism, avg_gmi_eff_weight=avg_gmi_eff_weight,
                            avg_instead_of_last_gmi=avg_instead_of_last_gmi)

    # Solve the SCIP model and extract all solve information
    solve_model_and_extract_solve_info(scip, rand_seed, permutation_seed, instance, results_dir,
                                       print_stats=print_stats, branch_option=branch_option)

    # Free the SCIP instance
    scip.freeProb()

    return


def solve_model_and_extract_solve_info(scip, rand_seed, permutation_seed, instance, results_dir,
                                       print_stats=False, branch_option='gomory'):
    """
    Solves the given SCIP model and after solving creates a YML file with all potentially interesting
    solve information. This information will later be read and used to update the neural_network parameters
    Args:
        scip: The PySCIPOpt model that we want to solve
        rand_seed: The random seed used in the scip parameter settings
        permutation_seed: The random seed used to permute the problems rows and columns before solving
        instance: The instance base name of our problem
        results_dir: The directory in which all all result files will be stored
        print_stats: A kwarg that informs if the .stats file from the run should be saved
        branch_option: The type of branching rule used. Please see parameters.py for a complete list
    Returns:
        Nothing. A YML results file is created
    """

    # Solve the MIP instance. All parameters should be pre-set
    scip.optimize()

    # Initialise the dictionary that will store our solve information
    data = {}

    # Get the solve_time
    data['solve_time'] = scip.getSolvingTime()
    # Get the number of cuts applied
    data['num_cuts'] = scip.getNCutsApplied()
    # Get the number of nodes in our branch and bound tree
    data['num_nodes'] = scip.getNTotalNodes()
    # Get the best primal solution if available
    data['primal_bound'] = scip.getObjVal() if len(scip.getSols()) > 0 else 1e+20
    # Get the gap provided a primal solution exists
    data['gap'] = scip.getGap() if len(scip.getSols()) > 0 else 1e+20
    # Get the best dual bound
    data['dual_bound'] = scip.getDualbound()
    # Get the number of LP iterations
    data['num_lp_iterations'] = scip.getNLPIterations()
    # Get the status of the solve
    data['status'] = scip.getStatus()
    # Get the primal-dual difference
    data['primal_dual_difference'] = data['primal_bound'] - data['dual_bound'] if len(scip.getSols()) > 0 else 1e+20
    # Get the number of separation rounds
    data['num_sepa_rounds'] = scip.getNSepaRounds()
    # Get the branching option for the experiment
    data['branching_option'] = branch_option

    # Add the cut-selector parameters
    if 'gomory' in branch_option:
        data['efficacy'] = scip.getParam('branching/gomory/efficacyweight')
        data['obj_parallelism'] = scip.getParam('branching/gomory/objparallelweight')
        data['useweakercuts'] = scip.getParam('branching/gomory/useweakercuts')
        data['performrelpscost'] = scip.getParam('branching/gomory/performrelpscost')
    if 'relpscost' in branch_option:
        data['gmiavgeffweight'] = scip.getParam('branching/relpscost/gmiavgeffweight')

    # Get the primal dual integral.
    # It is only accessible through the solver statistics. TODO: Write a wrapper function for this
    stat_file = get_filename(results_dir, instance, rand_seed, permutation_seed=permutation_seed, ext='stats')
    assert not os.path.isfile(stat_file)
    scip.writeStatistics(filename=stat_file)
    if data['status'] in ['optimal', 'timelimit', 'nodelimit', 'memlimit']:
        with open(stat_file) as s:
            stats = s.readlines()
        # TODO: Make this safer to access.
        for line_i, line in enumerate(stats):
            if 'primal-dual' in line:
                data['primal_dual_integral'] = float(line.split(':')[1].split('     ')[1])
            if 'First LP value' in line:
                data['initial_dual_bound'] = float(line.split(':')[-1].split('\n')[0])
            if 'number of runs' in line:
                data['num_runs'] = float(line.split(':')[1].strip(' '))
            if 'Branching Rules' in line:
                if 'relpscost' in branch_option:
                    data['branch_time'] = float(stats[line_i + 14].split(':')[1].strip().split(' ')[0])
                if 'gomory' in branch_option:
                    data['branch_time'] = float(stats[line_i + 5].split(':')[1].strip().split(' ')[0])
                if 'random' in branch_option:
                    data['branch_time'] = float(stats[line_i + 13].split(':')[1].strip().split(' ')[0])
                if 'fullstrong' in branch_option:
                    data['branch_time'] = float(stats[line_i + 4].split(':')[1].strip().split(' ')[0])

    # If we haven't asked to save the file, then remove it.
    if not print_stats:
        os.remove(stat_file)

    # Dump the yml file containing all of our solve info into the right place
    yml_file = get_filename(results_dir, instance, rand_seed=rand_seed, permutation_seed=permutation_seed, ext='yml')
    with open(yml_file, 'w') as s:
        yaml.dump(data, s)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=is_dir)
    parser.add_argument('instance_path', type=is_file)
    parser.add_argument('instance', type=str)
    parser.add_argument('rand_seed', type=int)
    parser.add_argument('permutation_seed', type=int)
    parser.add_argument('time_limit', type=int)
    parser.add_argument('print_stats', type=str_to_bool)
    parser.add_argument('solution_path', type=str)
    parser.add_argument('branch_option', type=str)
    parser.add_argument('efficacy', type=float)
    parser.add_argument('obj_parallelism', type=float)
    parser.add_argument('avg_gmi_eff_weight', type=float)
    parser.add_argument('avg_instead_of_last_gmi', type=str_to_bool)
    args = parser.parse_args()

    # Check if the solution file exists
    if args.solution_path == 'None':
        args.solution_path = None
    else:
        assert os.path.isfile(args.solution_path)

    # The main function call to run a SCIP instance with cut-sel params
    run_instance(args.results_dir, args.instance_path, args.instance, args.rand_seed, args.permutation_seed,
                 args.time_limit, args.print_stats, args.solution_path, args.branch_option, args.efficacy,
                 args.obj_parallelism, args.avg_gmi_eff_weight, args.avg_instead_of_last_gmi)
