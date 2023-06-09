import os
import argparse
from pyscipopt import Model


def check_solutions(instance_dir, solution_dir):
    # Get all the instances we downloaded from the MIPLIB 2017 website
    instance_files = os.listdir(instance_dir)
    solution_files = os.listdir(solution_dir)
    instances = set([instance.split('.mps')[0] for instance in instance_files])

    # Check is there is a matching solution for each instance
    for instance in list(instances):
        if instance + '.sol.gz' not in solution_files:
            print('Instance {} has no solution file'.format(instance))
            instances.remove(instance)

    instances = sorted(list(instances))

    # Start a list that will contain all instances with invalid MIPLIB solutions
    broken_solution_instances = []

    # Now build a model for each solution and try the solution
    for instance in instances:
        instance_path = os.path.join(instance_dir, instance + '.mps.gz')
        solution_path = os.path.join(solution_dir, instance + '.sol.gz')
        scip = Model()
        scip.readProblem(instance_path)
        try:
            solution = scip.readSolFile(solution_path)
            valid_solution = scip.checkSol(solution, completely=True)
        except:
            scip.freeProb()
            broken_solution_instances.append(instance)
            continue
        scip.freeSol(solution)
        scip.freeProb()
        if not valid_solution:
            # Remove instance and solution we have in storage
            broken_solution_instances.append(instance)

    print('{} many instances had invalid MIPLIB solutions. Complete list: {}'.format(len(broken_solution_instances),
                                                                                     broken_solution_instances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=str)
    parser.add_argument('solution_dir', type=str)
    args = parser.parse_args()

    assert os.path.isdir(args.instance_dir)
    assert os.path.isdir(args.solution_dir)

    check_solutions(args.instance_dir, args.solution_dir)
