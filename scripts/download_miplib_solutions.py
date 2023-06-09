import os
import wget
import argparse
import ssl
import time
import requests


def download_miplib_sols(instance_dir, solution_dir):
    # Workaround for error with SSL certificate verification (Would not use for random websites)
    ssl._create_default_https_context = ssl._create_unverified_context

    instance_files = os.listdir(instance_dir)
    instances = [instance_file.split('.mps')[0] for instance_file in instance_files]

    # Download the instances
    num_no_miplib_solution_instances = 0
    broken_instances = []
    for instance in instances:
        # Assume that there's been no more than 20 solutions submitted per instance
        no_solution_found = True
        for sol_i in range(20, 0, -1):
            sol_url = 'https://miplib.zib.de/downloads/solutions/{}/{}/{}.sol.gz'.format(instance, sol_i, instance)
            time.sleep(0.02)
            headers = requests.head(sol_url).headers
            if not ('text/html' in headers['Content-Type']):
                no_solution_found = False
                break
        if no_solution_found:
            num_no_miplib_solution_instances += 1
            broken_instances.append(instance)
            continue

        sol_file = '{}/{}.sol.gz'.format(solution_dir, instance)
        wget.download(sol_url, sol_file)
        time.sleep(0.1)

    print('Instances {} had no solution'.format(broken_instances))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_dir', type=str)
    parser.add_argument('solution_dir', type=str)
    args = parser.parse_args()

    assert os.path.isdir(args.instance_dir)
    assert os.path.isdir(args.solution_dir)

    download_miplib_sols(args.instance_dir, args.solution_dir)
