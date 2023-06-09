"""File containing the settings for the experiments you want to perform.
Each parameter here affects different bits of the experiments. The individual comments outline how.
"""

# Make sure to set this before you begin any runs. This is the SLURM node list that'll be used
# TODO: The user must define this by themselves!!!!!
SLURM_CONSTRAINT = 'Xeon&Gold6342'

# If you want to implement memory limits on the individual slurm jobs (units are megabytes). Set to None to ignore.
# This was set to 3000 in the paper.
SLURM_MEMORY = 3000

# If you want to implement memory limits on the individual slurm jobs (units are megabytes). Set to None to ignore.
# This was set to 2000 in the paper.
SCIP_MEMORY = 2000

# The random seeds used in all experiments
# These were [1,2,3] in the paper
RANDOM_SEEDS = [1, 2, 3, 4, 5]

# The permutation seeds used in all experiments
# These were [0] in the paper
PERMUTATION_SEEDS = [0]

# The time limit for all runs that were used. Time in seconds.
# This was set to 7200 in the paper
TIME_LIMIT = 7200

# This is the list of branching rules used in the paper
BRANCHING_OPTIONS = ['gomory', 'weakgomory', 'fullstrong', 'relpscost', 'relpscost_01', 'relpscost_001',
                     'relpscost_0001', 'relpscost_00001', 'relpscost_000001', 'random']
