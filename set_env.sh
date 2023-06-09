#! /usr/bin/env bash

# activate the virtual environment
source venv/bin/activate

# add the python path
export PYTHONPATH=$(pwd):$PYTHONPATH
