#! /usr/bin/env bash

# install the virtual environment
virtualenv -p python3.9 venv

# activate the virtual environment
source venv/bin/activate

# install everything
pip install -r requirements.txt

deactivate
