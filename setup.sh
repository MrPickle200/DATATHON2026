#!/bin/bash
pip install -r requirements.txt
python -m ipykernel install --user --name datathon --display-name "Python (datathon)"
echo "Done! Select kernel 'Python (datathon)' in Jupyter/VSCode."