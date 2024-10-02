#!/bin/bash
echo 'pyver=$(python -c "import sys;print(sys.version[:(len(str(sys.version_info.major))+len(str(sys.version_info.minor))+1)])")' >> ~/.bashrc
echo 'source activate matmek4270' >> ~/.bashrc
echo 'export PYTHONPATH=/opt/conda/envs/matmek4270/lib/python${pyver}/site-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
