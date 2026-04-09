#!/bin/bash --login

source run_slb.sh
source run_cra.sh

LOGFILE="$(pwd)/$(date '+%Y%m%d_%H%M%S').log"

runs=5
fes=100000


