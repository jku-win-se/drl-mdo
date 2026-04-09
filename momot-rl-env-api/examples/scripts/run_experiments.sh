#!/bin/bash --login

source run_slb.sh
source run_cra.sh
source run_ref.sh
source run_scrum.sh

LOGFILE="$(pwd)/$(date '+%Y%m%d_%H%M%S').log"

runs=10
fes=100000

: <<'END'
cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB

cd ../org.cra2/
sudo $RUN_CRA --model "CRA_A" --solution_length 30 --runs 3 --evaluations 5000

cd ../at.ac.tuwien.big.momot.examples.refactoring/
sudo $RUN_REFACTORING --model "SeveralRefactorings" --solution_length 8 --runs 1 --evaluations 2000

cd ../org.scrum/
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-119-items" --solution_length 250 --runs 1 --evaluations 5000



cd ../at.ac.tuwien.big.momot.examples.refactoring/
sudo $RUN_REFACTORING --model "SeveralRefactorings" --solution_length 10 --runs 5 --evaluations 10000 > ../scripts/several_ref.log 2>&1
sudo $RUN_REFACTORING --model "Pullup" --solution_length 20 --runs 5 --evaluations 10000 > ../scripts/pullup_ref.log 2>&1


cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_twentyfive_stacks" --solution_length 80 --runs $runs --evaluations $fes > ../scripts/110925/slb25.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_fifty_stacks" --solution_length 100 --runs $runs --evaluations $fes > ../scripts/110925/slb50.log 2>&1
sudo $RUN_SLB --model "model_hundred_stacks" --solution_length 250 --runs $runs --evaluations $fes > ../scripts/110925/slb100.log 2>&1

cd ../org.cra2/
sudo $RUN_CRA --model "CRA_A" --solution_length 9 --runs $runs --evaluations $fes > ../scripts/110925/cra_a.log 2>&1
sudo $RUN_CRA --model "CRA_B" --solution_length 36 --runs $runs --evaluations $fes > ../scripts/110925/cra_b.log 2>&1
sudo $RUN_CRA --model "CRA_C" --solution_length 80 --runs $runs --evaluations $fes > ../scripts/110925/cra_c.log 2>&1
sudo $RUN_CRA --model "CRA_D" --solution_length 160 --runs $runs --evaluations $fes > ../scripts/110925/cra_d.log 2>&1

cd ../org.scrum/
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-64-items" --solution_length 120 --runs 30 --evaluations $fes > ../scripts/160925/scrum5s64i.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_fifty_stacks" --solution_length 100 --runs $runs --evaluations $fes > ../scripts/110925/slb50.log 2>&1
sudo $RUN_SLB --model "model_hundred_stacks" --solution_length 250 --runs $runs --evaluations $fes > ../scripts/110925/slb100.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_ten_stacks" --solution_length 10 --runs $runs --evaluations $fes > ../scripts/110925/slb25.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_five_stacks" --solution_length 20 --runs $runs --evaluations $fes > ../scripts/110925/slb25.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_twentyfive_stacks" --solution_length 80 --runs $runs --evaluations $fes > ../scripts/210925/slb25.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_fifty_stacks" --solution_length 100 --runs $runs --evaluations $fes > ../scripts/210925/slb50_2.log 2>&1
sudo $RUN_SLB --model "model_hundred_stacks" --solution_length 250 --runs $runs --evaluations $fes > ../scripts/210925/slb100_2.log 2>&1

cd ../org.cra2/
sudo $RUN_CRA --model "CRA_A" --solution_length 9 --runs $runs --evaluations $fes > ../scripts/210925/cra_a.log 2>&1
sudo $RUN_CRA --model "CRA_B" --solution_length 36 --runs $runs --evaluations $fes > ../scripts/210925/cra_b.log 2>&1
sudo $RUN_CRA --model "CRA_C" --solution_length 80 --runs $runs --evaluations $fes > ../scripts/210925/cra_c.log 2>&1
sudo $RUN_CRA --model "CRA_D" --solution_length 160 --runs $runs --evaluations $fes > ../scripts/210925/cra_d.log 2>&1


cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_five_stacks" --solution_length 10 --runs $runs --evaluations $fes > ../scripts/210925/slb5.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_ten_stacks" --solution_length 20 --runs $runs --evaluations $fes > ../scripts/210925/slb10.log 2>&1


cd ../at.ac.tuwien.big.momot.examples.refactoring/
sudo $RUN_REFACTORING --model "Mixed" --solution_length 10 --runs 3 --evaluations 10000 > ../scripts/mixed.log 2>&1

cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_five_stacks" --solution_length 10 --runs $runs --evaluations $fes > ../scripts/250925/slb5.log 2>&1
sudo $RUN_SLB --model "model_ten_stacks" --solution_length 20 --runs $runs --evaluations $fes > ../scripts/250925/slb10.log 2>&1
sudo $RUN_SLB --model "model_twentyfive_stacks" --solution_length 50 --runs $runs --evaluations $fes > ../scripts/250925/slb25.log 2>&1
sudo $RUN_SLB --model "model_fifty_stacks" --solution_length 100 --runs $runs --evaluations $fes > ../scripts/250925/slb50.log 2>&1
sudo $RUN_SLB --model "model_hundred_stacks" --solution_length 250 --runs $runs --evaluations $fes > ../scripts/250925/slb100.log 2>&1

cd ../org.cra2/
sudo $RUN_CRA --model "CRA_A" --solution_length 9 --runs $runs --evaluations $fes > ../scripts/250925/cra_a.log 2>&1
sudo $RUN_CRA --model "CRA_B" --solution_length 36 --runs $runs --evaluations $fes > ../scripts/250925/cra_b.log 2>&1
sudo $RUN_CRA --model "CRA_C" --solution_length 80 --runs $runs --evaluations $fes > ../scripts/250925/cra_c.log 2>&1
sudo $RUN_CRA --model "CRA_D" --solution_length 160 --runs $runs --evaluations $fes > ../scripts/250925/cra_d.log 2>&1

cd ../org.scrum/
sudo $RUN_SCRUM --model "sprint-planning-model-3-stakeholders-24-items" --solution_length 40 --runs 30 --evaluations $fes > ../scripts/291025/scrum-24i.log 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-4-stakeholders-42-items" --solution_length 80 --runs 30 --evaluations $fes > ../scripts/291025/scrum-42i.log 2>&1

sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-64-items" --solution_length 120 --runs 30 --evaluations $fes > ../scripts/291025/scrum-64i.log 2>&1


sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-64-items" --solution_length 120 --runs 10 --evaluations $fes > ../scripts/311025/scrum-64i.log 2>&1

sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-119-items" --solution_length 220 --runs 10 --evaluations $fes > ../scripts/311025/scrum-119i.log 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-7-stakeholders-190-items" --solution_length 350 --runs 10 --evaluations $fes > ../scripts/311025/scrum-190i.log 2>&1

cd ../org.scrum/
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-64-items" --solution_length 120 --runs 10 --evaluations $fes > "$LOGPATH/scrum-64i-test.log" 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-119-items" --solution_length 200 --runs 10 --evaluations $fes > "$LOGPATH/scrum-119i-test.log" 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-7-stakeholders-190-items" --solution_length 300 --runs 10 --evaluations $fes > "$LOGPATH/scrum-190i-test.log" 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-4-stakeholders-42-items" --solution_length 80 --runs 10 --evaluations $fes > "$LOGPATH/scrum-42i-test.log" 2>&1

cd ../org.cra2/
sudo $RUN_CRA --model "CRA_E" --solution_length 320 --runs 10 --evaluations $fes > ../scripts/250925/cra_e.log 2>&1

sudo $RUN_SCRUM --model "sprint-planning-model-7-stakeholders-190-items" --solution_length 300 --runs 10 --evaluations $fes > "$LOGPATH/scrum-190i-test.log" 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-4-stakeholders-42-items" --solution_length 100 --runs 10 --evaluations $fes > "$LOGPATH/scrum-42i-test.log" 2>&1

LOGPATH="../scripts/151225"

cd ../org.scrum/
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-64-items" --solution_length 150 --runs 10 --evaluations $fes > "$LOGPATH/scrum-64i-_qlearn_64_119.log" 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-119-items" --solution_length 300 --runs 10 --evaluations $fes > "$LOGPATH/scrum-119i-_qlearn_64_119.log" 2>&1
cd ../org.cra2/
sudo $RUN_CRA --model "CRA_B" --solution_length 36 --runs $runs --evaluations $fes > ../scripts/070126/cra_b-2.log 2>&1
sudo $RUN_CRA --model "CRA_C" --solution_length 80 --runs $runs --evaluations $fes > ../scripts/070126/cra_c-2.log 2>&1
sudo $RUN_CRA --model "CRA_D" --solution_length 160 --runs $runs --evaluations $fes > ../scripts/070126/cra_d-2.log 2>&1


cd ../org.cra2/
sudo $RUN_CRA --model "CRA_E" --solution_length 320 --runs 30 --evaluations $fes > ../scripts/060326/cra_e.log 2>&1

cd ../org.scrum/
sudo $RUN_SCRUM --model "sprint-planning-model-4-stakeholders-42-items" --solution_length 96 --runs 10 --evaluations $fes > ../scripts/09_constrained/scrum-4s42i.log 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-64-items" --solution_length 147 --runs 10 --evaluations $fes > ../scripts/09_constrained/scrum-5s64i.log 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-5-stakeholders-119-items" --solution_length 273 --runs 10 --evaluations $fes > ../scripts/09_constrained/scrum-5s119i.log 2>&1
sudo $RUN_SCRUM --model "sprint-planning-model-6-stakeholders-124-items" --solution_length 285 --runs 10 --evaluations $fes > ../scripts/09_constrained/scrum-5s124i.log 2>&1

END






cd ../at.ac.tuwien.big.momot.examples.stack/
sudo $RUN_SLB --model "model_ten_stacks" --solution_length 20 --runs 30 --evaluations $fes > ../scripts/1203-2/slb10.log 2>&1
sudo $RUN_SLB --model "model_twentyfive_stacks" --solution_length 50 --runs 30 --evaluations $fes > ../scripts/1203-2/slb25.log 2>&1
sudo $RUN_SLB --model "model_fifty_stacks" --solution_length 100 --runs 30 --evaluations $fes > ../scripts/1203-2/slb50.log 2>&1
sudo $RUN_SLB --model "model_hundred_stacks" --solution_length 250 --runs 30 --evaluations $fes > ../scripts/1203-2/slb100.log 2>&1
