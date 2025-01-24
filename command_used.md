## Question 1.2
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.eval_batch_size=5000 alg.num_agent_train_steps_per_iter=175

## Question 1.3
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.eval_batch_size=5000

## IDM
python run_hw1_bc.py alg.n_iter=1 alg.do_dagger=false alg.train_idm=true alg.eval_batch_size=5000

## DAgger walker2d
python run_hw1_bc.py alg.n_iter=10 alg.do_dagger=true alg.train_idm=false alg.eval_batch_size=5000 alg.num_agent_train_steps_per_iter=175

## DAgger ant
python run_hw1_bc.py alg.n_iter=5 alg.do_dagger=true alg.train_idm=false alg.eval_batch_size=5000 alg.num_agent_train_steps_per_iter=175