for model in 'dt' 's4_dt';
  do
  for gymenv in 'walker2d-medium-v2' 'halfcheetah-medium-v2' 'hopper-medium-v2';
    do
      for flag in 'True' 'False';
      do
        python scripts/plot.py --model_name $model --env_d4rl_name $gymenv --rtg_sparse $flag
      done
    done
  done



