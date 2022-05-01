for gymenv in 'walker2d' 'halfcheetah' 'hopper';
  do
    for flag in 'True' 'False';
    do
      python3 scripts/train.py --env $gymenv --dataset medium --device cuda --rtg_sparse_flag $flag
    done
  done
