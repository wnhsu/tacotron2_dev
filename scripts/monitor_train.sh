#!/bin/bash 

show_one_exp() {
  d=$1
  echo ""; 
  # ls -v $d/*.out | tail -1 | xargs -I{} bash -c "grep 'Train loss' {} | tail -n5 | head -n4"; 
  ls -v $d/*.out | tail -1 | xargs -I{} bash -c "echo '##### {}'"; 
  ls -v $d/*.out | tail -1 | xargs -I{} bash -c "tail -n10 {}"; 
}

for d in ./exps/*; do
  if [[ ! $d =~ .*finished ]] && [[ ! $d =~ .*eval ]]; then
    show_one_exp $d
  fi
done
