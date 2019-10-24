#!/bin/bash 

show_one_exp() {
  d=$1
  echo ""; 
  echo "####### $d"; 
  # ls -v $d/*.out | tail -1 | xargs -I{} bash -c "grep 'Train loss' {} | tail -n5 | head -n4"; 
  ls -v $d/*.out | tail -1 | xargs -I{} bash -c "tail -n5 {}"; 
}

for d in ./exps/*; do
  show_one_exp $d
done
