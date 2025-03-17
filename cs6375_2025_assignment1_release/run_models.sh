#!/bin/bash


for hidden_dim in 2 4 8 16 32 64
do
  # echo "Running FFNN with hidden_dim=$hidden_dim"
  # python ffnn.py --hidden_dim $hidden_dim --epochs 10 --train_data ./training.json --val_data ./validation.json --test_data ./test.json --do_train
  echo "Running RNN with hidden_dim=$hidden_dim"
  python rnn.py --hidden_dim $hidden_dim --epochs 10 --train_data ./training.json --val_data ./validation.json --test_data ./test.json --do_train
done

echo "---------------- DONE! ----------------"