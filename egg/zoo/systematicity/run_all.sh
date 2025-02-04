#!/bin/bash

for VS in 10
do
	for LEN in 3 4 6 8
	do
		for SEED in 0 1 2 3 4
		do
			python train.py --n_attributes 2 --n_values 16 --vocab_size $VS --max_len $LEN --batch_size 5120 --data_scaler 60 --n_epochs 3000 --random_seed $SEED --sender_hidden 500 --receiver_hidden 500 --sender_entropy_coeff 0.5 --sender_cell gru --receiver_cell gru --lr 0.001 --receiver_emb 30 --sender_emb 5
		done
	done
done
