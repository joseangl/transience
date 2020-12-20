#!/bin/bash

data_root="./data/M2F"

echo "Training DCTW"
python3 -O train_model.py --view1_win 11 --view1_pca_ncomps 0.999  --noise_std 0.5 --dtw_distance 'cosine' --batch_size 512 --latent_shared_dim 20 --latent_private_dim 10 --epochs_per_iter 10 --learning_rate 1e-4 --dropout 0.0  --network_arch '64#32#32' --similarity 'contrastive' --autoencoder_loss_weight 1.0 dctw_model $data_root/sensor1 $data_root/world2

echo "Aligning the data"
python3 align_sequences.py dctw_model $data_root/sensor $data_root/mfcc $data_root/sensor_aligned $data_root/mfcc_aligned



echo "Done!!"
