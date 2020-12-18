#!/bin/bash

data_root="../data/TiDigits/Jim10Nov"

# echo "Vocoding with WORLD"
# mkdir $data_root/world
# python3 world.py $data_root/wav $data_root/world

# echo "Training the DCTW model"
# # Copy only the MFCC files
# mkdir $data_root/mfcc
# for f in $data_root/world/*.mfcc.npy
# do
#    fbname=$(basename $f .mfcc.npy)
#    cp $data_root/world/$fbname.mfcc.npy $data_root/mfcc/$fbname.npy 
# done

# # Convert the sensor data to numpy format
# python3 htk2npy.py $data_root/sensor htk

# Train the DCTW model

echo "Training DCTW"
python3 train_model.py --view1_win 3 --view1_pca_ncomps 0.995  --noise_std 0.0 --dtw_distance 'cosine' --batch_size 512 --latent_shared_dim 10 --latent_private_dim 7 --epochs_per_iter 10 --learning_rate 1e-4 --dropout 0.1  --network_arch '64#32#32' --similarity 'contrastive' --autoencoder_loss_weight 1.0 dctw_model $data_root/sensor $data_root/mfcc
echo "Aligning the data"
python3 align_sequences.py --plot_dir test_dtw_plots.contrastive.cosine dctw_model $data_root/sensor $data_root/mfcc $data_root/sensor_aligned $data_root/mfcc_aligned


echo "Training DCTW"
python3 train_model.py --view1_win 3 --view1_pca_ncomps 0.995  --noise_std 0.0 --dtw_distance 'sqeuclidean' --batch_size 512 --latent_shared_dim 10 --latent_private_dim 7 --epochs_per_iter 10 --learning_rate 1e-4 --dropout 0.1  --network_arch '64#32#32' --similarity 'contrastive' --autoencoder_loss_weight 1.0 dctw_model $data_root/sensor $data_root/mfcc
echo "Aligning the data"
python3 align_sequences.py --plot_dir test_dtw_plots.contrastive.sqeuclidean dctw_model $data_root/sensor $data_root/mfcc $data_root/sensor_aligned $data_root/mfcc_aligned


echo "Done!!"
