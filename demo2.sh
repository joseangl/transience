#!/bin/bash

data_root="../data/TiDigits/Cross_Jim10Nov_Trudy13Apr"

# echo "Vocoding with WORLD"
# mkdir ${data_root}/world
# python3 world.py $data_root/wav $data_root/world

echo "Training the DCTW model"
# Copy only the MFCC files
# mkdir $data_root/mfcc
# for f in $data_root/world/*.mfcc.npy
# do
#    fbname=$(basename $f .mfcc.npy)
#    cp $data_root/world/$fbname.mfcc.npy $data_root/mfcc/$fbname.npy 
# done

# # Convert the sensor data to numpy format
# python3 htk2npy.py ../data/sensor htk

# Train the DCTW model
python3 train_model.py --output_dim 16 --epochs_per_iter 10 --learning_rate 1e-4 --dropout 0.0 --noise_std 0.1 --network_arch '64#32#32' --view1_win 3 --similarity 'mmi' --autoencoder_loss_weight 0.1 dctw_model $data_root/sensor_matched $data_root/mfcc_matched
# python3 train_model.py --output_dim 16 --epochs_per_iter 10 --learning_rate 1e-4 --dropout 0.5 --network_arch '64#32#32' --view1_win 3 --similarity 'cca' --autoencoder_loss_weight 0.1 dctw_model $data_root/sensor_matched $data_root/mfcc_matched


echo "Aligning the data"
python3 align_sequences.py --plot_dir test_dtw_plots dctw_model $data_root/sensor_matched $data_root/mfcc_matched $data_root/sensor_matched_aligned $data_root/mfcc_matched_aligned

echo "Done!!"
