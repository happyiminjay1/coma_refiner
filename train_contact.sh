#!/usr/bin/env bash


python main.py --exp_name 'interpolation_exp_mano_contact[16 16 16 32] 128' --split 'interpolation' --device_idx 0 --in_channels 27 --output_channels 13 --latent_channels 128 --out_channels 16 16 16 32
python main.py --exp_name 'interpolation_exp_mano_contact[16 16 16 32] 256' --split 'interpolation' --device_idx 1 --in_channels 27 --output_channels 13 --latent_channels 256 --out_channels 16 16 16 32
python main.py --exp_name 'interpolation_exp_mano_contact[16 16 16 32] 64' --split 'interpolation' --device_idx 2 --in_channels 27 --output_channels 13 --latent_channels 64 --out_channels 16 16 16 32

python main.py --exp_name 'interpolation_exp_mano_contact[32, 32, 32, 64] 128' --split 'interpolation' --device_idx 3 --in_channels 27 --output_channels 13 --latent_channels 128 --out_channels 32 32 32 64
python main.py --exp_name 'interpolation_exp_mano_contact[32, 32, 32, 64] 256' --split 'interpolation' --device_idx 4 --in_channels 27 --output_channels 13 --latent_channels 256 --out_channels 32 32 32 64
python main.py --exp_name 'interpolation_exp_mano_contact[32, 32, 32, 64] 64' --split 'interpolation' --device_idx 5 --in_channels 27 --output_channels 13 --latent_channels 64 --out_channels 32 32 32 64

python main.py --exp_name 'interpolation_exp_mano_contact[32 32 32 32] 128' --split 'interpolation' --device_idx 6 --in_channels 27 --output_channels 13 --latent_channels 128 --out_channels 32 32 32 32
python main.py --exp_name 'interpolation_exp_mano_contact[32 32 32 32] 256' --split 'interpolation' --device_idx 7 --in_channels 27 --output_channels 13 --latent_channels 256 --out_channels 32 32 32 32

python main.py --exp_name 'interpolation_exp_mano_contact[32, 32, 32, 64] 128' --split 'interpolation' --device_idx 3 --in_channels 27 --output_channels 13 --latent_channels 128 --out_channels 32 32 32 64

interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 2/mesh_results/300

python main.py --exp_name 'interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none dataset' --ori_exp_name 'interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 2' --split 'interpolation' --device_idx 6 --in_channels 27 --output_channels 13 --latent_channels 128 --out_channels 32 32 32 64 --loss_weight 0 0 0 1 1 0 1 1 