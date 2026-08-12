CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python leanvae_train.py \
    --default_root_dir "../LeanVAE_ex/run1" \
    --num_nodes 1 \
    --gpus 8 \
    --grad_clip_val 1.0 \
    --lr 5e-5 \
    --lr_min 1e-5 \
    --warmup_steps 5000 \
    --discriminator_iter_start 600000 \
    --progress_bar_refresh_rate 500 \
    --max_steps 700000 \
    --data_path '' \
    --train_datalist "../kinetics-dataset/train/" \
    --val_datalist "../kinetics-dataset/valid/" \
    --batch_size 5 \
    --num_workers 12 \
    --sample_rate 1 \
    --sequence_length 17 \
    --latent_dim 48 \
    --ista_iter_num 5 \
    --ista_layer_num 2 \
    --l_dim 192 \
    --h_dim 576 \
    --embedding_dim 768 \
    --sep_num_layer 3 \
    --fusion_num_layer 5 \
    --patch_size 2 8 8 \
    --wan_vae_pth "./wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \






