python leanvae_inference.py --ckpt_path "./ckpts/LeanVAE-dim16.ckpt" \
                      --device "cuda:0" \
                      --input_video {PATH_TO_INPUT_DATA_DIR} \
                      --reconstruct_video {PATH_TO_OUTPUT_DATA_DIR} 
                      #FOR Tile Inference
                      #--tile_inference 
                      #--chunksize_enc 5  
                      #--chunksize_dec 5   
                     
