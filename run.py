import os

# run before the script - export NCCL_P2P_DISABLE = 1

# # Train classifier
# os.environ["CLASSIFIER_FLAGS"] = "--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 256 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
# os.environ["TRAIN_FLAGS"] = "--iterations 30000 --anneal_lr True --batch_size 2 --lr 3e-4 --save_interval 100000 --weight_decay 0.05"
# # os.system("mpiexec -n 2 python scripts/classifier_train.py --data_dir datasets/horse2zebra/train $TRAIN_FLAGS $CLASSIFIER_FLAGS")
#
# # Train diffusion
# os.environ["MODEL_FLAGS"] = "--image_size 256 --num_channels 64 --num_res_blocks 3 --class_cond True --learn_sigma True"
# os.environ["DIFFUSION_FLAGS"] = "--diffusion_steps 1000"
# os.environ["TRAIN_FLAGS"] = "--lr 1e-4 --batch_size 4 --save_interval 100000"
# # os.system("mpiexec -n 2 python scripts/image_train.py --data_dir datasets/horse2zebra/train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS")
#
# os.environ["SAMPLE_FLAGS"] = "--batch_size 1 --num_samples 1 --timestep_respacing 250"
# os.system("mpiexec -n 2 python scripts/classifier_sample.py --classifier_scale 1.0 --classifier_path models/horse2zebra_classifier.pt --model_path models/horse2zebra_diffusion.pt --data_dir datasets/horse2zebra/train $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS")


# Train classifier
os.environ["CLASSIFIER_FLAGS"] = "--image_size 64 --classifier_depth 4"
os.environ["TRAIN_FLAGS"] = "--iterations 300000 --anneal_lr True --batch_size 128 --lr 3e-4 --save_interval 100000 --weight_decay 0.05"
os.system("mpiexec -n 2 python scripts/classifier_train.py --data_dir datasets/horse2zebra_64 --feature_extraction True --load_model models/64x64_classifier.pt $TRAIN_FLAGS $CLASSIFIER_FLAGS")

# # Train diffusion
os.environ["MODEL_FLAGS"] = "--image_size 64 --attention_resolutions 32,16,8 --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
os.environ["DIFFUSION_FLAGS"] = "--diffusion_steps 1000 --learn_sigma True --noise_schedule cosine "
os.environ["TRAIN_FLAGS"] = "--lr 1e-4 --batch_size 4 --save_interval 100000"
os.system("mpiexec -n 2 python scripts/image_train.py --data_dir datasets/horse2zebra_64 --feature_extraction True --load_model models/64x64_diffusion.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS")
