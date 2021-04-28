python train.py \
    --output_directory=/tacotron2/output \
    --log_directory=/tacotron2/log \
    --n_gpus=1 \
    --training_files=/tacotron2/filelists/taco_train.txt \
    --validation_files=/tacotron2/filelists/train_test.txt \
    --epochs=5000000