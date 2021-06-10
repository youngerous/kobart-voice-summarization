#!/bin/sh

# TODO: ddp 아닐 경우 에러 발생

BATCH_SIZE=4
EPOCH=5
LR=5e-5
ACCUMULATION_STEP=2

N_ENC=3
N_DEC=3

# run distilBART-6-3
python src/kobart/main.py\
    --batch-size=${BATCH_SIZE}\
    --lr=${LR}\
    --epoch=${EPOCH}\
    --gradient-accumulation-step=${ACCUMULATION_STEP}\
    --amp\
    --distill\
    --n_enc=${N_ENC}\
    --n_dec=${N_DEC}\
    --distributed\
    
