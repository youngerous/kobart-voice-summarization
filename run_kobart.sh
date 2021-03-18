#!/bin/sh

BATCH_SIZE=8
EPOCH=5
LR=3e-5

python src/kobart/main.py\
    --batch-size=${BATCH_SIZE}\
    --lr=${LR}\
    --epoch=${EPOCH}\
    --distributed\
    --amp\
