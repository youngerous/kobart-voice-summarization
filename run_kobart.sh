#!/bin/sh

BATCH_SIZE=8
EPOCH=5


python src/kobart/main.py\
    --batch-size=${BATCH_SIZE}\
    --epoch=${EPOCH}\
    --amp\
    --distributed\
