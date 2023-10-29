# How to use
## prepare the data
create a data directory which contains:
train:
    sample
    mask
test:
    sample
    mask
val:
    sample
    mask
## preprocess the data
python preprocess.py

## train model
deepspeed train.py

## test model
deepspeed eval.py


