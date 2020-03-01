#!/bin/bash
python main.py --train --epochs 100 --eval_epochs 1 --save_epochs 10 --seed 42 --model_dir "./model/ckpt-test" --model_name "pix2pix"
