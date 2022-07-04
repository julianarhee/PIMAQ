#!/bin/sh

python acquire_single.py -c ./config.yaml -s -p
python acquire_single.py -c ./config2.yaml -s -p
python arduino.py

