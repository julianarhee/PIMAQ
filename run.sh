#!/bin/sh

python acquire.py -c ./config.yaml -s -p
python acquire.py -c ./config2.yaml -s -p
python arduino.py /:wq

