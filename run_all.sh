#!/bin/bash

echo "=== Running bicycle ==="
python [train.py](http://train.py/) -s data/bicycle -m output/bicycle

echo "=== Running garden ==="
python [train.py](http://train.py/) -s data/garden -m output/garden

echo "=== Running train ==="
python [train.py](http://train.py/) -s data/train -m output/train

echo "=== DONE ==="
