#!/bin/bash
python3 eval.py --net squeezenas_lat_small
python3 eval.py --net squeezenas_lat_large
python3 eval.py --net squeezenas_lat_xlarge
python3 eval.py --net squeezenas_mac_small
python3 eval.py --net squeezenas_mac_large
python3 eval.py --net squeezenas_mac_xlarge