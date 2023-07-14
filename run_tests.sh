#!/bin/bash

# Define the models you want to test
models=("bigscience/bloomz"
        #"tiiuae/falcon-7b-instruct"
        #"lmsys/vicuna-7b-v1.3"
       # "bigscience/bloom-560m"
       # "mosaicml/mpt-7b-instruct"
       #"tiiuae/falcon-7b"
       )

# Define the temperatures you want to test
temperatures=(0.3 0.6 0.9)

# Loop over models and temperatures
for model in "${models[@]}"
do
    for temperature in "${temperatures[@]}"
    do
        echo "Running script with model $model and temperature $temperature"
        python ./test.py --model "$model" --temperature "$temperature"
    done
done
