#!/bin/sh

#latent ro3 cnn16 vgg_noise vgg_noisero3
for model in averagedim
do
    for strength in 0 50 100 200 300 400 500 800 1000 2000
    do
        echo "model: $model strength: $strength"
        python repulse.py --method repulsive --model $model --strength $strength --report
    done
done