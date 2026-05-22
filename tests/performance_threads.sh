#!/bin/bash
#for hilos in 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128
for hilos in 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 144 160 176 192 208 224 240 256
do
    OMP_NUM_THREADS=$hilos python performance_threads.py
done