#!/usr/bin/env python3

def moving_average(data, beta):
    avg = []
    prev = 0
    for i, d in enumerate(data):
        prev = (beta * prev + (1 - beta) * d)
        correction = prev / (1 - (beta ** (i + 1)))
        avg.append(correction)
    return avg