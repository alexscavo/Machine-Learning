import numpy

def mcol(v):    #transposed vector vertically
    return v.reshape(v.size, 1)

def mrow(v):    #transposed vector horitzonally
    return v.reshape(1, v.size)