'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import essentialMatrix
import numpy as np

res = np.load("../results/q2_1.npz")
ks = np.load('../data/intrinsics.npz')
k1 = ks['K1']
k2 = ks['K2']
E = essentialMatrix(res['F'], k1, k2)
E