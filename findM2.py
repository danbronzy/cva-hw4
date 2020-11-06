'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
from helper import camera2
import submission as sub

#load correspondences
corrs = np.load('../data/some_corresp.npz')
pts1 = corrs['pts1']
pts2 = corrs['pts2']

#load intrinsic camera matrices
ks = np.load('../data/intrinsics.npz')
k1 = ks['K1']
k2 = ks['K2']

#calculate F and E
M = np.max(np.hstack((pts1, pts2)))

F = sub.eightpoint(pts1, pts2, M)
E = sub.essentialMatrix(F, k1, k2)

M1 = np.hstack((np.eye(3), np.zeros((3,1))))
C1 = k1 @ M1

M2s = camera2(E)
for ind in range(M2s.shape[2]):
    M2 = M2s[:,:,ind]
    C2 = k2 @ M2
    P, err = sub.triangulate(C1, pts1, C2, pts2)
    if np.all(P[:,2] > 0):
        #This means we one where the points are infront of the camera
        np.savez("../results/q3_3",M2 = M2, C2 = C2, P = P)


