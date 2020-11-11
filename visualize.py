'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import cv2
from submission import epipolarCorrespondence
from submission import triangulate
from matplotlib import pyplot as plt
from helper import epipolarMatchGUI
from helper import displayEpipolarF
#images
im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')

#preselcted points
ld = np.load('../data/templeCoords.npz')
pts1 = np.hstack((ld["x1"], ld["y1"]))

res = np.load("../results/q2_1.npz")
F = res['F']

pts2 = np.array([[epipolarCorrespondence(im1, im2, F, pts1[row, 0], pts1[row, 1])] for row in range(pts1.shape[0])]).squeeze()

M1 = np.hstack((np.eye(3), np.zeros((3,1))))
#load intrinsic camera matrices
ks = np.load('../data/intrinsics.npz')
k1 = ks['K1']
C1 = k1 @ M1

res = np.load("../results/q3_3.npz")
M2 = res["M2"]
C2 = res["C2"]
P = res["P"]
pts3, err = triangulate(C1, pts1, C2, pts2)

np.savez("../results/q4_2", M1=M1, M2=M2, F=F, C1=C1, C2=C2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3[:,0], pts3[:,1], pts3[:,2])

#matplotlib doesn't support axis 'equal' for 3D but there are hacks around it
#courtesy of the internet 
#(https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to)

x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()

x_range = abs(x_limits[1] - x_limits[0])
x_middle = np.mean(x_limits)
y_range = abs(y_limits[1] - y_limits[0])
y_middle = np.mean(y_limits)
z_range = abs(z_limits[1] - z_limits[0])
z_middle = np.mean(z_limits)

plot_radius = 0.5*max([x_range, y_range, z_range])

ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
plt.show()
