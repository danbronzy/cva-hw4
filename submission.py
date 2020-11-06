"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
from util import refineF


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

    assert pts1.shape == pts2.shape, "Points lists should be same size"
    assert pts1.shape[0] >= 8, "Need at least 8 points for 8 point method"

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    #normalize
    normT = np.array([[2/M, 0,   -1],
                    [0,   2/M,   -1],
                    [0,   0,   1]])

    #homogenous
    hom1 = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    hom2 = np.hstack((pts2, np.ones((pts2.shape[0],1))))

    #all col 3 is ones by design
    n1 = np.dot(normT, hom1.T).T
    n2 = np.dot(normT, hom2.T).T

    #calculate U
    U = np.array([[n1[row, 0] * n2[row, 0], n1[row, 1] * n2[row, 0], n2[row, 0], \
            n1[row, 0] * n2[row, 1], n1[row, 1] * n2[row, 1], n2[row, 1], \
            n1[row, 0], n1[row, 1], 1    ] for row in range(n1.shape[0])])

    #svd to get nonsingular F
    _, _, vh = np.linalg.svd(U)

    F_nonsing = np.reshape(vh[8,:],(3,3))

    #make F singular
    w, s, vh2 = np.linalg.svd(F_nonsing)

    F = (w @ np.diag([s[0], s[1], 0])) @ vh2

    #refine
    F_refined = refineF(F, n1[:,:2], n2[:,:2])

    #unscale
    F_unscaled = normT.T @ F_refined @ normT

    return F_unscaled
'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    #homogeneous
    hom1 = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    hom2 = np.hstack((pts2, np.ones((pts2.shape[0],1))))
    
    ws = np.zeros((pts1.shape[0], 4))
    for ind in range(pts1.shape[0]):
        pt1 = hom1[ind,:].reshape((3,1))
        pt1Stack = np.hstack((pt1,pt1,pt1,pt1)).T
        A1 = np.cross(pt1Stack, C1.T).T[:2,:]

        pt2 = hom2[ind,:].reshape((3,1))
        pt2Stack = np.hstack((pt2,pt2,pt2,pt2)).T
        A2 = np.cross(pt2Stack, C2.T).T[:2,:]

        A = np.vstack((A1, A2))
        #calculate least squares solutions
        _, _, vh = np.linalg.svd(A)

        ws[ind, :] = vh[-1,:].reshape(1,4)


    #normalize last coordinate
    wHom = np.array([ws[row, :]/ws[row, -1] for row in range(ws.shape[0])])
    P = wHom[:,:-1]

    #calculate reprojection
    xHats1 = np.dot(C1, wHom.T).T
    xHats2 = np.dot(C2, wHom.T).T

    #dehomogenize into camera plane and truncate
    xHat1Real = np.array([xHats1[row, :-1]/xHats1[row, -1] for row in range(xHats1.shape[0])])
    xHat2Real = np.array([xHats2[row, :-1]/xHats2[row, -1] for row in range(xHats2.shape[0])])

    err = np.sum([np.linalg.norm(xHat1Real - pts1, axis = 1),\
                  np.linalg.norm(xHat2Real - pts2, axis = 1)])
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
