import numpy as np

# define a rotation matrix. rotate 45 degree by z axis.

R = np.array([
    [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.],
    [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.],
    [0., 0., 1.]]
)

# calculate the axis and angle.

theta = np.arccos((R.trace() - 1) / 2) # rotation angle

# the rotation axis(column vector) is the eig vector corresponding to the eig value equals 1.
eig_vals, eig_vector = np.linalg.eig(R)
print('eig values:\n', eig_vals.real)
print('eig vector:\n', eig_vector.real)

n = eig_vector[eig_vals == 1].real.T
print('rotation angle:theta=', np.rad2deg(theta), 'rotation axis:n=', n.ravel())
print('i.e: we can find this R is rotating 45 degree by Z axis.')

# Rodrigue's formula: convert Axis-angle to Rotation matrix.

def skew_symmetric(a):
    a = a.ravel()
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

R_ = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * n @ n.T + np.sin(theta) * skew_symmetric(n)
print('restored rotation matrix:\n',R_)
print('check by mse:', np.linalg.norm(R - R_))