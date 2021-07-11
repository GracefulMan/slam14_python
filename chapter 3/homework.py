# 7
import numpy as np

from quaternion_ import Quaternion

q1 = Quaternion(0.35, 0.2, 0.3, 0.1).normalize()
t1 = np.array([[0.3], [0.1], [0.1]])

q2 = Quaternion(-0.5, 0.4, -0.1, 0.2).normalize()
t2 = np.array([[-0.1], [0.5], [0.3]])

p = np.array([[0.5], [0], [0.2], [1]])


def get_transform_matrix(rotation_matrix, translation):
    res = np.hstack((rotation_matrix, translation))
    tmp = np.array([[0, 0, 0, 1.]])
    return np.vstack((res, tmp))


r1 = q1.convert_to_rotation_matrix()
T1 = get_transform_matrix(r1, t1)
print('Transform Matrix:T1=\n', T1)
r2 = q2.convert_to_rotation_matrix()
T2 = get_transform_matrix(r2, t2)
print('Transform Matrix:T2=\n', T2)

res = T2 @ np.linalg.inv(T1) @ p
print('coordinate:', res.ravel()[:3])

# 6 TODO: solve Ax=b