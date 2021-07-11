'''
the basic implementation of rotation matrix, Axis-angle, Euler angle and Quaternion.
'''
import numpy as np


def cross_product():
    # study cross product and skew-symmetric
    a = np.random.rand(3)
    b = np.random.rand(3)
    print('a cross product b:', np.cross(a, b))
    print('using skew-symmetric:')

    def skew_symmetric(a):
        return np.array([
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0]
        ])

    print('cross by using skew_symmetric:', skew_symmetric(a) @ b)


def rotation_matrix():
    # create two coordinate which rotate by z axis: angle: 45 degree.
    e_1 = np.array([[1], [0], [0]])
    e_2 = np.array([[0], [1], [0]])
    e_3 = np.array([[0], [0], [1]])
    E = np.hstack((e_1, e_2, e_3))
    e_1p = np.array([[1.], [1.], [0.]]) / np.sqrt(2)
    e_2p = np.array([[-1.], [1.], [0.]]) / np.sqrt(2)
    e_3p = np.array([[0], [0], [1]])
    Ep = np.hstack((e_1p, e_2p, e_3p))
    R = E.T @ Ep
    print('rotation matrix:R=\n', R)
    print('R inv:\n', np.linalg.inv(R))
    print('R.T:\n', R.T)
    print('we can find R.T == inv(R)')
    print('check orthogonal property:', np.linalg.norm(R @ R.T - np.eye(3)))

    # using homogeneous representation
    # create two translation vector.
    t1 = np.array([[1], [2], [3]])
    t2 = np.array([[3], [2], [1]])
    R1 = R
    R2 = R1 @ R1 @ R1
    p = np.array([[2], [3], [5]])
    print('after transformation:')
    res1 = R2 @ (R1 @ p + t1) + t2
    print('res=', res1.ravel())
    print('using Homogeneous coordinates:')
    p1 = np.vstack((p, 1))
    T1 = np.hstack((R1, t1))
    tmp = np.array([[0, 0, 0, 1]])
    T1 = np.vstack((T1, tmp))
    print('Transform matrix:T1=\n', T1)

    T2 = np.hstack((R2, t2))
    tmp = np.array([[0, 0, 0, 1]])
    T2 = np.vstack((T2, tmp))
    print('Transform matrix:T2=\n', T2)
    res2 = T2 @ T1 @ p1
    print('res=', res2.ravel()[:3])


if __name__ == '__main__':
    rotation_matrix()
