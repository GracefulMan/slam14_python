import numpy as np
def skew_symmetric(a):
    a = a.ravel()
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


class Quaternion:
    def __init__(self, s=0, x=0, y=0, z=0):
        self.s = s
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        res = Quaternion()
        res.x = self.x + other.x
        res.y = self.y + other.y
        res.z = self.z + other.z
        res.s = self.s + other.s
        return res

    def __sub__(self, other):
        res = Quaternion()
        res.x = self.x - other.x
        res.y = self.y - other.y
        res.z = self.z - other.z
        res.s = self.s - other.s
        return res

    def __mul__(self, other):
        res = Quaternion()
        if not isinstance(other, Quaternion):
            res.s = self.s * other
            res.x = self.x * other
            res.y = self.y * other
            res.z = self.z * other
        else:
            # doesn't satisfy commutative law.
            res.s = self.s * other.s - self.x * other.x - self.y * other.y - self.z * other.z
            res.x = self.s * other.x + self.x * other.s + self.y * other.z - self.z * other.y
            res.y = self.s * other.y - self.x * other.z + self.y * other.s + self.z * other.x
            res.z = self.s * other.z + self.x * other.y - self.y * other.x + self.z * other.s
        return res

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError
        return self * (1. / other)

    def conjugate(self):
        res = Quaternion()
        res.s = self.s
        res.x = -self.x
        res.y = -self.y
        res.z = -self.z
        return res

    def normalize(self):
        return self / self.norm()

    def norm(self):
        return np.sqrt(self.s ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def inv(self):
        return self.conjugate() / self.norm() ** 2

    def dot(self, other):
        res = Quaternion()
        res.s = self.s * other.s
        res.x = self.x * other.x
        res.y = self.y * other.y
        res.z = self.z * other.z
        return res

    def rotate(self, p):
        return self * p * self.inv()

    def convert_to_rotation_matrix(self):
        R = np.array([
            [1 - 2 * self.y * self.y - 2 * self.z * self.z, 2 * self.x * self.y - 2 * self.s * self.z,
             2 * self.x * self.z + 2 * self.s * self.y],
            [2 * self.x * self.y + 2 * self.s * self.z, 1 - 2 * self.x * self.x - 2 * self.z * self.z,
             2 * self.y * self.z - 2 * self.s * self.x],
            [2 * self.x * self.z - 2 * self.s * self.y, 2 * self.y * self.z + 2 * self.s * self.x,
             1 - 2 * self.x * self.x - 2 * self.y * self.y]
        ])
        return R

    def load_from_rotation_matrix(self, R):
        self.s = np.sqrt(R.trace() + 1) / 2
        self.x = (R[1, 2] - R[2, 1]) / (self.s * 4)
        self.y = (R[2, 0] - R[0, 2]) / (self.s * 4)
        self.z = (R[0, 1] - R[1, 0]) / (self.s * 4)

    def load_from_axis_angle(self, theta, n):
        self.s = np.cos(theta / 2)
        n = n * np.sin(theta / 2)
        n = n.ravel()
        self.x = n[0]
        self.y = n[1]
        self.z = n[2]

    def convert_to_axis_angle(self):
        theta = 2 * np.arccos(self.s)
        n = np.array([self.x, self.y, self.z]) / np.sin(theta / 2)
        return theta, n

    def __str__(self):
        return f'[{self.s} {self.x} {self.y} {self.z}]'


class AngleAxis:
    def __init__(self, theta, n):
        self.theta = theta
        self.n = n
        pass

    def load_from_rotation_matrix(self, R):
        self.theta = np.arccos((R.trace() - 1) / 2)  # rotation angle
        eig_vals, eig_vector = np.linalg.eig(R)
        self.n = eig_vector[eig_vals == 1].real.T

    def convert_to_rotation_matrix(self):
        # use Rodrigues's Formula
        R = np.cos(self.theta) * np.eye(3) + (1 - np.cos(self.theta)) * self.n @ self.n.T + np.sin(self.theta) * skew_symmetric(self.n)
        return R

    def get_angle_axis(self):
        theta = self.theta
        n = self.n.reshape((3, 1))
        return theta, n


