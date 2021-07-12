import numpy as np
from rotation_tools import skew_symmetric, AngleAxis

def SO3_to_so3(theta, n):
    tmp = AngleAxis(theta, n)
    res = tmp.convert_to_rotation_matrix()
    return res



def SE3_to_se3(R, t):
    tmp = AngleAxis()
    tmp.load_from_rotation_matrix(R)
    theta, n = tmp.get_angle_axis()
    J = np.sin(theta) / theta * np.eye(3) + (1 - np.sin(theta) / theta) * n @ n.T + (1 - np.cos(theta)) / theta * skew_symmetric(n)
    roi = np.linalg.inv(J) @ t
    phi = theta * n
    return roi, phi, J




