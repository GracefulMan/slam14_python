import numpy as np


def euler_convert_to_rotation_matrix(yaw, pitch, roll):
    # ZXY rotation, called rpy. problem: Gimbal Lock.
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # ZYX
    return Rx @ Ry @ Rz


def rotation_matrix_to_euler(R):
    roll = np.arctan(- R[1, 2] / R[2, 2])
    pitch = np.arcsin(R[0, 2])
    yaw = np.arctan(-R[0, 1] / R[0, 0])
    return roll, pitch, yaw

R = np.array([
    [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.],
    [np.sqrt(2) / 2, np.sqrt(2) / 2, 0.],
    [0., 0., 1.]]
)

print('get euler angle from rotation matrix:')
roll, pitch, yaw = rotation_matrix_to_euler(R)
print(f'roll={np.rad2deg(roll)}, pitch={np.rad2deg(pitch)}, yaw={np.rad2deg(yaw)}')
print('convert euler angle to rotation matrix:')
R_ = euler_convert_to_rotation_matrix(yaw, pitch, roll)
print(R_)