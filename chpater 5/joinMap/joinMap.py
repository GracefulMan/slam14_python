import numpy as np
import cv2
from rotation_tools import Quaternion, Isometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



poses_list = np.loadtxt('pose.txt')
colorImgs, depthImgs = [], []
poses = []
for i in range(5):
    c_imgs = cv2.imread(f'color/{i+1}.png')
    d_imgs = cv2.imread(f'depth/{i+1}.pgm', -1)
    data = poses_list[i]
    q = Quaternion(data[6], data[3], data[4], data[5])
    t = np.array([data[0], data[1], data[2]])
    T = Isometry(q, t)
    poses.append(T)
    colorImgs.append(c_imgs)
    depthImgs.append(d_imgs)

cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0
depthScale = 1000.0
print('convert img to points cloud...')
pointcloud = []
colors = []
for i in range(5):
    print('convert img:', i)
    color = colorImgs[i]
    depth = depthImgs[i]
    T = poses[i]
    for u in range(color.shape[0]):
        for v in range(color.shape[1]):
            d = depth[u, v]
            if d == 0: continue
            d /= depthScale
            x = (v - cx) * d / fx
            y = (u - cy) * d / fy
            p = np.array([x, y, d])
            p_w = T * p
            pointcloud.append(p_w.ravel())
            colors.append(color[u, v].ravel())


print('total number of points:', len(pointcloud) )
fig=plt.figure(dpi=500)
ax = fig.add_subplot(111, projection='3d')
points= np.array(pointcloud)
colors = np.array(colors) / 255

ax.scatter(points[:,0], points[:,1], points[:,2],
           cmap='spectral',
           c=colors,
           s=0.1,
           linewidth=0,
           alpha=1,
           marker=".")

plt.show()
