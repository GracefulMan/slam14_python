# quaternion
# install : pip install numpy-quaternion
import numpy as np
import quaternion

q1 = np.quaternion(0.9238795325112867, 0.0, 0.0, -0.3826834323650898)
q2 = q1 * q1
print(q2)