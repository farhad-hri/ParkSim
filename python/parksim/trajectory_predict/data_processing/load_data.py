import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

trajectory_predict_path = os.path.dirname(Path(os.getcwd()))
data_path = os.path.join(trajectory_predict_path, 'data')
names = ["DJI_" + str(i).zfill(4) for i in range(6, 7)]
name = names[0]
image_history = np.load(os.path.join(data_path, '%s_image_history.npy' % name))
trajectory_history = np.load(os.path.join(data_path, '%s_trajectory_history.npy' % name))
trajectory_future = np.load(os.path.join(data_path, '%s_trajectory_future.npy' % name))
intent_pose = np.load(os.path.join(data_path, '%s_intent_pose.npy' % name))