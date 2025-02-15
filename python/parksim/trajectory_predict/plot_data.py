import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

trajectory_predict_path = os.path.dirname(Path(os.getcwd()))
data_path = os.path.join(trajectory_predict_path, 'trajectory_predict/data')
names = ['train', 'val', 'test']
name = names[0]
image_path = os.path.join(data_path, '%s_image_history.npy' % name)

with open(image_path, 'rb') as f:
    image_history = np.load(f, allow_pickle=True)

trajectory_history = np.load(os.path.join(data_path, '%s_trajectory_history.npy' % name))
trajectory_future = np.load(os.path.join(data_path, '%s_trajectory_future.npy' % name))
intent_pose = np.load(os.path.join(data_path, '%s_intent.npy' % name))

instance = 0

color_vehs = ['red', 'blue', 'green', 'magenta', 'brown']
plt.plot(trajectory_history[instance, :, 0], trajectory_history[instance, :, 1], linestyle='--', color=color_vehs[instance], marker='o', alpha=0.3)
plt.plot(trajectory_future[instance, :, 0], trajectory_future[instance, :, 1], linestyle='--', color=color_vehs[instance], marker='o', alpha=1)