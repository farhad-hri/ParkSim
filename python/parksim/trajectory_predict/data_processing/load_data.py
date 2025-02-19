import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

trajectory_predict_path = os.path.dirname(Path(os.getcwd()))
data_path = os.path.join(trajectory_predict_path, 'data')
names = ["DJI_" + str(i).zfill(4) for i in range(6, 7)]
name = names[0]

with open(os.path.join(data_path, '%s_image_history.npy' % name), 'rb') as f:
    image_history = np.load(f)
with open(os.path.join(data_path, '%s_trajectory_history.npy' % name), 'rb') as f:
    trajectory_history = np.load(f)
with open(os.path.join(data_path, '%s_trajectory_future.npy' % name), 'rb') as f:
    trajectory_future = np.load(f)
with open(os.path.join(data_path, '%s_intent_pose.npy' % name), 'rb') as f:
    intent_pose = np.load(f)

total_data = np.concatenate((trajectory_history, trajectory_future), axis=1)
total_data = total_data[..., np.newaxis]
total_data = np.transpose(total_data, (0, 2, 1, 3)) # data point (scenario) x 2 (states) x T (time) x N (no. of agents=1)
total_data_list = [total_data[i, :2] for i in range(total_data.shape[0])]
train_val_split_fraction = 0.8
train_val_split = int(len(total_data_list)*train_val_split_fraction)
train_data = total_data_list[:train_val_split]
val_data = total_data_list[train_val_split:]

with open(os.path.join(data_path, '%s_train.npy' % name), 'wb') as fp:
    pickle.dump(train_data, fp)

with open(os.path.join(data_path, '%s_val.npy' % name), 'wb') as fp:
    pickle.dump(val_data, fp)

# plt.plot(train_data[20][0, :, 0], train_data[20][1, :, 0], marker='o')
# plt.show()
# image_history = np.load(os.path.join(data_path, '%s_image_history.npy' % name))
# trajectory_history = np.load(os.path.join(data_path, '%s_trajectory_history.npy' % name))
# trajectory_future = np.load(os.path.join(data_path, '%s_trajectory_future.npy' % name))
# intent_pose = np.load(os.path.join(data_path, '%s_intent_pose.npy' % name))