from pathlib import Path
import torch

import matplotlib.pyplot as plt

from dlp.dataset import Dataset

from parksim.intent_predict.cnn.data_processing.utils import CNNDataProcessor
from parksim.trajectory_predict.data_processing.utils import TransformerDataProcessor

from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_vision_transformer import TrajectoryPredictorVisionTransformer
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_with_decoder_intent_cross_attention import TrajectoryPredictorWithDecoderIntentCrossAttention

from parksim.trajectory_predict.intent_transformer.multimodal_prediction import predict_multimodal
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ds = Dataset()
home_path = str(Path.home())
data_path = home_path + '/PycharmProjects/ParkSim_tc/data/DJI_0012'
ds.load(data_path)

MODEL_PATH = r'checkpoints/epoch=52-val_total_loss=0.0458.ckpt'
traj_model = TrajectoryPredictorWithDecoderIntentCrossAttention.load_from_checkpoint(MODEL_PATH)

traj_model.eval().to(DEVICE)
mode='v1'
INTENT_MODEL_PATH = r'models/smallRegularizedCNN_L0.068_01-29-2022_19-50-35.pth'
intent_model = SmallRegularizedCNN()
model_state = torch.load(INTENT_MODEL_PATH, map_location=DEVICE)
intent_model.load_state_dict(model_state)
intent_model.eval().to(DEVICE)

intent_extractor = CNNDataProcessor(ds=ds)
traj_extractor = TransformerDataProcessor(ds=ds)

def draw_prediction(multimodal_prediction, inst_centric_view, colors, intent_offsets):
    sensing_limit = 20
    img_size = inst_centric_view.size[0] / 2
    plt.cla()
    plt.imshow(inst_centric_view)
    y_label, _, _, _ = multimodal_prediction[0]
    traj_future_pixel = y_label[0, :, :2].detach().cpu().numpy() / \
                        sensing_limit * img_size + img_size
    plt.plot(traj_future_pixel[:, 0], traj_future_pixel[:, 1], 'wo', linewidth=2, markersize=2)
    for prediction, color, offset in zip(reversed(multimodal_prediction), reversed(colors), reversed(intent_offsets)):
        _, pred, intent, probability = prediction
        intent_pixel = intent[0, 0, :2].detach().cpu().numpy() / \
                       sensing_limit * img_size + img_size
        traj_pred_pixel = pred[0, :, :2].detach().cpu().numpy() / \
                          sensing_limit * img_size + img_size
        plt.plot(traj_pred_pixel[:, 0], traj_pred_pixel[:, 1],
                 '^', color=color, linewidth=2, markersize=2)
        plt.plot(intent_pixel[0], intent_pixel[1],
                 '*', color=color, markersize=8)
        plt.text(intent_pixel[0] + offset[0], intent_pixel[1] + offset[1],
                 f'{probability:.2f}', backgroundcolor=(170 / 255., 170 / 255., 170 / 255., 0.53), color='black',
                 size=7, weight='bold')
        print(color, probability)
    plt.axis('off')
    plt.show()

## Example 1
scene = ds.get('scene', ds.list_scenes()[0])
frame_index = 5900
frame = ds.get_future_frames(scene['first_frame'],timesteps=6000)[frame_index]
inst_token = frame['instances'][37]
multimodal_prediction, inst_centric_view = predict_multimodal(
    ds, traj_model, intent_model, traj_extractor, intent_extractor, inst_token, frame_index, 3, mode=mode)
time_all = []
for i in range(100):
    start_time = time.time()
    multimodal_prediction, inst_centric_view = predict_multimodal(
        ds, traj_model, intent_model, traj_extractor, intent_extractor, inst_token, frame_index, 3, mode=mode)
    time_all.append(time.time() - start_time)
print("Pred comp time: ", (np.mean(time_all), np.std(time_all)))
colors = ['darkviolet', 'C1', 'green']
intent_offsets = [[-50,0], [0,-30], [-50,0]]
draw_prediction(multimodal_prediction, inst_centric_view,
                colors, intent_offsets)