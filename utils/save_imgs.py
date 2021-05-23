import torch
import torchvision

from mlflow.tracking.client import MlflowClient
from os.path import join

def save_imgs(idx, inputs_tensor, outputs_tensor, run_id, tmp_results_dir):
    mlflow_client = MlflowClient()
    img_list = []
    for i in range(inputs_tensor.size(0)):
        img_list.append(inputs_tensor[i])
        img_list.append(outputs_tensor[i])
    # pair_imgs = torchvision.utils.make_grid(img_list, nrow=2, normalize=True)
    pair_imgs = torchvision.utils.make_grid(img_list, nrow=2)
    img_name = 'input&encoded_' + str(idx) + '.jpg'
    torchvision.utils.save_image(pair_imgs, join(tmp_results_dir, img_name))
    # mlflow_client.log_artifact(run_id, join(tmp_results_dir, 'input&encoded.jpg'))
    mlflow_client.log_artifact(run_id, join(tmp_results_dir, img_name))

