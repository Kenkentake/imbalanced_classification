import itertools
import matplotlib.pyplot as plt
import numpy as np

from mlflow.tracking.client import MlflowClient
from os.path import join
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(labels, preds, classes, run_id, tmp_results_dir):
    mlflow_client = MlflowClient()
    img_name = 'confusion_matrix.jpg'
    # making confusion matrix
    # label_name=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_name = list(range(len(classes)))
    conf_matrix = confusion_matrix(labels, preds, label_name, normalize='true')
    # plot and save
    plt.figure()
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], '.2f'),
                 horizontalalignment='center',
                 color='white' if conf_matrix[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel('Pred Label')
    plt.ylabel('Ground Truth')
    plt.savefig(join(tmp_results_dir, img_name))
    mlflow_client.log_artifact(run_id, join(tmp_results_dir, img_name))




