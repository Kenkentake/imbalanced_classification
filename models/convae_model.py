import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule
from utils import save_confusion_matrix


class ConvAEModel(LightningModule):
    def __init__(self, args, device):
        super(ConvAEModel, self).__init__()
        self.args = args
        self.new_device = device
        self.learning_rate = args.TRAIN.LEARNING_RATE
        self.mse_loss = nn.MSELoss()

        # encoder
        self.encoder = nn.Sequential(
                            ConvBatchNormRelu(3, 3, 32, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(32, 3, 64, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(64, 3, 128, 1),
                            nn.MaxPool2d(2, 2),
                            ConvBatchNormRelu(128, 3, 256, 1),
                            nn.MaxPool2d(2, 2),
                        )

        # decoder
        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(256, 128, 2, stride=2),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(128, 64, 2, stride=2),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(64, 32, 2, stride=2),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(32, 3, 2, stride=2),
                            nn.Sigmoid()
                        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        decoded = self(inputs)
        ae_loss = self.mse_loss(decoded, inputs)
        return {'loss': ae_loss,
                'count': labels.shape[0]}

    def training_epoch_end(self, outputs):
        ae_loss = 0.0
        count = 0
        for output in outputs:
            ae_loss += output['loss'].data.item()
            count += output['count']

        training_epoch_outputs = {'training_ae_loss': ae_loss / count}

        self.logger.log_metrics(training_epoch_outputs, step=self.current_epoch)
        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        decoded = self(inputs)
        ae_loss = self.mse_loss(decoded, inputs)

        return {'count': labels.shape[0],
                'loss': ae_loss}

    def validation_epoch_end(self, outputs) -> dict:
        ae_loss = 0.0
        count = 0
        for output in outputs:
            ae_loss += output['loss'].data.item()
            count += output['count']

        validation_epoch_outputs = {'validation_ae_loss': ae_loss / count}

        self.logger.log_metrics(validation_epoch_outputs, step=self.current_epoch)
        return None

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        decoded = self(inputs)

        # save input and decoded img
        if batch_idx < 5 :
            self.logger.experiment.add_images(tag='input_image', img_tensor=inputs[:5], dataformats='NCHW')
            self.logger.experiment.add_images(tag='decoded_image', img_tensor=decoded[:5], dataformats='NCHW')

        ae_loss = self.mse_loss(decoded, inputs)

        return {'count': labels.shape[0],
                'loss': ae_loss,}
            
    def test_epoch_end(self, outputs) -> dict:
        ae_loss = 0.0
        count = 0

        for output in outputs:
            ae_loss += output['loss'].data.item()
            count += output['count']
        
        test_epoch_outputs = {
            'test_ae_loss': ae_loss / count,
        }

        self.logger.log_metrics(test_epoch_outputs, step=self.current_epoch)

        return None

class ConvBatchNormRelu(LightningModule):
    def __init__(self, input_channel, kernel_size, output_channel, padding):
        super(ConvBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding)
        self.batchnorm = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
