import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule
from utils import save_confusion_matrix


class ConvAEClassifierModel(LightningModule):
    def __init__(self, args, device):
        super(ConvAEClassifierModel, self).__init__()
        self.args = args
        self.new_device = device
        self.learning_rate = args.TRAIN.LEARNING_RATE

        self.weight = self.args.TRAIN.LOSS_WEIGHT
        self.cross_entropy_loss = nn.CrossEntropyLoss()
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
        # classifier
        self.classifier = nn.Sequential(
                            nn.Linear(256 * 2 * 2, 512),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.2),
                            nn.Linear(512, 10),
                        )


    def forward(self, x):
        with torch.no_grad():    
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        clf_in = encoded.view(-1, 256 * 2 * 2)
        clf_out = self.classifier(clf_in)
        return clf_out, decoded

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)
        ae_loss = self.mse_loss(decoded, inputs)
        if len(self.args.TRAIN.CE_CLASS_WEIGHT) != 0:
            weight = torch.tensor(self.args.TRAIN.CE_CLASS_WEIGHT).to(self.new_device)
            cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
            clf_loss = cross_entropy_loss(outputs, labels)
        else:
            clf_loss = self.cross_entropy_loss(outputs, labels)
        loss = self.weight[0] * ae_loss + self.weight[1] * clf_loss
        accuracy = (outputs.argmax(1) == labels).sum().item()
        return {
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
            'ae_loss': ae_loss,
            'clf_loss': clf_loss
            }

    def training_epoch_end(self, outputs) -> None:
        accuracy = loss = ae_loss = clf_loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            ae_loss += output['ae_loss'].data.item()
            clf_loss += output['clf_loss'].data.item()
            count += output['count']

        training_epoch_outputs = {
            'training_accuracy': accuracy / count,
            'training_loss': loss / count,
            'training_ae_loss': ae_loss / count,
            'training_clf_loss': clf_loss / count
        }

        self.logger.log_metrics(training_epoch_outputs, step=self.current_epoch)
        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)

        ae_loss = self.mse_loss(decoded, inputs)
        clf_loss = self.cross_entropy_loss(outputs, labels)
        loss = self.weight[0] * ae_loss + self.weight[1] * clf_loss

        accuracy = (outputs.argmax(1) == labels).sum().item()

        return {
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
            'ae_loss': ae_loss,
            'clf_loss': clf_loss
        }

    def validation_epoch_end(self, outputs) -> dict:
        accuracy = loss = ae_loss = clf_loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            ae_loss += output['ae_loss'].data.item()
            clf_loss += output['clf_loss'].data.item()
            count += output['count']

        validation_epoch_outputs = {
            'validation_accuracy': accuracy / count,
            'validation_loss': loss / count,
            'validation_ae_loss': ae_loss / count,
            'validation_clf_loss': clf_loss / count
        }

        self.logger.log_metrics(validation_epoch_outputs, step=self.current_epoch)
        return None

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs, decoded = self(inputs)

        # save input and decoded img
        if batch_idx < 5 :
            self.logger.experiment.add_images(tag='input_image', img_tensor=inputs[:5], dataformats='NCHW')
            self.logger.experiment.add_images(tag='decoded_image', img_tensor=decoded[:5], dataformats='NCHW')

        ae_loss = self.mse_loss(decoded, inputs)
        clf_loss = self.cross_entropy_loss(outputs, labels)
        loss = self.weight[0] * ae_loss + self.weight[1] * clf_loss

        accuracy = (outputs.argmax(1) == labels).sum().item()

        # calc indivisual class accuracy
        class_correct = list(0 for i in range(10))
        class_counter = list(0 for i in range(10))
        is_correct = (outputs.argmax(1) == labels).squeeze()
        for i, pred, label in zip(list(range(len(outputs))), outputs, labels):
            class_correct[label] += is_correct[i].item()
            class_counter[label] += 1

        return {
            'preds': outputs.argmax(1),
            'labels': labels,
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
            'ae_loss': ae_loss,
            'clf_loss': clf_loss,
            'class_correct': class_correct,
            'class_counter': class_counter
        }
            
    def test_epoch_end(self, outputs) -> dict:
        accuracy = loss = ae_loss = clf_loss = 0.0
        count = 0
        all_class_correct = list(0 for i in range(10))
        all_class_counter = list(0 for i in range(10))
        preds_all = []
        labels_all = [] 

        for output in outputs:
            preds_all.extend(output['preds'].tolist())
            labels_all.extend(output['labels'].tolist())
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            ae_loss += output['ae_loss'].data.item()
            clf_loss += output['clf_loss'].data.item()
            count += output['count']
            for i in range(10):
                class_correct = output['class_correct']
                class_counter = output['class_counter']
                all_class_correct = [x + y for x, y in zip(all_class_correct, class_correct)]
                all_class_counter = [x + y for x, y in zip(all_class_counter, class_counter)]
        fig_conf_matrix = save_confusion_matrix(labels_all, preds_all, self.args.DATA.CLASSES)

        self.logger.experiment.add_figure("Confusion Matrix", fig_conf_matrix)
        
        class_accuracy = {'test_accuracy_{}'.format(i): x/y for i, x, y in zip(list(range(10)), all_class_correct, all_class_counter)} 
        test_epoch_outputs = {
            'test_mean_accuracy': accuracy / count,
            'test_loss': loss / count,
            'test_ae_loss': ae_loss / count,
            'test_clf_loss': clf_loss / count
        }
        test_epoch_outputs.update(**class_accuracy)
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
