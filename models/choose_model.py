from models.cnn_classifier_model import CNNClassifierModel
from models.convae_model import ConvAEModel
from models.convae_classifier_model import ConvAEClassifierModel

def choose_model(args, device):
    model_type = args.TRAIN.MODEL_TYPE
    # common params for model
    
    model_dict = {
            "cnn_classifier": CNNClassifierModel,
            "convae": ConvAEModel,
            "convae_classifier": ConvAEClassifierModel
            }

    if model_type not in model_dict:
        print("Choosed model type is not in model_dict")
        raise NotImplementedError()

    model = model_dict[model_type](args, device)
    return model
