from .user.net import UserNet
from .discriminator.crossformer import Discriminator

model_list = {
    "UserNet": UserNet,
    "discriminator": Discriminator
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
