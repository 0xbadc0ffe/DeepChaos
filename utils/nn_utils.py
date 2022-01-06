import torch
import torch.optim as optim
import random
import numpy as np

def set_reproducibility(seed=42):
    # reproducibility stuff
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
    torch.backends.cudnn.benchmark = False



def count_parameters(model: torch.nn.Module) -> int:
  """ Counts the number of trainable parameters of a module
  
  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_model_optimizer(model: torch.nn.Module, opt_type:str) -> torch.optim.Optimizer:
    """
    Encapsulate the creation of the model's optimizer, to ensure that we use the
    same optimizer everywhere

    :param model: the model that contains the parameter to optimize

    :returns: the model's optimizer
    """
    if opt_type == "Adam":
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    elif opt_type == "SGD":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.1, weight_decay=1e-5)
    else:
        # default
        return optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
