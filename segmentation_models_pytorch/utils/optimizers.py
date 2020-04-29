from torch.optim import Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD

optimizers = {"adadelta": Adadelta,
              "adagrad": Adagrad,
              "adam": Adam,
              "adamw": AdamW,
              "sadam": SparseAdam,
              "adamax": Adamax,
              "asgd": ASGD,
              "lbfgs": LBFGS,
              "rmsprop": RMSprop,
              "rprop": Rprop,
              "sgd": SGD
              }