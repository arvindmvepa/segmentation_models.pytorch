from .unet import Unet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3
from .pan import PAN

from . import encoders
from . import utils

from .__version__ import __version__

decoders = {"unet": Unet,
            "linknet": Linknet,
            "fpn": FPN,
            "pspnet": PSPNet,
            "deeplabv3": DeepLabV3,
            "pan": PAN}
