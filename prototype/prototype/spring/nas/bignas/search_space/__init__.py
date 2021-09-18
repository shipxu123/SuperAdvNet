from .bignas_resnet_basic import big_resnet_basic # noqa
from .bignas_resnet_bottleneck import big_resnet_bottleneck # noqa
from .bignas_mobilenetv3 import big_mobilenetv3 # noqa
from .bignas_regnet import big_regnet # noqa
from .base_bignas_searchspace import Bignas_SearchSpace # noqa
from .bignas_resnetd_basic import big_resnetd_basic # noqa
from .bignas_resnetd_bottleneck import big_resnetd_bottleneck # noqa

from prototype.spring.models import SPRING_MODELS_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        SPRING_MODELS_REGISTRY.register(var_name, var)
