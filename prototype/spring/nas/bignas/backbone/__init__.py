from .bignas_resnet_basic_subnet import (  # noqa: F401
    bignas_resnet_basic
)
from .bignas_resnet_bottleneck_subnet import (  # noqa: F401
    bignas_resnet_bottleneck
)
from .bignas_resnetd_bottleneck_subnet import (  # noqa: F401
    bignas_resnetd_bottleneck
)
from .bignas_mobilenetv3_subnet import (  # noqa: F401
    bignas_mobilenetv3
)
from .bignas_regnet_subnet import (  # noqa: F401
    bignas_regnet
)
from .bignas_fpn_subnet import (  # noqa: F401
    bignas_fpn
)
from .bignas_roi_head_subnet import (  # noqa: F401
    bignas_roi_head
)
from .bignas_resnetd_basic_subnet import (  # noqa: F401
    bignas_resnetd_basic
)

from prototype.spring.models import SPRING_MODELS_REGISTRY

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        SPRING_MODELS_REGISTRY.register(var_name, var)
