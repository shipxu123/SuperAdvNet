import linklink as link  # noqa
from prototype.spring.analytics.io import send, send_async  # noqa
from prototype.spring.utils.dist_helper import get_rank


__all__ = ["Registry"]


def _register_generic(module_dict, module_name, module):

    assert module_name not in module_dict, f"{module_dict.keys()} : {module_name}"
    module_dict[module_name] = module


def _register_commands(module_dict, module_name, module):
    assert module_name not in module_dict, f"{module_dict.keys()} : {module_name}"

    if hasattr(module, "add_subparser"):

        module_dict[module_name] = module.add_subparser


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creating a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn

    def register_command(self, module_name, module=None):
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_commands(self, module_name, fn)
            return fn

        return register_fn

    def get(self, module_name):
        if module_name not in self:
            assert module_name in self, '{} is not supported, avaiables are:{}'.format(module_name, self)

        # send information of getting model
        if get_rank() == 0:
            send_async({'name': "spring.models", 'action': 'get', 'model_name': module_name})

        return self[module_name]

    def build(self, cfg):
        """
        Arguments:
            cfg: dict with ``type`` and ``kwargs``
        """
        obj_type = cfg['type']
        obj_kwargs = cfg.get('kwargs', {})

        # send information of building model
        if get_rank() == 0:
            send_async({'name': "spring.models", 'action': 'build', 'model_name': obj_type,
                        'model_kwargs': obj_kwargs})

        if obj_type not in self:
            assert obj_type in self, '{} is not supported, avaiables are:{}'.format(obj_type, self)

        build_fn = self[obj_type]
        return build_fn(**obj_kwargs)

    def query(self):
        return self.keys()