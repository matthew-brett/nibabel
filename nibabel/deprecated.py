import warnings

class ModuleProxy(object):
    def __init__(self, module_name):
        self._module_name = module_name

    def __hasattr__(self, key):
        mod = __import__(self._module_name, fromlist=[''])
        return hasattr(mod, key)

    def __getattr__(self, key):
        mod = __import__(self._module_name, fromlist=[''])
        return getattr(mod, key)

    def __repr__(self):
        return "<module proxy for {0}".format(self._module_name)


class FutureWarningMixin(object):
    warn_message = 'This class will be removed in future versions'
    def __init__(self, *args, **kwargs):
        warnings.warn(self.warn_message,
                      FutureWarning,
                      stacklevel=2)
        super(FutureWarningMixin, self).__init__(*args, **kwargs)
