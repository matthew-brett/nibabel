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
