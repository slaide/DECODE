def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util
    assert sys.version_info>=(3,8) and sys.version_info<(3,9), "if this fails, add '?.so' file mentioned below for the different python version from https://anaconda.org/Turagalab/spline/files"
    __file__ = pkg_resources.resource_filename(__name__, 'spline.cpython-38-x86_64-linux-gnu.so')
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location(__name__,__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
__bootstrap__()
