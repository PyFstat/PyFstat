import os


def set_up_matplotlib_defaults():
    """Sets some defaults for matplotlib plotting."""
    plt.rcParams["axes.formatter.useoffset"] = False


def X_less_plt():

    if "DISPLAY" in os.environ:
        import matplotlib.pyplot as plt
    else:
        # logger.info(
        #    'No $DISPLAY environment variable found, so importing \
        #              matplotlib.pyplot with non-interactive "Agg" backend.'
        # )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    return plt


if __name__ == "__main__":
    plt = X_less_plt()
    import matplotlib

    print(matplotlib.get_backend())

    del os.environ["DISPLAY"]
    plt = X_less_plt()
    # import matplotlib
    print(matplotlib.get_backend())


def initializer(func):
    """Decorator to automatically assign the parameters of a class instantiation to self."""
    argspec = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(argspec.args[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(argspec.args), reversed(argspec.defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper
