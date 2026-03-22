from .base_runner import BaseFederatedRunner


def build_runner(args):
    return BaseFederatedRunner(args)
