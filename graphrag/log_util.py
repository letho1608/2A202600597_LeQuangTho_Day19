import sys

def p(*args, **kwargs):
    """Print with auto-flush for real-time output"""
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)
