"""Assorted utility functions"""

import os
import sys
from contextlib import contextmanager


# NOTE: This might be overkill if we always assume that this script is in the utils/ folder. But, it seems to work
def find_toplevel_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(
        current_dir
    ):  # Check if we've reached the root
        if "setup.py" in os.listdir(current_dir) or ".git" in os.listdir(current_dir):
            # Found the top-level package directory
            return current_dir
        current_dir = os.path.dirname(current_dir)
    # Top-level directory not found, handle appropriately
    raise RuntimeError("Top-level directory not found")


def find_assets_dir():
    assets_dir = os.path.join(find_toplevel_dir(), "oscbf/assets/")
    if not os.path.exists(assets_dir):
        raise RuntimeError("Assets directory not found")
    return assets_dir


# The motivation for this is Pybullet prints a ton of logging info when it launches, but I don't necessarily want this
# printed out. See the following stack overflow thread for the source:
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/17954769#17954769
@contextmanager
def stdout_redirected(to: str = os.devnull):
    """Temporarily redirects `sys.stdout` to the specified file

    This context manager is useful for silencing output or redirecting it to a
    file or other writable stream during the execution of a code block.

    Example:
        ```
        import os
        with stdout_redirected(to=filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")
        ```

    Args:
        to (str): The target file where stdout should be redirected.
            Defaults to `os.devnull` (silencing output).
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
