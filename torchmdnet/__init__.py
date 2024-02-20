import importlib.metadata
import subprocess

try:
    __version__ = importlib.metadata.version("torchmd-net")
except importlib.metadata.PackageNotFoundError:
    try:
        __version__ = (
            subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
            .strip()
            .decode("utf-8")
        )
    except:
        print("Failed to retrieve the current version, defaulting to 0")
        __version__ = "0"
