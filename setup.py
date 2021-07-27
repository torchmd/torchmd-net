import subprocess
from setuptools import setup, find_packages

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="torchmd-net",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
)
