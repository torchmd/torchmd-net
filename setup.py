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

setup(
    name="torchmd-net",
    version=version,
    packages=find_packages(),
    package_data={"torchmdnet": ["neighbors/neighbors*", "neighbors/*.cu*"]},
    include_package_data=True,
    entry_points={"console_scripts": ["torchmd-train = torchmdnet.scripts.train:main"]},
)
