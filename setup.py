import sys
from setuptools import setup,find_packages

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

# base="Win32GUI" should be used only for Windows GUI app
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="value_investing_strategy",
    version="0.1",
    packages=find_packages(),
    package_data={'value_investing_strategy': ['scripts/*']},
    include_package_data=True,
    description="",
    install_requires=open('requirements.txt').readlines(),
    author="James Mortensen",
    autho_email="jmortens@syr.edu",
    url="https://github.com/jm0rt1/CSE682-Project"
)
