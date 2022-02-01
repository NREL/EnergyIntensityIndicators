"""
setup.py
"""

from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
from warnings import warn


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


description = ("Energy Intensity Indicators")

setup(
    name="NREL-EnergyIntensityIndicators",
    description=description,
    url="https://github.com/NREL/EnergyIntensityIndicators/",
    packages=find_packages(),
    package_dir={"EnergyIntensityIndicators": "EnergyIntensityIndicators"},
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    python_requires='>=3.6',
    extras_require={
        "dev": ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
