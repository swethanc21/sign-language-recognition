from setuptools import setup, find_packages

setup(
    name="sign-language-detection",
    version="0.0.1",
    author="Ashutosh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)