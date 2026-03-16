from setuptools import setup

setup(
    name="gpushare",
    version="1.0.0",
    description="Python client for gpushare remote GPU access",
    packages=["gpushare"],
    python_requires=">=3.7",
    install_requires=["numpy"],
)
