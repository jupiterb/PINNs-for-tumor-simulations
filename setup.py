from setuptools import setup, find_namespace_packages

phynn_requirements = ["torch", "lightning", "h5py", "wandb", "numpy==1.23.1"]


setup(
    name="physics-learning",
    version="0.0.0",
    description="",
    packages=find_namespace_packages(),
    extras_require={"phynn": phynn_requirements},
)
