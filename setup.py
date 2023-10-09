from setuptools import setup, find_namespace_packages

version = "0.0.2"

setup(
    name="metaflow-tensorflow",
    version=version,
    description="An EXPERIMENTAL TensorFlow decorator for Metaflow",
    author="Eddie Mattia",
    author_email="eddie@outerbounds.co",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[
         "metaflow"
    ]
)