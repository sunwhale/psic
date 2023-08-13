import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="psic",
    version="0.1.1",
    author="Jingyu Sun",
    author_email="sun.jingyu@outlook.com",
    description="Packing Spheres In Cube",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunwhale/psic",
    project_urls={
        "Bug Tracker": "https://github.com/sunwhale/psic/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        'numpy',
        'scipy'
    ],
)