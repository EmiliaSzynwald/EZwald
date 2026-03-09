# Source - https://stackoverflow.com/a/56130390
# Posted by AdamGold, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-09, License - CC BY-SA 4.0

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ezwald",
    version="0.6.0",
    author="Emilia Szynwald",
    author_email="emiliaszynwald@gmail.com",
    description="Evaluate electrostatic energy of multi-layer vand der Waals systems with geometry optimization capabilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EmiliaSzynwald/EZwald",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

