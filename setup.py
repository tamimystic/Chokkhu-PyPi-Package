import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description=f.read()

__version__="0.3.23"

REPO_NAME="Chokkhu-PyPi-Package"
AUTHOR_USER_NAME="tamimystic"
AUTHOR_EMAIL="hossainsmtamim@gamil.com"
SRC_REPO="chokkhu"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A complete data preparation and EDA toolkit for Image and Tabular datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Source Code": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "numpy>=1.23",
        "Pillow>=9.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.2",
            "flake8>=6.1",
            "mypy>=1.5",
            "black>=23.3",
            "isort>=5.12"
        ]
    }
)
