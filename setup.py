import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description=f.read()

__version__="0.3.0"

REPO_NAME="Chokkhu-PyPi-Package"
AUTHOR_USER_NAME="tamimystic"
AUTHOR_EMAIL="hossainsmtamim@gamil.com"
SRC_REPO="chokkhu"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python image classification package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "tensorflow>=2.12",
        "numpy>=1.23",
        "Pillow>=9.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.2",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "scikit-image>=0.20.0",
        "imagehash>=4.3.0"
    ]
)
