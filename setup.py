import setuptools

with open('README.md','r',encoding="utf-8") as f:
    long_description=f.read()




__version__= "0.0.0"

REPO_NAME='e2e-mlops'
AUTHOR_USER_NAME="Aghilas.Sini"
SRC_REPO = "mlProject"
AUTHOR_EMAIL="aghilas.sini@univ-lemans.fr"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='a small  python package for  mlops ',
    long_description=long_description,
    long_destription_content='text/markdown',
    url="https://git.univ-lemans.fr/Aghilas.Sini/e2e-mlops",
    project_urls={

        "Bug Tracker":"https://git.univ-lemans.fr/Aghilas.Sini/e2e-mlops/-/issues",

    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where='src'),


)
