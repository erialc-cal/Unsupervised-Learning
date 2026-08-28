# STAT GR4244/5244: Unsupervised Learning

Clone the course repository, enter this directory, and launch Jupyter from here:

```bash
git clone https://github.com/erialc-cal/Unsupervised-Learning.git
cd Unsupervised-Learning
jupyter lab
```

The repository's organization is as follows: 

```text
data/                     Course datasets
docs/                     Course documents (including instructions for setting up lab materials)
notebooks/                Labs and exercises
scripts/                  Other useful python scripts and back-up materials
```

## Local Python environment
Miniconda and Anaconda users can create the course environment with:
```bash
conda env create -f environment.yml
conda activate statgr4244
python -m jupyter lab
```

Plain Python users can create an isolated virtual environment with:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Do not use `sudo pip install` or install course dependencies into the operating
system's Python. Lab Session 0 contains the complete macOS, Linux, Windows,
Conda, and Colab guidance.

If you wish to use Google Colab instead of running notebooks locally, see [Using the course notebooks in Google Colab](docs/COLAB.md).
