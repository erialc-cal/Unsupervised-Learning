# STAT GR4244 exercise notebooks

Clone the course repository, enter this directory, and launch Jupyter from here:

```bash
git clone https://github.com/erialc-cal/STAT_CLASSES.git
cd STAT_CLASSES/STATGR4244_F2026
jupyter lab
```

The local `course_checks` module lives beside the notebooks, so no separate check
package needs to be installed. Exercise checks use this small API:

```python
from course_checks import grader
grader.check("exercise_1", center_value)
```

Repository organization is intentionally simple:

```text
data/                     Course datasets
docs/                     Student setup guides
notebooks/                Labs and exercises
notebooks/course_checks/  Public notebook checks only
scripts/                  Instructor data-preparation utilities
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

The included checks are public practice checks. Separate hidden tests can be
added to the eventual submission grader. Python dependencies used by the actual
course exercises will be documented separately in an environment file.

For the hosted workflow, see [Using the course notebooks in Google Colab](docs/COLAB.md).
