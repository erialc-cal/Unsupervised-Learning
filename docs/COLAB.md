# Using the course notebooks in Google Colab

Colab can load a notebook from GitHub, but it does not automatically download the
rest of the repository. To get started, you will need to follow the next steps to ensure that the checks and data are properly loaded for the notebook exercises.

## 1. Open a notebook from GitHub

Each lab can be opened on Colab from GitHub via its GitHub url. For instance, 
```text
https://colab.research.google.com/github/OWNER/REPOSITORY/blob/main/notebooks/00_dummy_exercise.ipynb
```

You can also open Colab, choose **File > Open notebook > GitHub**, paste the
repository URL, and select a notebook.

## 2. Save your own working copy

Choose **File > Save a copy in Drive** before doing the exercises. The copy of your current work is then stored *locally* on your own Google Drive. **Important to note** that the cloned repository described
below belongs to the temporary Colab runtime and is not where students should save
their answers, this means if your session times out, you will need to clone the repository again. 

## 3. Clone the supporting repository

Run this setup cell near the top of the notebook. 
```python
from pathlib import Path
import os
import subprocess

REPOSITORY_URL = "https://github.com/OWNER/REPOSITORY.git"
REPOSITORY_NAME = "REPOSITORY"
REPOSITORY_DIR = Path("/content") / REPOSITORY_NAME

if REPOSITORY_DIR.exists():
    subprocess.run(
        ["git", "-C", str(REPOSITORY_DIR), "pull", "--ff-only"],
        check=True,
    )
else:
    subprocess.run(
        ["git", "clone", "--depth", "1", REPOSITORY_URL, str(REPOSITORY_DIR)],
        check=True,
    )

os.chdir(REPOSITORY_DIR / "notebooks")
print(f"Working directory: {Path.cwd()}")
```

The equivalent one-time shell commands are:
```bash
!git clone --depth 1 https://github.com/OWNER/REPOSITORY.git /content/REPOSITORY
%cd /content/REPOSITORY/notebooks
```
The Python version above is preferred because rerunning it updates an existing
clone instead of failing because the directory already exists.

## 4. Import and run the checks

After the setup cell finishes, imports work exactly as they do locally:
```python
from course_checks import grader
grader.check("exercise_1", center_value)
```

Run notebook cells from top to bottom. If Colab reconnects with a new runtime,
rerun the repository setup cell before continuing.

## 5. Save or submit the completed notebook

Colab's runtime filesystem, including `/content/REPOSITORY`, is temporary. Saving
the notebook to Google Drive preserves notebook edits, but it does not preserve
arbitrary files created inside the cloned repository.

To properly save your notebooks, use **File > Download > Download .ipynb**. If an exercise creates additional files that you will reuse, download those files separately before the runtime
ends, and adjust the file path accordingly.



