"""Create a student notebook from a tagged instructor notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def export_student_notebook(source: Path, destination: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    student_cells = []

    for cell in notebook.cells:
        tags = set(cell.metadata.get("tags", []))

        if "solution" in tags:
            continue

        if "hide" in tags:
            # Preserve the instructor-generated output but remove the code itself.
            cell.source = ""
            cell.execution_count = None
            cell.metadata.pop("jupyter", None)
        elif cell.cell_type == "code":
            # Avoid leaking stale results from non-hidden instructor cells.
            cell.outputs = []
            cell.execution_count = None

        student_cells.append(cell)

    notebook.cells = student_cells
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove solution cells and hide tagged code while keeping its output."
    )
    parser.add_argument("source", type=Path, help="Instructor notebook")
    parser.add_argument("destination", type=Path, help="Student notebook")
    args = parser.parse_args()
    export_student_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
