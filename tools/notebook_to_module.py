import json
from pathlib import Path
import argparse


def notebook_to_script(nb_path: Path, output_path: Path) -> None:
    data = json.loads(nb_path.read_text())
    lines = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            for line in cell.get("source", []):
                if line.lstrip().startswith("!") or line.lstrip().startswith("%"):
                    # Skip IPython magic or shell commands
                    continue
                lines.append(line)
            if lines and not lines[-1].endswith("\n"):
                lines.append("\n")
            lines.append("\n")
    output_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract code cells from a Jupyter notebook and save as a Python script.")
    parser.add_argument("notebook", type=Path, help="Notebook file")
    parser.add_argument("output", type=Path, nargs="?", help="Output python file. Defaults to <notebook>.py")
    args = parser.parse_args()
    output = args.output if args.output else args.notebook.with_suffix(".py")
    notebook_to_script(args.notebook, output)


if __name__ == "__main__":
    main()
