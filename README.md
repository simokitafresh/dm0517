# dm0517

This repository contains Jupyter notebooks. The notebook `notebook/2025_04_29` stores a trading strategy implementation. To make the code easier to reuse, you can convert the notebook into a Python script.

## Converting the notebook

Use the helper script in `tools/notebook_to_module.py` to extract the code cells from the notebook:

```bash
python tools/notebook_to_module.py notebook/2025_04_29
```

This command creates `notebook/2025_04_29.py`. The generated file is a plain Python module with all code cells from the notebook. You can import functions and classes from this script or run it directly.

## Notes for beginners

- The script simply copies the code cells. Commands starting with `!` (like `!pip install`) are also included. Remove or edit them if you plan to run the module outside Jupyter.
- You can read the generated `.py` file to understand how the notebook is structured. Learning by reading the code is a good first step toward modularisation.
