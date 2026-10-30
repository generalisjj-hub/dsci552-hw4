# DSCI 552 Homework 4

The completed notebook is `notebook/Jiang_Jun_HW4.ipynb`; the AReM data is in `data/AReM/`.

From the repository root, use Python 3.12 and run:

```sh
uv sync --frozen
uv run jupyter nbconvert --to notebook --execute --inplace notebook/Jiang_Jun_HW4.ipynb --ExecutePreprocessor.timeout=1800
```

Alternatively, install `requirements.txt` in a Python 3.12 environment and run the notebook with that environment's Jupyter kernel.
