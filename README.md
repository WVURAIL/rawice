# rawice

Read RAW ADC snapshots from ICE boards.

`rawice.py` is the library. The notebooks in this directory are the analyses
built on it, and each one starts with

```python
from rawice import *
```

so **`rawice.py` has to stay beside the notebooks.** It was briefly moved into
a `scripts/` directory in August 2026; Jupyter puts the notebook's own
directory on `sys.path` and nothing else, so every notebook stopped importing.
If you tidy the layout again, move the notebooks with it or add the new
location to `sys.path` in each one.

## What is here

| | |
|---|---|
| `rawice.py` | the `raw_acq` class, `analyse_maser`, and the curve fit for clock stability |
| `raw_acq_diagnostics/` | diagnostic plots for raw acquisitions, plus a CLI that can mail the results — Bridget Andersen, 2024 |
| `*.ipynb` | per-session analyses, named by date and frame rate |

## Running it

```bash
pip install numpy scipy matplotlib h5py allantools jupyter
jupyter notebook
```

`allantools` is imported lazily, inside the one method that computes an
overlapping Allan deviation, so everything else works without it.

The diagnostics CLI has its own dependencies:

```bash
pip install click python-dateutil pytz
python3 raw_acq_diagnostics/raw_acq_cli.py --help
```
