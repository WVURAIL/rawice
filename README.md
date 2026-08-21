# rawice

Read RAW ADC snapshots from ICE boards and produce clock-stability and raw
acquisition diagnostics.

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

## Install dependencies

Use a virtual environment. The core reader and plotting library require:

```bash
python -m pip install --requirement requirements.txt
```

For the historical notebooks and the optional Allan-deviation method:

```bash
python -m pip install --requirement requirements-notebooks.txt
jupyter notebook
```

`allantools` is imported lazily, inside the one method that computes an
overlapping Allan deviation, so everything else works without it.

The separate diagnostics module and CLI require their full dependency set:

```bash
python -m pip install --requirement requirements-diagnostics.txt
python3 raw_acq_diagnostics/raw_acq_cli.py --help
```

## Core usage

The coordinate order is `[crate, slot, input]`:

```python
from rawice import raw_acq

acquisition = raw_acq("/path/to/acquisition.h5")
single_input = acquisition.check_input([0, 1, 2])
single_input.inspect_maser()
```

Acquisition arrays now belong to each `raw_acq` instance. For compatibility
with existing notebooks, class-level access such as `raw_acq.timestream` and
`raw_acq.check_input(...)` still refers to the most recently loaded file. New
code should use the instance form so loading another file cannot change which
data an analysis helper reads.

Expected HDF5 inputs contain `crate`, `slot`, `adc_input`, `timestamp`, and
`timestream` datasets, plus `index_map/timestream`. The `timestamp` records
must contain `fpga_count` and Unix `ctime` fields.

## Diagnostics CLI

Loading diagnostics by date requires the raw acquisition root. Pass it on the
command line or set `RAW_ACQ_DIR`:

```bash
python3 raw_acq_diagnostics/raw_acq_cli.py plot-summed-spectrum \
  --raw-acq-dir /path/to/raw_acq \
  --plot-dir .
```

Email delivery is optional. Supply credentials through
`RAW_ACQ_EMAIL_USERNAME` and `RAW_ACQ_EMAIL_APP_PASSWORD`; do not place an app
password in source code.

## Tests

The tests use small synthetic HDF5 files and do not require observatory data:

```bash
python -m pip install --requirement requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -s -p no:cacheprovider
```

## Current limitations

- No observatory acquisition data is distributed here. Several historical
  notebooks contain site-specific `/home/observer/...` paths and saved output;
  they are research records, not automated tests.
- Some notebooks use historical kernels (`py37mkl`, `py38`, or `chimefrb`) and
  may need a compatible environment or manual path updates.
- The legacy curve-fit `tau_shift` calculation applies the final fitted clock
  stability to every fitted phase. That scientific convention is preserved
  pending domain validation rather than changed during maintenance.
- Date-based diagnostics assume the existing `*_gbo_rawadc*` directory naming
  convention. Direct filename loading does not require `raw_acq_dir`.

## Downstream snapshot

[`WVURAIL/DigitalNoiseSource`](https://github.com/WVURAIL/DigitalNoiseSource)
vendors `rawice.py` from commit
[`d6e3c3c0b650c978962921d44e5a54aaf6967583`](https://github.com/WVURAIL/rawice/commit/d6e3c3c0b650c978962921d44e5a54aaf6967583)
for its recorded PEACC analysis. That snapshot is intentionally pinned; current
`rawice.py` changes must be reviewed against the downstream notebooks and must
not be copied or merged automatically.

## Contributors and license

Git history records contributions to `rawice.py` by Pranav Sanghavi and Audrey
Zinn, and attributes the original diagnostics module to Bridget Andersen. It
preserves the available attribution record.

This repository is distributed under the [MIT License](LICENSE). The complete
standard text replaces the truncated placeholder added during 2026
maintenance and retains attribution to the rawice contributors.
