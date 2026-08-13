# Experiment data example

This directory is a small, self-contained example of the physical experiment
format consumed by `qft_dynamic.experiment_data`. It is intended for interface
testing and coordination with the physical experiment team, not for scientific
analysis.

## Contents

```text
manifest.toml  Dataset location, filename metadata, and qubit mapping
readout_calibration.toml  Independent readout calibration by physical qubit
raw/           Four two-qubit NPZ runs with eight shots per run
```

Each NPZ contains classified `int32` bit arrays and matching `complex128` IQ
arrays for physical qubits `Q10` and `Q11`:

```text
Q10_0, Q10_0_iq, Q11_0, Q11_0_iq
```

The saved physical columns are `[Q10, Q11]`, while
`logical_from_physical = [1, 0]`. Consequently, the loader constructs logical
MSB-first bit rows as `[Q11, Q10]`. This deliberately non-identity mapping makes
the example sensitive to physical-to-logical conversion errors.

The filename regex extracts `k` as string metadata. The expected logical
counters are:

| `k` | Expected logical counter |
|---:|---|
| `00` | `{0: 6, 1: 1, 2: 1}` |
| `01` | `{1: 6, 2: 1, 3: 1}` |
| `02` | `{0: 1, 2: 6, 3: 1}` |
| `03` | `{0: 1, 1: 1, 3: 6}` |

## Validate the example

From the repository root, normalize the physical files to JSON:

```bash
uv run python devtools/normalize_experiment_data.py \
  data/experiments_example/manifest.toml \
  /tmp/experiment-example-counts.json
```

The counter JSON can be restored directly as the common Pydantic model:

```python
from pathlib import Path

from qft_dynamic.experiment_data import ExperimentDataset

dataset = ExperimentDataset.load(Path("/tmp/experiment-example-counts.json"))
first_run = dataset.groups[0].runs[0]

assert first_run.metadata == {"k": "00"}
assert first_run.run_id == first_run.source_ref
assert first_run.num_shots == 8
assert dataset.schema_version == 2
assert dataset.producer == "physical_npz"
```

The counter JSON preserves measured integer counts as the canonical normalized
artifact. Consumers derive probabilities in memory when needed. The calibration
filename remains a generic group attribute for callers that explicitly invoke
the library-level mitigation operation.
