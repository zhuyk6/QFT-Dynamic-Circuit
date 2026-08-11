# Experiment data normalization

The experiment-data package has one responsibility: convert physical
single-shot files into logical integer counters. It does not select or compute
metrics.

```text
physical NPZ + manifest
  -> classified physical-bit columns
  -> physical-to-logical column mapping
  -> MSB-first logical Counter[int]
  -> ProbabilityExperimentDataset
  -> optional independent readout mitigation
```

## Manifest

The current six-qubit data is described by
`configs/experiments/qft_fidelity_6q.toml`. The manifest describes how files are
located and converted into logical outputs; it does not define metrics.

### Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Manifest schema version. The current and only supported value is `1`. |
| `dataset_id` | Stable, human-readable identifier copied into normalized datasets. It need not match a directory name. |
| `experiment_type` | Free-form experiment family such as `process_fidelity`. Consumers may use it to select experiment-specific logic. |
| `num_qubits` | Number of logical output bits. Every group must list exactly this many physical qubits. |
| `bit_order` | Canonical logical integer convention. The current loader supports only `msb_first`. |
| `raw_data_dir` | Directory containing raw NPZ files. It may be absolute or relative to the manifest directory. |
| `filename_metadata_regex` | Regular expression applied to each filename stem. It must contain at least one named capture group. Captured text is preserved as run metadata. |
| `groups` | One or more file collections with their own glob, physical layout, logical mapping, and attributes. |

For `bit_order = "msb_first"`, a logical row `[b0, b1, ..., bn-1]` is converted
to the integer

```text
b0 * 2^(n-1) + b1 * 2^(n-2) + ... + bn-1.
```

The filename regex is searched in `Path.name.stem`, not in the full path. For
example, `_k(?P<k>\d+)` extracts `{"k": "02"}` from a filename ending in
`_k02.npz`. The loader does not interpret `k`, convert it to an integer, or
require a complete parameter sweep.

### Group fields

| Field | Meaning |
|---|---|
| `name` | Unique group label copied to the dataset. |
| `file_glob` | Glob evaluated inside `raw_data_dir`. A group must match at least one file. |
| `physical_qubits` | Physical qubit identifiers used to locate NPZ bit arrays and establish the physical-column order. Values must be unique. |
| `logical_from_physical` | Permutation of physical-column positions. Element `j` selects the physical column used as logical output column `j`. |
| `attributes` | Optional scalar or scalar-list context copied unchanged into the normalized group. The generic loader does not interpret it. |

The distinction between qubit identifiers and column positions is important.
For example:

```toml
physical_qubits = [10, 11]
logical_from_physical = [1, 0]
```

The loader first constructs physical columns `[Q10, Q11]`, then produces
logical columns `[Q11, Q10]`. `logical_from_physical` must therefore be a
permutation of `0..num_qubits-1`, not a list of physical qubit identifiers.

Attributes belong to downstream workflows. For example,
`completed_output_qubits` is interpreted by the stages plot, while `batch_size`
is only descriptive. Readout calibration is likewise not part of the fixed
manifest schema; a group opts into mitigation with a generic attribute:

```toml
[groups.attributes]
readout_calibration_file = "readout_calibration_batch13.toml"
```

This filename is only looked up when mitigation is explicitly requested.

Path-valued settings may be absolute or relative. Relative `raw_data_dir` and
`readout_calibration_file` values are resolved from the directory containing
the manifest, whose absolute source path is retained by the loaded
`ExperimentManifest` model but excluded from serialization.

### Raw NPZ contract

For every physical qubit in a group, an NPZ file must contain exactly one
non-IQ, one-dimensional binary array under `Q<id>` or a key beginning with
`Q<id>_`. Keys ending in `_iq` are ignored. All selected bit arrays must be
nonempty and have the same shot count. IQ arrays remain in the raw file and are
not copied into the normalized dataset.

Different groups should use non-overlapping globs unless duplicating a source
run is intentional. The loader treats each group independently and does not
deduplicate files across groups.

## Python API

```python
from pathlib import Path

from qft_dynamic.experiment_data import (
    ExperimentDataset,
    ProbabilityExperimentDataset,
    counts_to_probabilities,
    load_experiment_dataset,
)

dataset = load_experiment_dataset(
    Path("configs/experiments/qft_fidelity_6q.toml")
)
probabilities: ProbabilityExperimentDataset = counts_to_probabilities(dataset)

for group in dataset.groups:
    for run in group.runs:
        logical_counts = run.counts
        # Select TVD, fidelity, success probability, or plotting here.
```

Each `LogicalCountsRun` contains its logical `Counter[int]`, source filename, and
named string metadata extracted from the filename. The source path identifies
the run; there is no separate run ID. The loader does not apply readout
mitigation, calculate target probabilities, aggregate runs, or produce plotting
data.

Counter-to-probability conversion is generic and does not imply mitigation.
Mitigation is a subsequent Probability-to-Probability transform, so the
measured counters and unmitigated probabilities remain available unchanged.
Each referenced calibration TOML stores independent assignment fidelities by
physical-qubit number:

```toml
schema_version = 1

[qubits.13]
p0_given_0 = 0.9875
p1_given_1 = 0.9780
```

The mitigation step matches these entries to `physical_qubits` and applies
`logical_from_physical` before constructing the tensor-product assignment
matrix. It models independent single-qubit readout errors; correlated readout
errors are outside this format.

The normalized model supports JSON round trips directly:

```python
dataset.save(Path("results/qft-logical-counts.json"))
restored = ExperimentDataset.load(Path("results/qft-logical-counts.json"))
```

For a counter run, `num_shots` is derived and therefore not duplicated in its
JSON representation. Counter-to-probability conversion preserves it explicitly
on each probability run, allowing downstream analysis to retain shot-dependent
uncertainty information without requiring the Counter dataset.

## Optional normalized JSON

The generic normalization command writes probabilities by default. Counter and
readout-mitigated probability outputs are optional:

```bash
uv run python devtools/normalize_experiment_data.py \
  configs/experiments/qft_fidelity_6q.toml \
  --probability-output results/qft-logical-probabilities.json \
  --counter-output results/qft-logical-counts.json \
  --mitigation-output results/qft-logical-probabilities-mitigated.json
```

- `--probability-output` is required and produces a `ProbabilityExperimentDataset`.
- `--counter-output` optionally preserves the
intermediate `ExperimentDataset`; it is only needed by downstream work that requires integer shot counts.
- The optional mitigation output is also an ordinary `ProbabilityExperimentDataset`; mitigation is an operation, not a distinct representation. Omitting `--mitigation-output` does not look for a
calibration attribute.

Raw IQ arrays remain in the source NPZ files. They are not loaded because the current common representation starts from the already classified bit arrays.
