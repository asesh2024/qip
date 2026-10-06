# Quantum Classification Under Noise: A Comparative Study of Encoding Strategies

Basis, Angle and Entangling (CNOT-correlated) encodings of a two-qubit, four-class problem,
simulated as exact density matrices in NumPy (no Qiskit), with Random Forest + Platt calibration
and confidence-gated LIME-style surrogates.

## Run
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python src/pipeline.py --quick --dataset_path data/quantum_dataset_10000.csv --output_dir outputs/test   # ~1 min smoke test
    python src/pipeline.py --dataset_path data/quantum_dataset_10000.csv --output_dir outputs/run4          # full run, ~1-2 min

## Data
`data/quantum_dataset_10000.csv`: 4,000 base inputs (`feature_a`, `feature_b`, `label`), i.i.d. uniform on [0, pi]^2.
First 2,000 rows are training inputs, last 2,000 are test inputs. Each circuit is run for 5 shots, and each shot is one instance.

## Outputs (`outputs/run4/`)
`tables/` (CSV and LaTeX for every paper table), `figures/`, `run_manifest.json`.
Models, posterior arrays and predictions are not committed.

## Notes
- Results come from one seed pair (train 42, test 100); variation across seeds was not measured.
- `legacy/pipeline_original.py` is the earlier script, kept for provenance. It is NOT the code behind the current paper.
