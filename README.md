# Repository for Master's Thesis

A codebase accompanying the Master's thesis on evaluating fairness across differential privacy techniques in machine learning. The repository includes:
- Data loaders for the `adult`, `acs_income`, and `acs_employment` datasets.
- Model definitions for each dataset.
- Implementations of privacy mechanisms: DP-SGD, PATE, and a novel smooth-sensitivity-based approach.
- Experiment scripts for training, evaluation, and fairness metric computation.

## Repository Structure

- `src/`: Source code for data loading, model architectures, privacy wrappers, training, and evaluation.
- `experiments/`: Configuration files and scripts to run cross-dataset, cross-method experiments.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
- `results/`: Generated outputs, logs, and summary CSVs of experiment results.
- `README.md`: This overview document.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd haashim_repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install datasets. For AIF360 fairness datasets, see specific instructions in their corresponding READMEs in the AIF360 source.
4. Run experiments:
   TODO
