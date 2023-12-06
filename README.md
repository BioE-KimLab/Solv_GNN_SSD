# Solv_GNN_SSD

## Python environment setup

The 'tf24gpu.yml' file contains the information about dependencies required to run model training and prediction.

```python
conda env create -f tf24gpu.yml
```

```python
conda activate tf24gpu
```

## Run prediction

First, create 'molecules_to_predict.csv' file that contains the following columns:

- Each row: solute-solvent pairs whose Gibbs free energy of solvation needs to be predicted
- 'solute' column: SMILES string of solute
- 'solvent' column: SMILES string of solvent
- DGsolv: Experimental Gibbs free energy of solvation of a given solute-solvent pair (if available)

in the directory where 'main.py' is located.
Next, run

```python
python main.py -predict_df -modelname SSD_models/student35
```

Then, a user can find 'molecules_to_predict_results.csv' file which contains the predicted Gibbs free energies of solvation.
