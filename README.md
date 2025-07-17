# reproducing PRECIOUS secondary analysis of treatment restrictions and outcomes

installing:

```bash
pip install -e .[dev]
```
# running the analysis

The analysis assumes the raw data is in `data/PRECIOUS_treatmnetrestrictions.sav`.

## prepare data

```bash
Rscript import.R
```

This will create a curated dataset in `data/df_curated.csv`.

## run the analysis

```bash
python analysis.py
```