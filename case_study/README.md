Link to [paper](https://www.cell.com/cell/fulltext/S0092-8674(18)31178-4)

- single cell data of 33 melanoma tumors from 2 studies and cell lines IGR39, A2058 and UACC62
  - immune cells study: use 10-PCs after reducing to 194 known biomarkers + t-SNE + DBScan on the t-SNE embedding
- **Process_Data.ipynb**: the processing code to create the data as explained by the paper
- The way the data is stored via anndata format: 
  - adata.X (obs:cells x var:genes matrix of normalized expression) 
  - adata.var: gene-specific metadata
  - adata.obs: cell-specific metadata 
  - adata.layers[‘counts’] : contains raw counts of data
  - adata.layers[‘tpm’] : contains TPM data
- **medal_immune_meta/**: folder containing MEDAL training
- **data/**: folder containing annotations and miscellaenous meta data sourced from the paper; do not need to worry as the **immune.ipynb** will handle all the stuff
- **cell.annotations.csv**: sample meta annotations
- **immune.ipynb**: where the feature attribution analysis is taking place
- 
