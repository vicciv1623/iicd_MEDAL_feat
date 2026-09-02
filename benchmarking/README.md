## MEDAL feature attributions 
- found in **feature_attribution.ipynb**
- feature attributions included: 
  - Deep Lift
  - Feature ablation
  - Integrated gradients
  - Occlusion
  - Kernel SHAP

## Benchmarking methods
- [Gradient-based explanation for non-linear non-parametric dim red](https://link.springer.com/article/10.1007/s10618-024-01055-6)
  - found in **grad_expP.ipynb**
  - when running this file make sure you import the editable version of scikit then replace `sklearn/manifold/_t_sne.py` with the one here
- LRP
  - found in **lrpP.ipynb** and **lrp/** 
- [LXDR](https://arxiv.org/abs/2204.14012)
  - found in **lxdr_mnistP.ipynb** and **lxdr.py**
- [XGBOOST on UMAP + SHAP](https://journals.sagepub.com/doi/10.1089/cmb.2022.0366?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed)
   - Note that we use t-SNE instead of UMAP
   - found in **xgboost_shapP.ipynb**
 
## MEDAL-o
- contains the MEDAL object trained on MNIST dataset (784) using t-SNE as the teacher
