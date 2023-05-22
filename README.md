# CPT
Cross-protein transfer learning for variant effect prediction

This repository contains the codes and data for reproducing main results from the manuscript "Cross-protein transfer learning substantially improves zero-shot prediction of disease variant effects".

`analysis.ipynb`: Jupyter notebook for the main analyses.

`CPT/`: Python files for models and utility functions.

`data/`: Data necessary to train and evaluate the models.

We also provide pre-computed CPT-1 scores for 18,602 human proteins at
1. [Zenodo](https://doi.org/10.5281/zenodo.7954657)
2. [Huggingface](https://huggingface.co/spaces/songlab/CPT) (an interactive app to visualize and download individual proteins)

If the user would like to generate whole-proteome predictions with the trained model by themselves, the feature matrices are available upon request.

## Citation

Jagota, M.\*, Ye, C.\*, Rastogi, R., Albors, C., Koehl, A., Ioannidis, N., and Song, Y.S.&dagger;<br>
"Cross-protein transfer learning substantially improves zero-shot prediction of disease variant effects", bioRxiv (2022)

\*These authors contributed equally to this work. <br>
&dagger;To whom correspondence should be addressed:  yss@berkeley.edu

DOI: https://doi.org/10.1101/2022.11.15.516532
