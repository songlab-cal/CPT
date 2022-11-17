# CPT
Cross-protein transfer learning for variant effect prediction

This repository contains the variant effect preditions of CPT-1 for 18,602 human proteins, initially released with the manuscript "Cross-protein transfer learning substantially improves zero-shot prediction of disease variant effects". The proteins are split into two sets.

`transter_proteome_eve`: Proteins in the EVE set ([Frazer et al., 2021](https://www.nature.com/articles/s41586-021-04043-8))

`transfer_proteome_xgimpute`: Proteins not in the EVE set. Predictions for these proteins use imputed values for features depending on the EVE MSA.

## Citation

Jagota, M.\*, Ye, C.\*, Rastogi, R., Albors, C., Koehl, A., Ioannidis, N., and Song, Y.S.&dagger;<br>
"Cross-protein transfer learning substantially improves zero-shot prediction of disease variant effects", bioRxiv (2022)

\*These authors contributed equally to this work. <br>
&dagger;To whom correspondence should be addressed:  yss@berkeley.edu

DOI: https://doi.org/10.1101/2022.11.15.516532
