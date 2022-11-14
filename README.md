# CPT
Cross-protein transfer learning for variant effect prediction

This repository contains the variant effect preditions of CPT-1 for 18,602 human proteins, initially released with the manuscript "Cross-protein transfer learning substantially improves zero-shot prediction of disease variant effects". The proteins are split into two sets.

`transter_proteome_eve`: Proteins in the EVE set ([Frazer et al., 2021](https://www.nature.com/articles/s41586-021-04043-8))

`transfer_proteome_xgimpute`: Proteins not in the EVE set. Predictions for these proteins used imputed values for features depending on the EVE MSA.
