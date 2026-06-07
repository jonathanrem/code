# Dataset

The dataset used in this project cannot be shared in this repository for patient confidentiality reasons. It is derived from a private multi-center clinical cohort and contains protected health information.

## Features

| Feature | Description |
|---|---|
| `age` | Patient age (years) |
| `psa` | Prostate-Specific Antigen value (ng/mL) |
| `psa_density` | PSA density (PSA / prostate volume) |
| `prostate_volume` | Prostate volume (mL) |
| `clinical_stage` | Clinical stage (0 = T1c, 1 = T2+) |
| `pirads` | PI-RADS v2 score of the index lesion (1–5) |
| `diameter` | Diameter of the index lesion (mm) |
| `nb_susp_lesions` | Number of suspicious lesions on MRI |
| `suspicious_trus` | Suspicious finding on transrectal ultrasound (0/1) |
| `family_history` | Family history of prostate cancer (0/1) |
| `prev_neg_trus_biopsy` | Previous negative TRUS-guided biopsy (0/1) |
| `contralateral_suspicious` | Contralateral side suspicious on MRI (0/1) |
| `contralateral_pirads` | PI-RADS score of the contralateral lesion (1–5) |
| `contralateral_diameter` | Diameter of the contralateral lesion (mm) |
| `epe` | Suspected extraprostatic extension on MRI (0/1) |
| `ant` | Lesion location — anterior (0/1) |
| `mid` | Lesion location — middle (0/1) |
| `post` | Lesion location — posterior (0/1) |
| `base` | Lesion location — base (0/1) |
| `median` | Lesion location — median (0/1) |
| `apical` | Lesion location — apical (0/1) |
| `outcome` | **Target variable** — clinically significant prostate cancer (0/1) |

## Data splits

The dataset was split by center to ensure geographic external validation: patients from centers unseen during training form the test set (`test_df.csv`), and the remaining patients form the training set (`train_df.csv`).
