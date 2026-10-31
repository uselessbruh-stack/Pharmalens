# Frozen affinity benchmark

Defaults were selected using validation only. Test scores did not change the selections.

| Dataset | Endpoint | Model | Seeds | Mean validation RMSE | Mean test RMSE |
|---|---|---|---:|---:|---:|
| NonKinasePanel | Ki | xgboost | 1 | 0.7589 | 0.7488 |
| NonKinasePanel | Ki | deeptda | 1 | 0.8054 | 0.7765 |
| NonKinasePanel | Ki | gnn | 1 | 0.7908 | 0.7600 |
| NonKinase_ion_channel | Kd | xgboost | 1 | 0.8839 | 1.0506 |
| NonKinase_ion_channel | Kd | deeptda | 1 | 1.0324 | 1.1198 |
| NonKinase_ion_channel | Kd | gnn | 1 | 0.9979 | 1.1476 |
| NonKinase_protease | Kd | xgboost | 1 | 0.8749 | 0.8346 |
| NonKinase_protease | Kd | deeptda | 1 | 0.8905 | 0.9191 |
| NonKinase_protease | Kd | gnn | 1 | 0.9308 | 0.9754 |
| NonKinase_nuclear_receptor | Kd | xgboost | 3 | 0.7791 | 0.8453 |
| NonKinase_nuclear_receptor | Kd | deeptda | 3 | 0.7517 | 0.8484 |
| NonKinase_nuclear_receptor | Kd | gnn | 3 | 0.7716 | 0.8582 |
| NonKinase_gpcr | Kd | xgboost | 1 | 0.8886 | 0.9425 |
| NonKinase_gpcr | Kd | deeptda | 1 | 0.9323 | 0.8751 |
| NonKinase_gpcr | Kd | gnn | 1 | 0.8944 | 0.8702 |
| NonKinase_nuclear_receptor | Ki | xgboost | 3 | 0.8052 | 0.7223 |
| NonKinase_nuclear_receptor | Ki | deeptda | 3 | 0.7754 | 0.7649 |
| NonKinase_nuclear_receptor | Ki | gnn | 3 | 0.7823 | 0.7867 |
| NonKinase_transporter | Ki | xgboost | 1 | 0.6695 | 0.6694 |
| NonKinase_transporter | Ki | deeptda | 1 | 0.6978 | 0.6682 |
| NonKinase_transporter | Ki | gnn | 1 | 0.6960 | 0.6655 |
| NonKinase_other_enzyme | Kd | xgboost | 1 | 1.0967 | 1.1403 |
| NonKinase_other_enzyme | Kd | deeptda | 1 | 1.1083 | 1.1625 |
| NonKinase_other_enzyme | Kd | gnn | 1 | 1.0952 | 1.1680 |
| NonKinase_ion_channel | Ki | xgboost | 1 | 0.6241 | 0.6026 |
| NonKinase_ion_channel | Ki | deeptda | 1 | 0.6237 | 0.6613 |
| NonKinase_ion_channel | Ki | gnn | 1 | 0.6469 | 0.6657 |
| NonKinaseBroad | Kd | xgboost | 1 | 1.0151 | 1.0388 |
| NonKinaseBroad | Kd | deeptda | 1 | 1.0373 | 1.0731 |
| NonKinaseBroad | Kd | gnn | 1 | 1.0433 | 1.0677 |
| NonKinase_protease | Ki | xgboost | 1 | 0.7267 | 0.7424 |
| NonKinase_protease | Ki | deeptda | 1 | 0.7507 | 0.7460 |
| NonKinase_protease | Ki | gnn | 1 | 0.7490 | 0.7520 |
| NonKinase_other_enzyme | Ki | xgboost | 1 | 0.7202 | 0.7408 |
| NonKinase_other_enzyme | Ki | deeptda | 1 | 0.7570 | 0.7727 |
| NonKinase_other_enzyme | Ki | gnn | 1 | 0.7610 | 0.7762 |
| NonKinase_gpcr | Ki | xgboost | 1 | 0.7143 | 0.7243 |
| NonKinase_gpcr | Ki | deeptda | 1 | 0.7280 | 0.7317 |
| NonKinase_gpcr | Ki | gnn | 1 | 0.7356 | 0.7380 |
| NonKinaseBroad | Ki | xgboost | 1 | 0.7601 | 0.7633 |
| NonKinaseBroad | Ki | deeptda | 1 | 0.7593 | 0.7645 |
| NonKinaseBroad | Ki | gnn | 1 | 0.7573 | 0.7519 |
| KIBA | KIBA | xgboost | 1 | 0.4278 | 0.4189 |
| KIBA | KIBA | deeptda | 1 | 0.3863 | 0.3730 |
| KIBA | KIBA | gnn | 1 | 0.3860 | 0.3689 |

These are identity-grouped random splits. Family views overlap broad datasets; they are not independent external evidence. Most tasks have one model seed. The nuclear-receptor tasks have three. Neural gains are task-dependent and some validation gains do not carry over to test.
