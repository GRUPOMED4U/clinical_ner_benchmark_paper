### Mean and STD metrics after 10 iterations

**1. Base**
|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |  0.108606   |              0.692487  |           0.7214    |       0.692875  |               0.682658 |           0.785696  |       0.730106  |
| std  |  0.00792812 |              0.0412459 |           0.0315137 |       0.0290931 |               0.032707 |           0.0175237 |       0.0199676 |

---

**2. With weighted loss** `test_metrics_001`

|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |   0.296936  |              0.748879  |           0.721645  |      0.728304   |              0.766157  |            0.752666 |      0.759157   |
| std  |   0.0289572 |              0.0226268 |           0.0209345 |      0.00912782 |              0.0149161 |            0.011507 |      0.00361029 |

---

**3. With weighted loss and 1 as minimal weight** `test_metrics_002`

|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |   0.326035  |              0.74314   |            0.73636  |       0.734083  |              0.757438  |           0.771034  |      0.763469   |
| std  |   0.0351608 |              0.0282968 |            0.020937 |       0.0114547 |              0.0292072 |           0.0203336 |      0.00791768 |

---

**4. With weighted loss and 1 as minimal weight and weights powered to 0.25** `test_metrics_003`

|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |    0.162516 |              0.742913  |           0.729793  |        0.729426 |              0.755119  |           0.77604   |       0.764806  |
| std  |    0.015274 |              0.0347032 |           0.0122253 |        0.020653 |              0.0330262 |           0.0176277 |       0.0137318 |

---

**5. With weighted loss and weights powered to 0.25** `test_metrics_004`

|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |   0.159914  |              0.743584  |           0.731342  |       0.727939  |              0.749723  |           0.778869  |      0.763841   |
| std  |   0.0123598 |              0.0181416 |           0.0251816 |       0.0198994 |              0.0129931 |           0.0138876 |      0.00562414 |

---

**6.  With weighted loss and 1 as minimal weight and with oversampling** `test_metrics_005`

|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |   0.168241  |              0.722945  |           0.730733  |        0.720674 |              0.744992  |            0.770806 |      0.757171   |
| std  |   0.0199579 |              0.0292648 |           0.0169793 |        0.013766 |              0.0276423 |            0.014483 |      0.00855044 |

---

**7.  With oversampling** `test_metrics_006`

|      |   eval_loss |   eval_macro_precision |   eval_macro_recall |   eval_macro_f1 |   eval_micro_precision |   eval_micro_recall |   eval_micro_f1 |
|:-----|------------:|-----------------------:|--------------------:|----------------:|-----------------------:|--------------------:|----------------:|
| mean |   0.149727  |              0.734173  |           0.712603  |       0.714048  |              0.752502  |           0.765247  |       0.758433  |
| std  |   0.0150912 |              0.0265015 |           0.0182918 |       0.0110687 |              0.0257817 |           0.0136373 |       0.0105635 |