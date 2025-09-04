# Designing-STARR-seq-Library
Designing STARR-seq library to explore the regulatory background. 

## Oricle model 
Oricle model is the distilled student DeepSTARR model with `ensemble_training_deepstarr.py`. This model was distilled from 10 [EvoAug](https://github.com/aduranu/evoaug) DeepSTARR models with different initializations with `distill_EvoAug.py`. 
If training an Oricle model be sure to:
```
pip install evoaug2
```



