# MolAttnNet: A Predictive Model for Organic Drug Solubility Based on Graph Convolutional Networks and Transformer-Attention

<img width="779" alt="image" src="https://github.com/xunyoyo/YZS-Model/assets/33387866/b0b34260-67ac-4d64-971f-93f009c4eb40">



## Configuration of the MolAttnNet environment

```
python==3.6
torch==1.7.1
scikit-learn==0.24.2
scipy==1.5.4
torch-geometric==1.7.0
einops==0.4.1
networkx==2.5.1
rdkit-pypi
```

Other packages can be found from `requirements.txt`.

## Datasets

You can download the training and testing dataset from:

[All datasets in the raw folders.](https://github.com/xunyoyo/MolAttnNet/tree/main/Datasets)

## Useage

+ `smile2topology.py`: Convert .csv files to datasets.
+ `model.py`: The whole YZS-model.
+ `opti.py`: Using searching package to find the perfect parameters.
+ `train.py`: Training the YZS-model.
+ `test.py`:Tests and evaluates the YZS model.

## About

Statement:
+ Part of code come from:
```
https://github.com/ziduzidu/CSDTI
https://github.com/ltorres97/FS-CrossTR
https://github.com/waqarahmadm019/AquaPred
```
When using the above-mentioned open-source code, we have already indicated this in the documents.

## Implementation Details

We focused on optimizing the hyperparameters of our MolAttnNet to maximize predictive accuracy for organic drug solubility. Using the [hyperopt](https://github.com/hyperopt/hyperopt) library and the Tree-structured Parzen Estimator (TPE) algorithm, we systematically explored several key hyperparameters, including:

- Learning rate
- Feature vector dimension
- Dropout rate
- Transformer layer depth
- Number of attention heads in each layer
- Batch size

Hyperopt training was configured to stop after 200 iterations, with details recorded in a log file. Our search strategy and final choices are summarized below.

### Final Model Configuration

- **Transformer depth:** 6 layers
- **Attention heads per layer:** 8 (dimension: 92)
- **Linear Layer dimension:** 256
- **Dropout rate:** 0.25
- **Number of input features:** 92
- **Feature dimension:** 128
- **Final dropout (finely tuned):** 0.2519

These optimized parameters, derived from extensive experimentation and validation, significantly enhanced the MolAttnNet’s performance in solubility prediction.

### Hyperparameter Search Space

| Item      | Range           | Selection Method   |
|-----------|-----------------|-------------------|
| `lr`      | (0.0003, 0.0007) | `hp.uniform`      |
| `dim`     | (92, 128, 2)     | `hp.quniform`     |
| `dropout` | (0.25, 0.35)     | `hp.uniform`      |
| `depth`   | [2, 4, 6, 8, 12] | `hp.choice`       |
| `heads`   | [4, 8, 12, 16]   | `hp.choice`       |
| `batch_size` | (24, 72, 8)   | `hp.quniform`     |

- `hp.uniform` — samples uniformly within the specified range  
- `hp.choice` — picks from the specified list of options  
- `hp.quniform` — samples discrete values within the range at specified intervals
