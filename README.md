# Towards Accurate Recommendation via LLM-enhanced Contrastive Learning

This project is a pytorch implementation of 'Towards Accurate Recommendation via LLM-enhanced Contrastive Learning'.
This project provides executable source code with adjustable arguments and preprocessed datasets used in the paper.
The backbone model of ARCoL is SASRec, and we built our implementation using the following repository: https://github.com/pmixer/SASRec.pytorch

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.13.1 

## Datasets
We use 2 datasets in our work: Books and Movies from Amazon
You can download the dataset [here](https://www.naver.com/). 


## Running the code
The different hyperparameters for each dataset are set in `main.py`.
You can train the model by following code.

```
bash run.sh
```




