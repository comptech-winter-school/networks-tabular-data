# "Neural networks for tabular data" project in Winter School CompTech 2021
  Purpose of this project is benchmarking performance of neural networks architectures on tabular data. For this reason was used [openML autoML benchmark](https://github.com/openml/automlbenchmark). New framework modules were coded for benchmark. This work is based on paper "[An Open Source AutoML Benchmark](https://arxiv.org/abs/1907.00909)". 
  
# Architectures used in work
- SNN (Self-Normalizing Neural Networks) (paper: [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515))
- NODE(Neural Oblivious Decision Ensembles) (paper: [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/pdf/1909.06312.pdf))
- TabNet (paper: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442))

# Experiment description
We used implementations of the architectures:
- SNN
- NODE - 
- TabNet on PyTorch - https://github.com/dreamquark-ai/tabnet
   All architectures were wrapped in sklearn Model classes for better compatibility with benchmark.
   Default hyperparameters were used for TabNet.
  
  
# How to use this repository
You can find `.py` files for openML autoML benchmark in folder `frameworks`. You can put them in folder `frameworks` of benchmark and use like default benchmark frameworks.

# Results
You can see table of results in [results.csv](https://github.com/comptech-winter-school/networks-tabular-data/blob/main/results/results.csv). There are results from original paper for comparison.

# Conclusions
