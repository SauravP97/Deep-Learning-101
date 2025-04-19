# Deep Learning 101 [DL-101]
## Everything you need to learn to train Deep Neural Nets 

The repository holds the implementation of technical concepts involved the Deep Learning domain. We opt a practical approach of learning where we create a character level generation model from scratch and use it to generate more Indian Male names from an example dataset.

## Understanding the Problem Statement

We have a [dataset](https://gist.github.com/mbejda/7f86ca901fe41bc14a63) `Indian Male Names` csv file having `~15K` names. Now the idea is to feed these example names into the created Character level generation model and generate more names like these which should probably be new and not present in the dataset.

The implementation is influenced by:

  - Andrej's [makemore](https://github.com/karpathy/makemore/)
  - Multi Layer Perceptron (MLP) architecture, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Wavenet, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)


<img src="/media/mlp.png" width=600 height=400>


## Topics Covered

1. Building your Deep Neural Net architecture.
2. Training your Neural Net
3. Evaluating Dev and Test Losses
4. L2 Regularization
5. Drop-out Regularization
6. Vanishing / Exploding Gradients
7. Gradient Descent with Momentum
8. RMS Prop
9. Adam Optimization 
10. Batch Normalization (Batch Norm)
11. Early Stopping
12. Conclusions