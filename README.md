# ImplicitSLIM and How it Improves Embedding-based Collaborative Filtering

The official implementation of the paper "ImplicitSLIM and How it Improves Embedding-based Collaborative Filtering" ([arXiv](https://arxiv.org/abs/2406.00198), [OpenReview](https://openreview.net/forum?id=6vF0ZJGor4)).

`implicit_slim.py` contains implementations of ImplicitSLIM and LLE-SLIM.

`downstream_models.py` contains implementations of simple downstream models, including Matrix Factorization and PLRec. 

`ImplicitSLIM.ipynb` provides several examples of applying ImplicitSLIM and LLE-SLIM to Matrix Factorization and PLRec.

An example of applying ImplicitSLIM to a deep model is provided in the [RecVAE repository](https://github.com/ilya-shenbin/RecVAE).

If you find this paper or this code useful, please cite our paper:

```
@inproceedings{
  shenbin2024implicitslim,
  title={Implicit{SLIM} and How it Improves Embedding-based Collaborative Filtering},
  author={Ilya Shenbin and Sergey Nikolenko},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=6vF0ZJGor4}
}
```
