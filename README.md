# Implicit Neural Representations

WIP. Implicit Neural Representations in JAX with DMHaiku.

## Requirements

```commandline
pip install --upgrade jax jaxlib pillow optax
# for a GPU enviroment,
# pip install -U jax jaxlib==0.1.62+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install git+https://github.com/deepmind/dm-haiku
```

## COIN

```commandline
python coin.py
```

### Original Image

![](cat.jpg)

### Compressed Image

![](coin_example.png)