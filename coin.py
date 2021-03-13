# Dupond+2021 COIN: COmpression with Implicit Neural representations, https://arxiv.org/abs/2103.03123

import functools
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image


def model_fn(xy: np.ndarray,
             depth: int,
             width: int
             ) -> np.ndarray:
    mlp0 = hk.nets.MLP([width], activation=jnp.sin,
                       w_init=hk.initializers.RandomUniform(-np.sqrt(1 / 2), np.sqrt(1 / 2)))
    w_init = hk.initializers.RandomUniform(-np.sqrt(6 / width), np.sqrt(6 / width))
    mlp1 = hk.nets.MLP([width for _ in range(depth - 2)], activation=jnp.sin, w_init=w_init)
    mlp2 = hk.Linear(3, w_init=w_init)
    return jax.nn.sigmoid(mlp2(mlp1(mlp0(xy))))


def load_img(path: str
             ) -> np.ndarray:
    with open(path, "rb") as f:
        img = jnp.asarray(Image.open(f)) / 255
    return img


def _img_sample(key: np.ndarray,
                image: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
    w, h, _ = image.shape
    xy = jax.random.uniform(key, (2,), minval=-1, maxval=1)
    x = ((xy[0] + 1) / 2 * w).astype(int)
    y = ((xy[1] + 1) / 2 * h).astype(int)
    rgb = image[x, y]
    return xy, rgb


def main(iters=100_000,
         batch_size=1024,
         depth=10,
         width=50,
         report_freq=10_000,
         out_width=355 * 4,
         out_height=200 * 4):
    rng_key = jax.random.PRNGKey(42)
    _model = hk.transform(functools.partial(model_fn, depth=depth, width=width))
    model = hk.without_apply_rng(_model)
    image = load_img("cat.jpg")
    image_sample = jax.jit(jax.vmap(functools.partial(_img_sample, image=image)))
    opt = optax.adam(3e-4)

    def loss_impl(params, batch):
        xy, rgb = batch
        out = model.apply(params=params, xy=xy)
        return jnp.mean((rgb - out) ** 2)

    rng_key, next_rng = jax.random.split(rng_key)
    params = model.init(next_rng, np.zeros(2))
    opt_state = opt.init(params)
    loss_list = []

    @jax.jit
    def update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_impl)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for i in range(iters + 1):
        rng_key, next_rng = jax.random.split(rng_key)
        params, opt_state, loss = update(params, opt_state, image_sample(jax.random.split(next_rng, batch_size)))
        loss_list.append(loss)
        if i % report_freq == 0:
            x, y = jnp.meshgrid(jnp.linspace(-1, 1, out_height), jnp.linspace(-1, 1, out_width).reshape(-1))
            xy = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=1)
            out = model.apply(params=params, xy=xy)
            out_img = Image.fromarray(
                np.asarray(out * 255).astype(np.uint8).reshape(out_width, out_height, 3).transpose(1, 0, 2)
            )
            out_img.save(f"{i}.jpg")
            print(f"{i:>7} {sum(loss_list) / report_freq:.3e}")
            loss_list = []


if __name__ == '__main__':
    main()
