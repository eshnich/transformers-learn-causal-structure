from pathlib import Path

import optax
import tyro
from util import *
from PIL import Image

import wandb
from catformer import CatFormer
from plots import *
from problems import *


def main(
    vocab_size: int = 10,
    seq_len: int = 20,
    alpha: float = 0.1,
    dag: str = "chain",
    seed: int = 0,
    lr: float = 1,
    wd: float = 0,
    steps: int = 2**17,
    n_save: int = 128,
    batch_size: int = 1024,
    max_size: int = 2**20,
):
    config = locals()
    rng = RNG(seed)
    if dag.lower() == "chain":
        dag = jnp.arange(seq_len - 1) - 1
    elif dag.lower() == "icl":
        dag = jnp.zeros(seq_len - 1, dtype=int) - 1
        dag = dag.at[1::2].set(2 * jnp.arange((seq_len - 1) // 2))
    elif dag.lower() == "random":
        dag = rng.randint((seq_len - 1,), minval=-1, maxval=jnp.arange(seq_len - 1))
    elif dag.lower() == "manual":
        dag = jnp.array([-1, 0, 0, 1, 2])
        vocab_size = 3
        seq_len = 6
    problem = InContextTree(
        vocab_size=vocab_size,
        dag=dag,
        alpha=alpha,
    )
    model = CatFormer(
        seq_len=seq_len,
        vocab_size=vocab_size,
        heads=[1, 1],
    )

    @jit
    def criterion(f, y):
        _criterion = lambda f, y: -jnp.log(f) @ y
        for _ in range(y.ndim - 1):
            _criterion = vmap(_criterion)
        return _criterion(f, y).mean()

    @jit
    def loss_fn(model, batch):
        x, y = batch
        return criterion(model(x), y)

    A = jnp.zeros((seq_len, seq_len))
    idx = jnp.where(dag >= 0)
    A = A.at[idx, dag[idx]].set(1)

    print("Computing Bayes")

    testx, testy = vmap(problem.sample)(rng.next(2**16))
    logits = vmap(problem.bayes)(testx)
    bayes = criterion(logits, testy)

    print("Training")
    save_every = steps // n_save
    epoch_len = max_size // batch_size
    sample_fn = jit(lambda k: vmap(problem.sample)(jr.split(k, epoch_len * batch_size)))

    def batch_iterator(key):
        while True:
            key, subkey = jr.split(key)
            batches = sample_fn(subkey)
            for i in range(epoch_len):
                yield tree_map(
                    lambda x: x[batch_size * i : batch_size * (i + 1)], batches
                )

    @jit
    def step_fn(model, batch, lr, wd):
        g = jax.grad(loss_fn)(model, batch)
        g = tree_add_scalar_mul(g, wd, model)
        model = tree_add_scalar_mul(model, -lr, g)
        return model

    iterator = batch_iterator(rng.next())
    schedule = optax.cosine_decay_schedule(lr, steps)
    test_losses = []
    pbar = tqdm(total=steps)
    wandb.init(project="ICL", config=config, name="single_parent_20")
    for i in range(steps):
        if i % save_every == 0:
            test_loss = loss_fn(model, (testx, testy))
            test_losses.append(test_loss)
            wandb.log(dict(loss=test_loss, bayes=bayes, step=i, lr=schedule(i)))
            pbar.n = i
            pbar.refresh()
        model = step_fn(model, next(iterator), lr=schedule(i), wd=wd)
    pbar.n = steps
    pbar.refresh()
    pbar.close()
    test_losses = jnp.array(test_losses)

    fig = plot_losses(test_losses, bayes, save_every)
    filename = wandb.run.dir + "/losses.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({"losses/test": wandb.Image(Image.open(filename))})
    plt.close(fig)

    fig = plot_A(A)
    filename = wandb.run.dir + "/A.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({"A": wandb.Image(Image.open(filename))})
    plt.close(fig)

    lower = 15
    upper = 100
    
    fig = plot_A1(model.A[0][0], vocab_size, seq_len, lower, upper)
    filename = wandb.run.dir + f"/A1.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({f"A1/{lower}_{upper}": wandb.Image(Image.open(filename))})
    plt.close(fig)

    fig = plot_A2(model.A[1][0], vocab_size, seq_len, lower, upper)
    filename = wandb.run.dir + f"/A2.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({f"A2/{lower}_{upper}": wandb.Image(Image.open(filename))})
    plt.close(fig)

    fig = plot_W(model.W.T, vocab_size, seq_len, lower, upper)
    filename = wandb.run.dir + f"/W.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({f"W/{lower}_{upper}": wandb.Image(Image.open(filename))})
    plt.close(fig)
        
    wandb.finish()


tyro.cli(main)
tyro.cli(main)
