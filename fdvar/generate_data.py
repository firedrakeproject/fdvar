import firedrake as fd
import numpy as np


def noisify(u, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for dat in u.dat:
        dat.data[:] += np.random.normal(0, sigma, dat.data.shape)
    return u


def generate_observation_data(ensemble, ic, stepper, un, un1, H,
                              nw, nt, sigma_b, sigma_r, sigma_q, seed=6):
    if seed is not None:
        np.random.seed(seed)

    rank = ensemble.ensemble_rank

    nlocal_stages = nw//ensemble.ensemble_size
    if rank == 0:
        nlocal_stages -= 1

    # background is reference plus noise
    background = ic.copy(deepcopy=True)
    noisify(background, sigma_b)

    y = []
    # initial observation
    if rank == 0:
        y.append(noisify(H(ic), sigma_r))

    widx = 0
    with ensemble.sequential(widx=widx, un=un) as ctx:
        # if rank != 0:
        #     y.extend([None for _ in range(ctx.widx)])
        un.assign(ctx.un)
        un1.assign(un)
        for k in range(nlocal_stages):
            for i in range(nt):
                stepper.solve()
                un.assign(un1)
            noisify(un, sigma_q)
            y.append(noisify(H(un), sigma_r))
            ctx.widx += 1

    return y, background
