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

    if ensemble is None:
        rank = 0
        nlocal_stages = nw
    else:
        rank = ensemble.ensemble_rank
        nlocal_stages = nw//ensemble.ensemble_size

    if rank == 0:
        nlocal_stages -= 1

    # background is reference plus noise
    background = noisify(ic.copy(deepcopy=True), sigma_b)

    y = []
    # initial observation
    if rank == 0:
        y.append(noisify(H(ic), sigma_r))

    def solve_local_stages():
        for k in range(nlocal_stages):
            for i in range(nt):
                un1.assign(un)
                stepper.solve()
                un.assign(un1)
            noisify(un, sigma_q)
            y.append(noisify(H(un), sigma_r))

    un.assign(ic)
    if ensemble is None:
        solve_local_stages()
    else:
        with ensemble.sequential(u=un) as ctx:
            un.assign(ctx.u)
            solve_local_stages()

    return y, background
