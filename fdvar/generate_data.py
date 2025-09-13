import firedrake as fd
import numpy as np


def generate_observation_data(W, ic, stepper, un, un1, bcs, t, H,
                              nw, nt, B, R, Q, seed=6):
    Print = fd.PETSc.Sys.Print
    if seed is not None:
        np.random.seed(seed)

    ensemble = W.ensemble if W else None

    if ensemble is None:
        rank = 0
        nlocal_stages = nw
    else:
        rank = ensemble.ensemble_rank
        assert nw % ensemble.ensemble_size == 0
        nlocal_stages = nw//ensemble.ensemble_size

    # background is reference plus noise
    bcs = bcs or []
    for bc in bcs:
        bc.apply(ic)

    if ensemble:
        ground_truth = fd.EnsembleFunction(W)
    else:
        ground_truth = []

    bnoise = B.correlated_noise()
    if ensemble:
        bnoise = ensemble.bcast(bnoise, root=0)

    background = ic.copy(deepcopy=True)
    background.assign(background + bnoise)

    un.assign(ic)
    un1.assign(ic)

    if ensemble and rank == 0:
        ground_truth.subfunctions[0].assign(ic)

    y = []
    # initial observation
    if rank == 0:
        Hx = H(ic)
        Hx.assign(Hx + R.correlated_noise())
        y.append(Hx)

    # We need to make sure that the generated noise
    # is the same regardless of the partition.
    # To do this we generate all noise on rank 0
    # and broadcast to all other ranks.
    if ensemble:
        rnoise = [ensemble.bcast(R.correlated_noise(), root=0)
                  for _ in range(nw)]
        qnoise = [ensemble.bcast(Q.correlated_noise(), root=0)
                  for _ in range(nw)]
    else:
        rnoise = [R.correlated_noise() for _ in range(nw)]
        qnoise = [Q.correlated_noise() for _ in range(nw)]

    def solve_local_stages(stage_offset=0):

        if not ensemble:
            ground_truth.append(un.copy(deepcopy=True))
        for k in range(nlocal_stages):
            stage_idx = stage_offset + k
            for i in range(nt):
                stepper()
                if not ensemble:
                    ground_truth.append(un.copy(deepcopy=True))
            un.assign(un + qnoise[stage_idx])

            Hx = H(un)
            Hx.assign(Hx + rnoise[stage_idx])
            y.append(Hx)

            if ensemble:
                ix = k + (rank == 0)
                ground_truth.subfunctions[ix].assign(un)

    un.assign(ic)
    un1.assign(ic)
    t.assign(0)
    if ensemble is None:
        solve_local_stages(stage_offset=0)
        end = un.copy(deepcopy=True)
    else:
        with ensemble.sequential(u=un, t=t, stage_idx=0) as ctx:
            un.assign(ctx.u)
            t.assign(ctx.t)
            solve_local_stages(stage_offset=ctx.stage_idx)
            ctx.stage_idx += nlocal_stages
        end = un.copy(deepcopy=True)
        ensemble.bcast(
            end, root=ensemble.ensemble_size-1)

    return y, background, end, ground_truth
