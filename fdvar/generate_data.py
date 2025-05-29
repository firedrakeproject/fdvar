import firedrake as fd
import numpy as np


def noisify(u, sigma, bcs=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for dat in u.dat:
        noise = np.random.normal(0, sigma, dat.data.shape)
        # fd.PETSc.Sys.Print(f"{sigma = :.3e} | {np.std(noise)/sigma = :.2e}")
        dat.data[:] += noise
    if bcs:
        for bc in bcs:
            bc.apply(u)
    return u


def generate_observation_data(ensemble, ic, stepper, un, un1, bcs, t, H,
                              nw, nt, B, R, Q, seed=6):
    Print = fd.PETSc.Sys.Print
    if seed is not None:
        np.random.seed(seed)

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

    # FunctionSpaces

    background = ic.copy(deepcopy=True)
    background.assign(background + B.correlated_noise())
    # Print(f"{fd.errornorm(ic, background)/fd.norm(ic) = :.4e}")

    un.assign(ic)
    un1.assign(ic)

    y = []
    ground_truth = []
    # initial observation
    if rank == 0:
        Hx = H(ic)
        Hx.assign(Hx + R.correlated_noise())
        y.append(Hx)

    def solve_local_stages(stage_offset=0):
        ground_truth.append(un.copy(deepcopy=True))
        for k in range(nlocal_stages):
            stage_idx = stage_offset + k
            for i in range(nt):
                stepper()
                ground_truth.append(un.copy(deepcopy=True))
            un.assign(un + Q.correlated_noise())
            # Print(f"{fd.errornorm(ground_truth[-1], un)/fd.norm(ground_truth[-1]) = :.4e}")

            Hx = H(un)
            yx = Hx.copy(deepcopy=True)
            Hx.assign(Hx + R.correlated_noise())
            # Print(f"{fd.errornorm(yx, Hx)/fd.norm(yx) = :.4e}")
            y.append(Hx)

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
