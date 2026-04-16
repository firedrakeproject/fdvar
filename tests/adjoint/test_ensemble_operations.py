from firedrake import *
from firedrake.adjoint import *
from fdvar.adjoint import (
    EnsembleTransform,
    EnsembleReduce,
    EnsembleBcast,
    EnsembleAdjVec
)
import pytest
from pytest_mpi.parallel_assert import parallel_assert


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.parallel(nprocs=[1, 2, 3, 6])
def test_ensemble_bcast_float():
    ensemble = Ensemble(COMM_WORLD, 1)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    eps = 1e-12

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    y = EnsembleAdjVec(
        [AdjFloat(0.0) for _ in range(nlocal_cpts)],
        ensemble)

    # B: X -> Y
    bcast = EnsembleBcast(dst=y)

    # =======
    # Forward
    # =======

    # Recomputation

    x = AdjFloat(3.0)
    y = bcast(x)

    expect = x
    match_local = all((yi - expect) < eps for yi in y.subvec)

    parallel_assert(
        match_local,
        msg=f"Broadcast AdjFloats {y.subvec} do not match expected value {expect}."
    )

    # Tangent linear

    x = AdjFloat(2.0)
    y = bcast.tlm(x)

    expect = x
    match_local = all((yi - expect) < eps for yi in y.subvec)

    parallel_assert(
        match_local,
        msg=f"Broadcast tlm AdjFloats {y.subvec} do not match expected value {expect}."
    )

    # ========
    # Backward
    # ========

    # Check the adjoint is reduced back to all ranks.
    # Because the functional is an array we need to
    # pass an adj_input of an array.
    offset = rank*nlocal_cpts

    # Derivative

    yhat = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0) for i in range(nlocal_cpts)],
        ensemble=ensemble)

    expect = AdjFloat(sum(i+1.0 for i in range(nglobal_cpts)))

    xhat = bcast.derivative(adj_input=yhat, apply_riesz=True)

    match_local = abs(xhat - expect) < eps

    parallel_assert(
        match_local,
        msg=f"Broadcast derivative {xhat} does not match"
            f" expected value {expect}."
    )

    # Hessian

    yhat = EnsembleAdjVec(
        [AdjFloat(offset + i + 10.0) for i in range(nlocal_cpts)],
        ensemble=ensemble)

    expect = AdjFloat(sum(i+10.0 for i in range(nglobal_cpts)))

    xhat = bcast.hessian(
        m_dot=None, hessian_input=yhat,
        evaluate_tlm=False, apply_riesz=True)

    match_local = abs(xhat - expect) < eps

    parallel_assert(
        match_local,
        msg=f"Broadcast hessian {xhat} does not match"
            f" expected value {expect}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_ensemble_bcast_function():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    eps = 1e-12

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    mesh = UnitIntervalMesh(1, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Y = EnsembleFunctionSpace(
        [R for _ in range(nlocal_cpts)], ensemble)

    y = EnsembleFunction(Y)

    # B: X -> Y
    bcast = EnsembleBcast(dst=y)

    # =======
    # Forward
    # =======

    # Recomputation

    x = Function(R).assign(3.0)
    y = bcast(x)

    expect = x
    match_local = all(errornorm(yi, expect) < eps
                      for yi in y.subfunctions)

    parallel_assert(
        match_local,
        msg="Broadcast Functions do not match."
    )

    # Tangent linear

    x = Function(R).assign(2.0)
    y = bcast.tlm(x)

    expect = x
    match_local = all(errornorm(yi, expect) < eps
                      for yi in y.subfunctions)

    parallel_assert(
        match_local,
        msg="Broadcast tlm Functions do not match."
    )

    # ========
    # Backward
    # ========

    # Check the adjoint is reduced back to all ranks.
    # Because the functional is an EnsembleFunction we
    # need to pass an adj_input of an EnsembleCofunction.
    offset = rank*nlocal_cpts

    # Derivative

    yhat = EnsembleFunction(Y)
    for i, yi in enumerate(yhat.subfunctions):
        yi.assign(offset + i + 1.0)
    yhat = yhat.riesz_representation()

    expect = Function(R).assign(
        sum(i+1.0 for i in range(nglobal_cpts)))

    xhat = bcast.derivative(adj_input=yhat, apply_riesz=True)

    match_local = errornorm(xhat, expect) < eps

    parallel_assert(
        match_local,
        msg=f"Broadcast derivative {xhat.dat.data[:]} does not match"
            f" expected value {expect.dat.data[:]}."
    )

    # Hessian

    yhat = EnsembleFunction(Y)
    for i, yi in enumerate(yhat.subfunctions):
        yi.assign(offset + i + 10.0)
    yhat = yhat.riesz_representation()

    expect = Function(R).assign(
        sum(i+10.0 for i in range(nglobal_cpts)))

    xhat = bcast.hessian(
        m_dot=None, hessian_input=yhat,
        evaluate_tlm=False, apply_riesz=True)

    match_local = errornorm(xhat, expect) < eps

    parallel_assert(
        match_local,
        msg=f"Broadcast derivative {xhat.dat.data[:]} does not match"
            f" expected value {expect.dat.data[:]}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 6])
def test_ensemble_reduce_float():
    ensemble = Ensemble(COMM_WORLD, 1)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    eps = 1e-12

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    x = EnsembleAdjVec(
        [AdjFloat(0.0) for _ in range(nlocal_cpts)],
        ensemble=ensemble)

    # R: X -> Y
    reduce = EnsembleReduce(src=x)

    # =======
    # Forward
    # =======
    offset = rank*nlocal_cpts

    # Recomputation

    x = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0) for i in range(nlocal_cpts)],
        ensemble=ensemble)

    y = reduce(x)

    expect = AdjFloat(sum(i+1.0 for i in range(nglobal_cpts)))
    match_local = abs(y - expect) < eps

    parallel_assert(
        match_local,
        msg=f"Reduced AdjFloat {y} does not match"
            f" expected value {expect}"
    )

    # Tangent linear

    x = EnsembleAdjVec(
        [AdjFloat(offset + i + 10.0) for i in range(nlocal_cpts)],
        ensemble=ensemble)

    y = reduce.tlm(x)

    expect = AdjFloat(sum(i+10.0 for i in range(nglobal_cpts)))
    match_local = abs(y - expect) < eps

    parallel_assert(
        match_local,
        msg=f"Reduced TLM AdjFloat {y} does not match"
            f" expected value {expect}"
    )

    # ========
    # Backward
    # ========

    # Check the adjoint is broadcast back to all ranks.
    # Because the functional is an vector we need to
    # pass an adj_input of an flaot.

    # Derivative

    yhat = AdjFloat(20.0)

    expect = AdjFloat(yhat)
    xhat = reduce.derivative(adj_input=yhat, apply_riesz=True)

    match_local = all((xi - expect) < eps for xi in xhat.subvec)

    parallel_assert(
        match_local,
        msg=f"Reduced derivatives {xhat.subvec} do not match expected value {expect}."
    )

    # Hessian

    yhat = AdjFloat(200.0)

    expect = AdjFloat(yhat)
    xhat = reduce.hessian(
        m_dot=None, hessian_input=yhat,
        evaluate_tlm=False, apply_riesz=True)

    match_local = all((xi - expect) < eps for xi in xhat.subvec)

    parallel_assert(
        match_local,
        msg=f"Reduced Hessian {xhat.subvec} do not match expected value {expect}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_ensemble_reduce_function():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    eps = 1e-12

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    mesh = UnitIntervalMesh(1, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    X = EnsembleFunctionSpace(
        [R for _ in range(nlocal_cpts)], ensemble)

    x = EnsembleFunction(X)

    # R: X -> Y
    reduce = EnsembleReduce(src=x)

    # =======
    # Forward
    # =======
    offset = rank*nlocal_cpts

    # Recomputation

    x = EnsembleFunction(X)

    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    y = reduce(x)

    expect = Function(R).assign(sum(i+1.0 for i in range(nglobal_cpts)))
    match_local = errornorm(y, expect) < eps

    parallel_assert(
        match_local,
        msg=f"Reduced Function {y.dat.data[:]} does not match"
            f" expected value {expect.dat.data[:]}"
    )

    # Tangent linear

    x = EnsembleFunction(X)

    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 10.0)

    y = reduce(x)

    expect = Function(R).assign(sum(i+10.0 for i in range(nglobal_cpts)))
    match_local = errornorm(y, expect) < eps

    parallel_assert(
        match_local,
        msg=f"Reduced TLM Function {y.dat.data[:]} does not match"
            f" expected value {expect.dat.data[:]}"
    )

    # ========
    # Backward
    # ========

    # Check the adjoint is broadcast back to all ranks.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    # Derivative

    adj_value = 20.0
    yhat = Function(R).assign(adj_value).riesz_representation()
    expect = Function(R).assign(adj_value)

    xhat = reduce.derivative(adj_input=yhat, apply_riesz=True)

    match_local = all(errornorm(xi, expect) < eps
                      for xi in xhat.subfunctions)

    parallel_assert(
        match_local,
        msg=f"Reduced derivatives {[xi.dat.data[:] for xi in xhat.subfunctions]}"
            f" do not match expected value {expect.dat.data[:]}."
    )

    # Hessian

    hessian_value = 20.0
    yhat = Function(R).assign(hessian_value).riesz_representation()
    expect = Function(R).assign(hessian_value)

    xhat = reduce.hessian(
        m_dot=None, hessian_input=yhat,
        evaluate_tlm=False, apply_riesz=True)

    match_local = all(errornorm(xi, expect) < eps
                      for xi in xhat.subfunctions)

    parallel_assert(
        match_local,
        msg=f"Reduced Hessian {[xi.dat.data[:] for xi in xhat.subfunctions]}"
            f" do not match expected value {expect.dat.data[:]}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_ensemble_transform_float():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_cpts)], ensemble)

    c = EnsembleFunction(Re)

    rfs = []
    J = []
    offset = rank*nlocal_cpts
    for ci in c.subfunctions:
        with set_working_tape() as tape:
            Ji = assemble(ci*ci*dx)
            J.append(Ji)
            rfs.append(ReducedFunctional(Ji, Control(ci), tape=tape))

    J = EnsembleAdjVec(J, ensemble)

    Jhat = EnsembleTransform(J, Control(c), rfs)

    # check the control is reduced to all ranks
    eps = 1e-12

    x = EnsembleFunction(Re)

    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    # check
    Jx = Jhat(x)

    expect = [rf(xi) for rf, xi in zip(rfs, x.subfunctions)]

    match_local = all((Ji - ei) < eps for Ji, ei in zip(Jx.subvec, expect))

    parallel_assert(
        match_local,
        msg=f"Transformed results {Jx} do not match expected values {expect}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a list[AdjFloat] we need to
    # pass an adj_input of a list[AdjFloat].

    adj_input = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0)
         for i in range(nlocal_cpts)],
        ensemble=ensemble)

    expect = EnsembleFunction(Re)
    for rf, adji, ei in zip(rfs, adj_input.subvec, expect.subfunctions):
        ei.assign(rf.derivative(adj_input=adji, apply_riesz=True))

    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ.subfunctions, expect.subfunctions))

    parallel_assert(
        match_local,
        msg=f"Reduced derivatives {[dJi.dat.data[:] for dJi in dJ.subfunctions]}"
            f" do not match expected value {[ei.dat.data[:] for ei in expect.subfunctions]}."
    )

    _ = Jhat.tlm(x)
    _ = Jhat.hessian(x, hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_ensemble_transform_float_two_controls():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_cpts)], ensemble)

    c0 = EnsembleFunction(Re)
    c1 = EnsembleFunction(Re)

    rfs = []
    J = []
    offset = rank*nlocal_cpts
    for c0i, c1i in zip(c0.subfunctions, c1.subfunctions):
        with set_working_tape() as tape:
            Ji = assemble((c0i*c0i + c1i*c1i)*dx)
            J.append(Ji)
            rfs.append(ReducedFunctional(
                Ji, [Control(c0i), Control(c1i)], tape=tape))

    J = EnsembleAdjVec(J, ensemble)

    Jhat = EnsembleTransform(
        J, [Control(c0), Control(c1)], rfs)

    # check the control is reduced to all ranks
    eps = 1e-12

    x0 = EnsembleFunction(Re)
    x1 = EnsembleFunction(Re)

    for i, (x0i, x1i) in enumerate(zip(x0.subfunctions, x1.subfunctions)):
        x0i.assign(offset + i + 1.0)
        x1i.assign(2*(offset + i + 1.0))

    # check
    Jx = Jhat([x0, x1])

    expect = [rf([x0i, x1i])
              for rf, x0i, x1i in zip(rfs, x0.subfunctions, x1.subfunctions)]

    match_local = all((Ji - ei) < eps for Ji, ei in zip(Jx.subvec, expect))

    parallel_assert(
        match_local,
        msg=f"Transformed results {Jx} do not match expected values {expect}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a AdjFloat we need to
    # pass an adj_input of a list[AdjFloat].

    adj_input = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0)
         for i in range(nlocal_cpts)],
        ensemble=ensemble)

    expect0 = EnsembleFunction(Re)
    expect1 = EnsembleFunction(Re)
    for rf, adji, e0i, e1i in zip(rfs, adj_input.subvec,
                                  expect0.subfunctions,
                                  expect1.subfunctions):
        e0, e1 = rf.derivative(adj_input=adji, apply_riesz=True)
        e0i.assign(e0)
        e1i.assign(e1)

    dJ0, dJ1 = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local0 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ0.subfunctions, expect0.subfunctions))

    match_local1 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ1.subfunctions, expect1.subfunctions))

    parallel_assert(
        match_local0,
        msg=f"Reduced derivatives {[dJ0i.dat.data[:] for dJ0i in dJ0.subfunctions]}"
            f" do not match expected value {[e0i.dat.data[:] for e0i in expect0.subfunctions]}."
    )

    parallel_assert(
        match_local1,
        msg=f"Reduced derivatives {[dJ1i.dat.data[:] for dJ1i in dJ1.subfunctions]}"
            f" do not match expected value {[e1i.dat.data[:] for e1i in expect1.subfunctions]}."
    )

    _ = Jhat.tlm([x0, x1])
    _ = Jhat.hessian([x0, x1], hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_ensemble_transform_function():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_cpts)], ensemble)

    c = EnsembleFunction(Re)
    J = EnsembleFunction(Re)

    rfs = []
    offset = rank*nlocal_cpts
    for i, (Ji, ci) in enumerate(zip(J.subfunctions, c.subfunctions)):
        with set_working_tape() as tape:
            Ji.assign(ci)
            Ji += 2*(offset + i + 1.0)
            rfs.append(ReducedFunctional(Ji, Control(ci), tape=tape))

    Jhat = EnsembleTransform(J, Control(c), rfs)

    # check the control is reduced to all ranks
    eps = 1e-12

    x = EnsembleFunction(Re)

    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    # check
    Jx = Jhat(x)

    expect = EnsembleFunction(Re)
    for rf, xi, ei in zip(rfs, x.subfunctions,
                          expect.subfunctions):
        ei.assign(rf(xi))

    match_local = all(
        errornorm(Ji, ei) < eps
        for Ji, ei in zip(Jx.subfunctions, expect.subfunctions))

    parallel_assert(
        match_local,
        msg=f"Transformed Functions {[Ji.dat.data[:] for Ji in Jx.subfunctions]}"
            f" do not match expected value {[ei.dat.data[:] for ei in expect.subfunctions]}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    adj_input = EnsembleFunction(Re)
    for i, adj in enumerate(adj_input.subfunctions):
        adj.assign(offset + i + 1.0)

    adj_input = adj_input.riesz_representation()

    expect = EnsembleFunction(Re)
    for rf, adji, ei in zip(rfs, adj_input.subfunctions,
                            expect.subfunctions):
        ei.assign(rf.derivative(adj_input=adji, apply_riesz=True))

    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ.subfunctions, expect.subfunctions))

    parallel_assert(
        match_local,
        msg=f"Reduced derivatives {[dJi.dat.data[:] for dJi in dJ.subfunctions]}"
            f" do not match expected value {[ei.dat.data[:] for ei in expect.subfunctions]}."
    )

    _ = Jhat.tlm(x)
    _ = Jhat.hessian(x, hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
def test_ensemble_transform_function_two_controls():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_cpts = 6
    nlocal_cpts = nglobal_cpts // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_cpts)], ensemble)

    c0 = EnsembleFunction(Re)
    c1 = EnsembleFunction(Re)
    J = EnsembleFunction(Re)

    rfs = []
    offset = rank*nlocal_cpts
    for i, (Ji, c0i, c1i) in enumerate(zip(J.subfunctions,
                                           c0.subfunctions,
                                           c1.subfunctions)):
        with set_working_tape() as tape:
            Ji.assign(c0i + c1i)
            rfs.append(ReducedFunctional(
                Ji, [Control(c0i), Control(c1i)], tape=tape))

    Jhat = EnsembleTransform(J, [Control(c0), Control(c1)], rfs)

    # check the control is reduced to all ranks
    eps = 1e-12

    x0 = EnsembleFunction(Re)
    x1 = EnsembleFunction(Re)

    for i, (x0i, x1i) in enumerate(zip(x0.subfunctions,
                                       x1.subfunctions)):
        x0i.assign(offset + i + 1.0)
        x1i.assign(2*(offset + i + 1.0))

    Jx = Jhat([x0, x1])

    expect = EnsembleFunction(Re)
    for rf, x0i, x1i, ei in zip(rfs, x0.subfunctions,
                                x1.subfunctions,
                                expect.subfunctions):
        ei.assign(rf([x0i, x1i]))

    match_local = all(
        errornorm(Ji, ei) < eps
        for Ji, ei in zip(Jx.subfunctions,
                          expect.subfunctions))

    parallel_assert(
        match_local,
        msg=f"Transformed Functions {[Ji.dat.data[:] for Ji in Jx.subfunctions]}"
            f" do not match expected value {[ei.dat.data[:] for ei in expect.subfunctions]}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    adj_input = EnsembleFunction(Re)
    for i, adj in enumerate(adj_input.subfunctions):
        adj.assign(offset + i + 1.0)

    adj_input = adj_input.riesz_representation()

    expect0 = EnsembleFunction(Re)
    expect1 = EnsembleFunction(Re)
    for rf, adji, e0i, e1i in zip(rfs, adj_input.subfunctions,
                                  expect0.subfunctions,
                                  expect1.subfunctions):
        e0, e1 = rf.derivative(adj_input=adji, apply_riesz=True)
        e0i.assign(e0)
        e1i.assign(e1)

    dJ0, dJ1 = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local0 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ0.subfunctions, expect0.subfunctions))

    match_local1 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ1.subfunctions, expect1.subfunctions))

    parallel_assert(
        match_local0,
        msg=f"Reduced derivatives {[dJ0i.dat.data[:] for dJ0i in dJ0.subfunctions]}"
            f" do not match expected value {[e0i.dat.data[:] for e0i in expect0.subfunctions]}."
    )

    parallel_assert(
        match_local1,
        msg=f"Reduced derivatives {[dJ1i.dat.data[:] for dJ1i in dJ1.subfunctions]}"
            f" do not match expected value {[e1i.dat.data[:] for e1i in expect1.subfunctions]}."
    )

    _ = Jhat.tlm([x0, x1])
    _ = Jhat.hessian([x0, x1], hessian_input=adj_input)
