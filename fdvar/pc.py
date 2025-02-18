

class EnsembleBlockDiagonalPC:
    prefix = "ensemblejacobi_"

    def __init__(self):
        self.initialized = False

    def setUp(self, pc):
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        pcprefix = pc.getOptionsPrefix()
        prefix = pcprefix + self.prefix
        options = PETSc.Options(prefix)

        _, P = pc.getOperators()
        self.mat = P.getPythonContext()
        self.function_space = ensemble_mat.function_space
        self.ensemble = function_space.ensemble

        submats = ensemble_mat.blocks

        self.x = fd.EnsembleFunction(self.function_space)
        self.y = fd.EnsembleCofunction(self.function_space.dual())

        subksps = []
        for i, submat in enumerate(self.mat.blocks):
            ksp = PETSc.KSP().create(comm=ensemble.comm)
            ksp.setOperators(mat)

            sub_prefix = pcprefix + f"sub_{i}_"
            # TODO: default options
            options = OptionsManager({}, sub_prefix)
            options.set_from_options(ksp)
            self.subksps.append((ksp, options))

        self.subksps = tuple(subksps)

    def apply(self, pc, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(xvec)

        subvecs = zip(self.x.subfunctions, self.y.subfunctions)
        for (subksp, suboptions), (subx, suby) in zip(self.subksps, subvecs):
            with subx.dat.vec_ro as rhs, suby.dat.vec_wo as sol:
                with suboptions.inserted_options():
                    subksp.solve(rhs, sol)

        with self.y.vec_ro() as yvec:
            yvec.copy(y)

