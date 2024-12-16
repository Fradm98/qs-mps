import numpy as np
import h5py

from qs_mps.utils import get_precision
from qs_mps.applications.Z2.ground_state_multiprocessing import ground_state_Z2


class DMRGRunner:
    def __init__(self, opts: dict):
        """
        TODO DOCUMENTATION
        """
        # TODO DOCUMENTATION
        # TODO OPTIONAL reduce complexity

        # questi devono essere obbligatori
        assert 'l' in opts, "Missing number of ladders (vertical dimension)"
        assert 'L' in opts, "Missing the horizontal dimension"
        assert 'chi' in opts, "Missing the bond dimension"
        self.l = int(opts.get("l"))
        self.L = int(opts.get("L"))
        self.chi = int(opts.get("chi"))
        self.bc = str(opts.get("bc", "obc"))

        # Not generalized
        self.model = opts.get("model", "Z2_dual")
        if self.model == "Z2_dual":
            self.d = int(2**(self.l))
        else:
            raise ValueError(f"Model {self.model} unrecognized, do not know the physical dimension")

        # couplings and number of points
        self.npoints = int(opts.get("npoints", 50))
        self.h_i = float(opts.get("h_i", 0.1))
        self.h_f = float(opts.get("h_f", ))
        self.interval_type = opts.get("interval_type", "lin")
        if self.interval_type not in ("lin", "log"):
            # default to "lin"
            self.interval_type = "lin"
        if self.interval_type == "lin":
            self.interval = np.linspace(self.h_i, self.h_f, self.npoints)
            num = (self.interval[-1] - self.interval[0]) / self.npoints
            self.precision = get_precision(num)
        elif self.interval_type == "log":
            self.interval = np.logspace(self.h_i, self.h_f, self.npoints)
            self.precision = int(np.max([np.abs(self.h_f),np.abs(self.h_i)]))


        # Device and run options
        self.device = opts.get("device", "pc")
        # take the path and precision to save files
        # if we want to save the tensors we save them locally because they occupy a lot of memory
        if self.device == "pc":
            self.parent_path = "C:/Users/HP/Desktop/projects/1_Z2"
            self.path_tensor = "D:/code/projects/1_Z2"
        elif self.device == "mac":
            self.path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
            self.parent_path = self.path_tensor
        elif self.device == "marcos":
            self.path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
            self.parent_path = self.path_tensor
        else:
            raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")
        self.multiprocessing = bool(opts.get("multiprocessing", False))

        # Charges and strings
        self.charges = opts.get('charges', None)
        self.string_length = opts.get('string_length', None)

        # If string_length is defined, overwrite charges position
        if self.string_length is not None:
            sl = self.string_length
            Lhalf = self.L // 2
            pos_y = self.l // 2 if self.bc == "obc" else 0
            self.charges = [[Lhalf - sl//2, pos_y], [Lhalf + sl//2, pos_y]]

        if type(self.charges) == list and len(self.charges) > 0:
            for val in self.charges:
                if type(val) != list and type(val) != tuple:
                    raise ValueError(f"List of charges is not valid, element {val} not recognized")
            self.sector = f"{len(self.charges)}_particle(s)"
            self.charges_x = [ x[0] for x in self.charges ],
            self.charges_y = [ x[1] for x in self.charges ],
        elif self.charges is None or self.charges == []:
            self.sector = "vacuum_sector"
            self.charges_x = None
            self.charges_y = None
        else:
            raise ValueError("Unrecognized option for list of charges")


        # DMRG options
        self.init_tensor = opts.get("init_tensor", [])
        self.nsweeps = opts.get("nsweeps", 5)
        self.type_shape = opts.get("type_shape", "rectangular")
        # Allowed values for `which_bond`: 'middle', int, 'all'
        which_bond_ = opts.get("which_bond", "middle")
        if which_bond_ == "middle":
            self.which_bond = self.L // 2
            self.bond = True
        elif type(which_bond_) == int:
            self.which_bond = which_bond_
            self.bond = True
        elif which_bond_ == "all":
            self.which_bond = -2 #magic number
            self.bond = False
        else:
            raise ValueError(f"Invalid value of `which_bond` = {which_bond_}")

        self.conv_tol = opts.get("conv_tol", 1e-12)
        self.training = opts.get("training", True)
        self.save_tensors = opts.get("save_tensors", True)
        self.trunc_tol = opts.get("trunc_tol", False)
        self.trunc_chi = opts.get("trunc_chi", True)


    def create_args_mps(self):
        args_mps = {
            "L": self.L,
            "d": self.d,
            "l": self.l,
            "chi": self.chi,
            "type_shape": self.type_shape,
            "model": self.model,
            "trunc_tol": self.trunc_tol,
            "trunc_chi": self.trunc_chi,
            "where": self.which_bond,
            "bond": self.bond,
            "path": self.path_tensor,
            "save": self.save_tensors,
            "precision": self.precision,
            "sector": self.sector,
            "charges_x": self.charges_x,
            "charges_y": self.charges_y,
            "n_sweeps": self.nsweeps,
            "conv_tol": self.conv_tol,
            "training": self.training,
            "guess": self.init_tensor,
            "bc": self.bc,
            "multiprocessing": self.multiprocessing
        }
        return args_mps

    def run(self):
        # DONE: in case of multiprocessing compute only the total time inside DMRGRunner
        args_mps = self.create_args_mps()
        energy, \
        entropy, \
        schmidt_vals, \
        t_dmrg, \
        tensors = ground_state_Z2(
            args_mps=args_mps,
            multpr=self.multiprocessing,
            interval=self.interval
        )

        print(f" >>> tensors length: {len(tensors)}")
        print(f" >>> tensors[0] length: {len(tensors[0])}")

        return DMRGData(
            args_mps=args_mps,
            interval=self.interval,
            energies=energy,
            entropies=entropy,
            schmidt_vals=schmidt_vals,
            t_dmrg=t_dmrg,
            tensors=tensors
        )


class DMRGData:
    def __init__(self, args_mps, interval, energies, entropies, schmidt_vals, t_dmrg, tensors):
        self.args_mps = args_mps
        self.interval = interval
        self.energies = energies
        self.entropies = entropies
        self.schmidt_vals = schmidt_vals
        self.times = t_dmrg
        self.tensors = tensors


    def save(self, obj: str | h5py.File | h5py.Group, mode='w-'):
        """
        Availables modes
        r         Readonly, file must exist (default)
        r+        Read/write, file must exist
        w         Create file, truncate if exists
        w- or x   Create file, fail if exists
        a         Read/write if exists, create otherwise
        """
        if type(obj) == str:
            file = h5py.File(f"{obj}.hdf5", mode)
        if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
            file = obj
        file.attrs['interval'] = self.interval
        self.save_energy(file)
        self.save_time(file)
        # Save an array of entropies, one for each coupling
        if self.args_mps['bond']:
            self.save_entropy(file, coupling_index="all")
            self.save_schmidt_vals(file, coupling_index="all")
        for k, coupling in enumerate(self.interval):
            print(f" ** saving for k={k} (g = {coupling})")
            group = file.create_group(f"coupling_index_{k}")
            group.attrs['coupling'] = coupling
            if self.args_mps['training']:
                self.save_training(group, coupling_index=k)
            if not self.args_mps['bond']:
                self.save_entropy(group, coupling_index=k)
                self.save_schmidt_vals(group, coupling_index=k)


    def save_training(self, file: h5py.File, coupling_index: int):
        train_group = file.create_group('training')
        train_group.create_dataset('energies', data=self.energies[coupling_index], dtype=np.float64)

    def save_schmidt_vals(self, file, coupling_index):
        print(f"    * schmidt_vals type: {type(self.schmidt_vals)}")
        # print(f"    * schmidt_vals shape: {np.asarray(self.schmidt_vals).shape}")
        if coupling_index == "all":
            file.attrs['which_bond'] = self.args_mps['where']
            file.attrs['bond_dim'] = self.args_mps['chi']
            file.create_dataset("schmidt_vals", data=self.schmidt_vals)
        else:
            # TODO create a group for each bond
            sv = self.schmidt_vals[coupling_index]
            for i, val in enumerate(sv):
                bond_group = file.create_group(f"bond_{i:0>3d}")
                bond_group.create_dataset("schmidt_vals", data=val)
                bond_group.attrs['bond_dim'] = len(val)
            file.attrs['nbonds'] = len(self.schmidt_vals[coupling_index])
            # file.attrs['bond_dim'] = self.args_mps['chi']
            # file.create_dataset("schmidt_vals_by_bond", data=self.schmidt_vals[coupling_index])

    def save_entropy(self, file, coupling_index="all"):
        print(f"    * entropies type: {type(self.entropies)}")
        print(f"    * entropies shape: {np.asarray(self.entropies).shape}")
        if coupling_index == "all":
            file.attrs['which_bond'] = self.args_mps['where']
            file.create_dataset("entropies", data=self.entropies)
        else:
            file.attrs['nbonds'] = len(self.entropies[coupling_index])
            file.create_dataset("entropies_by_bond", data=self.entropies[coupling_index])

    def save_energy(self, file):
        if self.args_mps['training']:
            energies = [ v[-1] for v in self.energies]
        else:
            energies = self.energies
        file.create_dataset("energies", data=energies)

    def save_time(self, file):
        if not self.args_mps['multiprocessing']:
            file.create_dataset("times", data=self.times)

    @property
    def l(self):
        return self.args_mps['l']

    @property
    def L(self):
        return self.args_mps['L']

    @property
    def chi(self):
        return self.args_mps['chi']


    def __repr__(self):
        return f"<DMRGData for l={self.l} L={self.L} chi={self.chi}>"
