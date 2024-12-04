import h5py
from dmrg_runner import DMRGRunner
from read_opts import read_json, unpack_opts

class Simulation:
    def __init__(self, filename: str):
        self.opts_list = unpack_opts(read_json(filename))
        self.results = []

    def run(self):
        # TODO implement multithreading
        self.results = []
        for opts in self.opts_list:
            print("> Running simulation, parameters:")
            print(opts)
            dmrg = DMRGRunner(opts)
            self.results.append(dmrg.run())


    def save(self, filename: str, mode='w-'):
        with h5py.File(f"{filename}.hdf5", mode=mode) as file:
            for result in self.results:
                # Hierarchy:
                # - by l:
                #   - by L:
                #     - by chi:
                dir = f"width_{result.l}/length_{result.L:0>2d}/chi_{result.chi:0>3d}"
                if dir is not file:
                    file.create_group(dir)
                dest = file[dir]
                result.save(dest)

