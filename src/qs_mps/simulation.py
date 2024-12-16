import h5py
from dmrg_runner import DMRGRunner, DMRGData
from read_opts import read_json, unpack_opts



class Simulation:
    def __init__(self, json_filename: str):
        file_contents = read_json(json_filename)
        if "filename" not in file_contents:
            raise ValueError(f"Filename missing in {json_filename}")
        self.filename = file_contents.pop("filename")
        self.opts_list = unpack_opts(file_contents)
        self.results = []


    def run(self, save_progress: bool = True):
        # TODO OPTIONAL implement multithreading
        self.results = []
        for opts in self.opts_list:
            print("> Running simulation, parameters:\n\n")
            print(opts)
            dmrg = DMRGRunner(opts)
            result = dmrg.run()
            self.results.append(result)
            if save_progress:
                self.save_instance(result, opts)
            print("\n ****** ")


    def group_name(self, opts: dict):
        # Hierarchy:
        # - by l:
        #   - by L:
        #     - by chi:
        #       - by string_length (if present):
        gname = f"width_{opts['l']}/" + \
            f"length_{opts['L']:0>2d}/" + \
            f"chi_{opts['chi']:0>3d}/"
        if "string_length" in opts:
            gname = gname + f"string_length_{opts['string_length']:0>2d}/"
        return gname


    def save_instance(self, instance_result: DMRGData, opts: dict, mode="a"):
        # TODO Implement save_tensors
        print(f"*** save_instance opts:\n {opts}\n****")
        group_name = self.group_name(opts)
        with h5py.File(f"{self.filename}.hdf5", mode=mode) as file:
            if group_name is not file:
                file.create_group(group_name)
            dest = file[group_name]
            instance_result.save(dest)


    def save(self, filename: str, mode='w-'):
        for result, opts in zip(self.results, self.opts):
            self.save_instance(result, opts)
