from torch_geometric.data import Data, Dataset, download_url


class ANIMD(Dataset):

    @property
    def raw_url(self):
        return 'https://raw.githubusercontent.com/isayev/COMP6/master/COMP6v1/ANI-MD/ani_md_bench.h5'

    def download(self):
        download_url(self.raw_url, self.raw_dir)

    @property
    def raw_file_names(self):
        return 'ani_md_bench.h5'


class DrugBank(Dataset):

    @property
    def raw_url(self):
        return 'https://raw.githubusercontent.com/isayev/COMP6/master/COMP6v1/DrugBank/drugbank_testset.h5'

    def download(self):
        download_url(self.raw_url, self.raw_dir)

    @property
    def raw_file_names(self):
        return 'drugbank_testset.h5'


class GDB07to09(Dataset):

    @property
    def raw_url(self):
        return [f'https://raw.githubusercontent.com/isayev/COMP6/master/COMP6v1/GDB07to09/{name}' for name in self.raw_file_names]

    def download(self):
        for url in self.raw_url:
            download_url(self.url, self.raw_dir)

    @property
    def raw_file_names(self):
        return ['gdb11_07_test500.h5', 'gdb11_08_test500.h5', 'gdb11_09_test500.h5']


class GDB10to13(Dataset):

    @property
    def raw_url(self):
        return [f'https://raw.githubusercontent.com/isayev/COMP6/master/COMP6v1/GDB10to13/{name}' for name in self.raw_file_names]

    def download(self):
        for url in self.raw_url:
            download_url(self.url, self.raw_dir)

    @property
    def raw_file_names(self):
        return ['gdb11_10_test500.h5', 'gdb11_11_test500.h5', 'gdb13_12_test500.h5', 'gdb13_13_test500.h5']


class Tripeptides(Dataset):

    @property
    def raw_url(self):
        return 'https://raw.githubusercontent.com/isayev/COMP6/master/COMP6v1/Tripeptides/tripeptide_full.h5'

    def download(self):
        download_url(self.raw_url, self.raw_dir)

    @property
    def raw_file_names(self):
        return 'tripeptide_full.h5'


class S66X8(Dataset):

    @property
    def raw_url(self):
        return 'https://raw.githubusercontent.com/isayev/COMP6/master/COMP6v1/s66x8/s66x8_wb97x6-31gd.h5'

    def download(self):
        download_url(self.raw_url, self.raw_dir)

    @property
    def raw_file_names(self):
        return 's66x8_wb97x6-31gd.h5'