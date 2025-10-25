from .pt_dataset import PTDataset
from .dict_dataset import DictDataset
from .mesh_datamodule import MeshDataModule
from .car_cfd_dataset import CarCFDDataset
from .seismic_dataset import SeismicDataset, SeismicDataProcessor
from .zarr_seismic_dataset import ZarrSeismicDataset

# only import TheWell if the_well is built
try:
    from .the_well_dataset import (TheWellDataset,
                           ActiveMatterDataset,
                           MHD64Dataset)
except ModuleNotFoundError:
    pass