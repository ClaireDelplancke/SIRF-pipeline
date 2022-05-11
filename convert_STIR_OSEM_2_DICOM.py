# %%
import os
import sys
import scipy.io
import numpy as np
import sirf.STIR as pet
from cil.utilities.display import show2D, show_geometry
from cil.utilities.jupyter import islicer
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import \
    KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm, ScaledFunction
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from ccpi.filters import regularisers
from sirf.STIR import ImageData
skrt_path = "/u/s/cd902/GitHub/scikit-rt/src"
sys.path.append(skrt_path)
import skrt
import sirf.Reg as reg
import matplotlib.pyplot as plt
import pydicom
from importlib import reload 
src_path = '{}/src'.format(os.getcwd())
sys.path.append(src_path)
import utils

# %%
patients = ['11972', '12580', '14779', '6910']

# %%
for patient in patients:
    # read STIR OSEM reco
    r = pet.ImageData('/home/cd902/data_azure/Patients/{}/stir_osem/a1_n1_d1_r1_s1/OSEM_48.hv'.format(patient))
    # save as nifti
    reg.NiftiImageData(r).write('/home/cd902/data_azure/Patients/{}/stir_osem/OSEM_48.nii'.format(patient))
    # save as DICOM
    utils.save_dicom('/home/cd902/data_azure/Patients/{}/stir_osem/OSEM_48'.format(patient),
                        series_description='STIR OSEM',
                        ref_dicom='/home/cd902/data_azure/Patients/{}/duetto_reco/offline3D'.format(patient),
                        description='STIR OSEM 2 iter')
# %%
