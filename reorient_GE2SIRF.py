# %% Import libraries

import os
import sys
import scipy.io
import numpy as np
import sirf.STIR as pet
from cil.utilities.display import show2D, show_geometry
import skrt
import sirf.Reg as reg
import matplotlib.pyplot as plt
import pydicom
import argparse

def reorient(ref_path):
    '''
    Save the DICOM image as a .nii image which has the same orientation 
    as a STIR-SIRF PET .nii reconstruction
    if both are read with sirf.STIR.ImageData
    '''

    ref_image = skrt.Image(ref_path)
    ref_data = ref_image.get_standardised_data()
    rref_data = ref_data[::-1,:,::-1]
    rref_image = skrt.Image(path=rref_data, affine=ref_image.get_affine(standardise=True))
    rref_image.write(outname='{}/reoriented.nii'.format(ref_path), standardise=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save GE CT or MR DICOM to .nii with STIR-SIRF PET orientation if read with SIRF')
    parser.add_argument('--dicom_path', metavar='path', required=True,
                        help='path to DICOM folder')
    args = parser.parse_args()
    reorient(args.dicom_path)

