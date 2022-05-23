from calendar import day_abbr
import os
import runpy
import time
import logging
from datetime import datetime
import argparse
import numpy as np
import scipy.io
from functools import partial
from numbers import Number
import warnings
import skrt

import sirf.STIR as pet
pet.set_verbosity(1)
import sirf.Reg as reg
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import \
    KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm, ScaledFunction, TotalVariation
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from ccpi.filters import regularisers


#############################################################################
#               Helper functions                                            #
#############################################################################

def load_data(data_dir, name):
    # Load Matlab data
    # and convert to 3D Numpy
    # 'data_dir': directory path
    # 'name': for example 'f1b1' (bed position)
    
    dict = {}

    for key in ['acf', 'dtPuc', 'prompts', 'randoms', 'scatter']:
        if os.path.exists('{}/{}_{}.mat'.format(data_dir, key, name)):
            # load from Matlab format to Numpy
            array = scipy.io.loadmat('{}/{}_{}.mat'.format(data_dir, key, name))['data']
            dict[key] = np.transpose(array, (1, 2, 0))
        
    for key in ['norm']:
        # load from Matlab format to Numpy
        array = scipy.io.loadmat('{}/{}.mat'.format(data_dir, key))['data']
        dict[key] = np.transpose(array, (1, 2, 0))
        
    for key in ['decay', 'duration']:
        dict[key] = scipy.io.loadmat('{}/{}_{}.mat'.format(data_dir, key, name))['data']
      
    for key in ['ir3d','pifa']:
        if os.path.exists('{}/{}_{}.mat'.format(data_dir, key, name)):
            # load from Matlab format to Numpy
            array = scipy.io.loadmat('{}/{}_{}.mat'.format(data_dir, key, name))['data']
            dict[key] = np.flip(np.transpose(array, (2,1,0)),axis=1)
        
    return dict

def augment_dim(data):
    "input: array (x,y,z), outputs array (1,x,y,z) with same content"
    augmented_data = np.zeros((1,*data.shape))
    augmented_data[0,:,:,:] = data
    return augmented_data

def pre_process_sinogram(raw_sinos,  num_segs_to_combine, num_views_to_combine):

    if num_segs_to_combine * num_views_to_combine > 1:
        sinos = []
        for raw_sino in raw_sinos:
            sino = raw_sino.rebin(num_segs_to_combine, 
                                num_views_to_combine, 
                                do_normalisation=False)
            print("Rebinned acquisition data dimensions: {}".format(sino.dimensions()))
            sinos.append(sino)
    else:
        sinos = raw_sinos

    return sinos

def FGP_TV_check_input(self, input):
    # work-around CIL's issue #1189
    if len(input.shape) > 3:
        raise ValueError('{} cannot work on more than 3D. Got {}'.format(self.__class__.__name__, input.geometry.length))

def get_proj_norms(K, n, postfix, folder):
    # load or compute and save norm of each sub-operator
    file_path = '{}/normKs_nsub{}_{}.npy'.format(folder, n, postfix)
    if os.path.isfile(file_path):
        print('Norm file {} exists, load it'.format(file_path))
        norms = np.load(file_path, allow_pickle=True).tolist()
    else: 
        print('Norm file {} does not exist, compute it'.format(file_path))
        # normK = [PowerMethod(Ki)[0] for Ki in K]
        norms = [Ki.norm() for Ki in K]
        # save to file
        np.save(file_path, norms, allow_pickle=True)
    return norms

def get_tau(K, prob):
    taus_np = []
    for (Ki,pi) in zip(K,prob):
        tau = Ki.adjoint(Ki.range_geometry().allocate(1.))
        # CD take care of edge of the FOV
        filter = pet.TruncateToCylinderProcessor()
        filter.apply(tau)
        backproj_np = tau.as_array()
        vmax = np.max(backproj_np[backproj_np>0])
        backproj_np[backproj_np==0] = 10 * vmax
        tau_np = 1/backproj_np
        tau.fill(tau_np)
        # apply filter second time just to be sure
        filter.apply(tau)
        tau_np = tau.as_array()
        tau_np[tau_np==0] = 1 / (10 * vmax)
        taus_np.append(pi * tau_np)
    taus = np.array(taus_np)
    tau_np = np.min(taus, axis = 0)
    tau.fill(tau_np)
    return tau

def get_sigmas(K):
    i = 0
    sigma = []
    xx = K.domain_geometry().allocate(1.)
    for Ki in K:
        tmp_np = Ki.direct(xx).as_array()
        tmp_np[tmp_np==0] = 10 * np.max(tmp_np)
        sigmai = Ki.range_geometry().allocate(0.)
        sigmai.fill(1/tmp_np)
        sigma.append(sigmai)
        i += 1
    sigmas = BlockDataContainer(*sigma)
    return sigmas 


def save_callback(save_interval, nifti, outpath, outp_file,
                      num_iter, iteration, obj_value, x):
    """Save callback function.
        File should be saved at "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations)
    """
    completed_iterations = iteration
    if completed_iterations % save_interval == 0 or \
            completed_iterations == num_iter:
        # save current reco
        if nifti==0:
            x.write("{}/{}_iters_{}.hv".format(outpath,outp_file, completed_iterations))
        else:
            reg.NiftiImageData(x).write(
                "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
                

def save_dicom(name, series_description='PET++ IR', ref_dicom=None, description=None):
    "Take path to a nifti file as input, creates folder with DICOM as output"
    nii1 = skrt.Image(name + '.nii')
    # Flip axis
    nii2_data = nii1.get_data().transpose(1, 0, 2)[::-1, ::-1, :]
    nii2_affine = nii1.get_affine(standardise=True)
    nii2 = skrt.Image(path=nii2_data, affine=nii2_affine)

    root_uid = "1.2.826.0.1.3680043.10.937."
    # Fill additional info
    if description is not None:
        series_description += ', ' + description
    if ref_dicom is None:
        patient_name = 'unknown'
        patient_id = 'unknown'
        study_date = 'unknown'
        study_id = 'unknwon'
        study_description = 'unknown'
        study_instance_uid = 'unknown'
        z0 = 0.0
    else:
        # load ref dicom
        skrt_im = skrt.Image(ref_dicom)
        dicom_data = skrt_im.get_dicom_dataset()
        patient_name = str(dicom_data.PatientName)
        if patient_name.startswith('ANON'):
            patient_name = patient_name.strip('ANON')
        patient_id = dicom_data.PatientID
        study_date = dicom_data.StudyDate
        study_id = dicom_data.StudyID
        study_description = dicom_data.StudyDescription
        study_instance_uid = dicom_data.StudyInstanceUID
        z0 = skrt_im.get_dicom_dataset(sl=dicom_data.NumberOfSlices).SliceLocation

    header_extras = {
        'PatientName': patient_name,
        'PatientID': patient_id,
        'StudyDate' : study_date,
        'StudyID' : study_id,
        'StudyDescription' : study_description,
        'StudyInstanceUID' : study_instance_uid,
        'Series Description' : series_description,
        'RescaleSlope': 0.0001,
        }

    # Center z-axis
    nii2_affine[2][3] = z0

    nii2.write( 
        outname=name, 
        root_uid=root_uid, 
        modality='PT', 
        standardise=True, 
        patient_id=patient_id,
        header_extras=header_extras)
