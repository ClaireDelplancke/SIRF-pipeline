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




def CIL_TV_proximal(self, x, tau, out = None):
    
    ''' Returns the solution of the FGP_TV algorithm '''         
    try:
        self._domain = x.geometry
    except:
        self._domain = x
    
    # initialise
    t = 1
    if not self.warmstart:   
        self.tmp_p = self.gradient.range_geometry().allocate(0)  
        self.tmp_q = self.tmp_p.copy()
        self.tmp_x = self.gradient.domain_geometry().allocate(0)     
        self.p1 = self.gradient.range_geometry().allocate(0)
    else:
        if not self.hasstarted:
            self.tmp_p = self.gradient.range_geometry().allocate(0)  
            self.tmp_q = self.tmp_p.copy()
            self.tmp_x = self.gradient.domain_geometry().allocate(0)     
            self.p1 = self.gradient.range_geometry().allocate(0)
            self.hasstarted = True

    should_break = False
    for k in range(self.iterations):
                                                                                
        t0 = t
        self.gradient.adjoint(self.tmp_q, out = self.tmp_x)
        
        # axpby now works for matrices
        self.tmp_x.sapyb(-self.regularisation_parameter*tau, x, 1.0, out=self.tmp_x)
        self.projection_C(self.tmp_x, out = self.tmp_x)                       

        self.gradient.direct(self.tmp_x, out=self.p1)
        if isinstance (tau, (Number, np.float32, np.float64)):
            self.p1 *= self.L/(self.regularisation_parameter * tau)
        else:
            self.p1 *= self.L/self.regularisation_parameter
            self.p1 /= tau

        if self.tolerance is not None:
            
            if k%5==0:
                error = self.p1.norm()
                self.p1 += self.tmp_q
                error /= self.p1.norm()
                if error<=self.tolerance:                           
                    should_break = True
            else:
                self.p1 += self.tmp_q
        else:
            self.p1 += self.tmp_q
        if k == 0:
            # preallocate for projection_P
            self.pptmp = self.p1.get_item(0) * 0
            self.pptmp1 = self.pptmp.copy()

        self.projection_P(self.p1, out=self.p1)
        

        t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
        
        #self.tmp_q.fill(self.p1 + (t0 - 1) / t * (self.p1 - self.tmp_p))
        self.p1.subtract(self.tmp_p, out=self.tmp_q)
        self.tmp_q *= (t0-1)/t
        self.tmp_q += self.p1
        
        self.tmp_p.fill(self.p1)

        if should_break:
            break
    
    #clear preallocated projection_P arrays
    self.pptmp = None
    self.pptmp1 = None
    
    # Print stopping information (iterations and tolerance error) of FGP_TV     
    if self.info:
        if self.tolerance is not None:
            print("Stop at {} iterations with tolerance {} .".format(k, error))
        else:
            print("Stop at {} iterations.".format(k))                
        
    if out is None:                        
        self.gradient.adjoint(self.tmp_q, out=self.tmp_x)
        self.tmp_x *= tau
        self.tmp_x *= self.regularisation_parameter 
        x.subtract(self.tmp_x, out=self.tmp_x)
        return self.projection_C(self.tmp_x)
    else:          
        self.gradient.adjoint(self.tmp_q, out = out)
        out*=tau
        out*=self.regularisation_parameter
        x.subtract(out, out=out)
        self.projection_C(out, out=out)