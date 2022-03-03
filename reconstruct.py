"""
Author: Claire Delplancke, University of Bath
Member of the PET++ project: https://petpp.github.io
"""

from calendar import day_abbr
import os
import runpy
import time
#  Run prerequisite files
os.chdir('/u/s/cd902/GitHub/SIRF-pipeline')
import logging
from datetime import datetime
import argparse
import numpy as np
import scipy
from functools import partial


import sirf.STIR as pet
import sirf.Reg as reg
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import \
    KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm, ScaledFunction
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from ccpi.filters import regularisers



def main():

    #############################################################################
    #               Arguments                                                   #
    #############################################################################

    parser = argparse.ArgumentParser(description="GE PET reconstruction")
    
    # input / output
    parser.add_argument("--type", type=str, 
                        default='PET-CT',
                        choices=['PET-CT','PET-MR'])
    parser.add_argument("--folder_input", type=str)
    parser.add_argument("--duetto_prefix", default='f1b1', type=str)
    parser.add_argument("--folder_output", type=str)

    # helper to set mutually exclusive flags
    def add_bool_arg(parser, name, default=True, helpp=None):
        if helpp is None:
            helpp = name
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true',
                           help='{} flag'.format(helpp))
        group.add_argument('--no-' + name, dest=name, action='store_false',
                           help='No {} flag'.format(helpp))
        parser.set_defaults(**{name:default})
    
    # corrective factors
    for name, helpp in zip(['acf','normf','dtpucf','randoms','scatter'],
                          ['attenuation correction factor', 'normalisation',
                          'dead-time and pile-up correction', 'randoms',
                          'scatter']):
        add_bool_arg(parser, name, helpp=helpp)

    # size of reconstructed image
    parser.add_argument("--nxny", help="Number of voxels", 
                        type=int)
    parser.add_argument("--dxdy", help="Size of voxels", 
                        type=float)

    # model parameters
    parser.add_argument("--reg", help="Regularisation", 
                        default='FGP-TV',  type=str, 
                        choices = ['FGP-TV','None', 'Explicit-TV'])
    parser.add_argument("--reg_strength", help="Parameter of regularisation", 
                        default=0.1, type=float)
    
    # reconstruction parameters
    parser.add_argument("--nepoch", help="Number of epochs", 
                        default=2, type=int)
    parser.add_argument("--nsubsets", help="Number of subsets", 
                        default=24, type=int)
    parser.add_argument("--pd_par", help="Primal-dual balancing parameter gamma", 
                        default=1.0, type=float) 
    parser.add_argument("--precond", help="Preconditioning flag", 
                        default=0, type=int) 
    
    # output parameters
    parser.add_argument("--nsave", help="Frequency to which save iterative recos", 
                        default=50, type=int)
    parser.add_argument("--nobj", help="Frequency to compute objective functions", 
                        default=50, type=int) 
    parser.add_argument("--nifti", type=int, default=0, help="Save reconstruction in nifti format")
    args = parser.parse_args()

    ###########################################################################
    # Create param and output folders
    ###########################################################################

    for folder in [args.folder_output, './logs']:
        if not os.path.exists(folder):
            logging.info('Create output or log folder')
            os.makedirs(folder)

    ###########################################################################
    # Create logger
    ###########################################################################

    # logging options
    logfile = 'logs/log_{}.log'.format(datetime.now().strftime("%Y_%m_%d_%H:%M"))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(logfile, 'a'))

    ###########################################################################
    # Load data
    ###########################################################################

    logging.info('Load data')
    data_dict = load_data(args.folder_input, args.duetto_prefix)

    ###########################################################################
    # Define additive and multiplicative correction factors
    ###########################################################################

    logging.info('Define correction factors')
    data_shape = data_dict['prompts'].shape
    multfact_data = np.ones(data_shape)
    if args.acf == True:
        multfact_data *= data_dict['acf']
    if args.normf == True:
        multfact_data *= 1/data_dict['norm']
    if args.dtpucf == True:
        multfact_data *= 1/data_dict['dtPuc']
    addfact_data = np.zeros(data_shape)
    if args.randoms == True:
        addfact_data += data_dict['randoms']
    if args.scatter == True:
        addfact_data += data_dict['scatter']

    
    ###########################################################################
    # Create SIRF objects
    ###########################################################################

    logging.info('Create SIRF objects')
    # Set-up aqc data template using scanner name
    if args.type == 'PET-CT':
        scanner_name = 'GE Discovery 690'
    elif args.type == 'PET-MR':
        scanner_name = 'GE Signa PET/MR'
    acq_data_template = acq_data_from_scanner_name(scanner_name)
    
    # Create and fill SIRF acq data objects
    sinogram = acq_data_template.clone()
    multfact = acq_data_template.clone()
    addfact = acq_data_template.clone()  
    sinogram.fill(data_dict['prompts'].flat)
    multfact.fill(multfact_data.flat)
    addfact.fill(addfact_data.flat)

    # Initial image
    image_template = acq_data_template.create_uniform_image(0.0)
    if args.dxdy is not None:
        logging.warning('Option -dxdy not implemented')
    if args.nxny is not None:
        logging.warning('Option -dxdy not implemented')

    ###########################################################################
    # Set-up acquisition model
    ###########################################################################

    logging.info('Set-up acquisition model')

    # number of subsets
    num_subsets = args.nsubsets 
    # set-up acquisition model
    acq_models = [pet.AcquisitionModelUsingParallelproj() for k in range(num_subsets)]
    # create masks
    im_one = image_template.clone()
    im_one.fill(1.)
    masks = []
    asm = pet.AcquisitionSensitivityModel(multfact)

    # Loop over physical subsets
    for k in range(num_subsets):
        # Set up
        acq_models[k].set_acquisition_sensitivity(asm)
        acq_models[k].set_up(sinogram, image_template)    
        acq_models[k].num_subsets = num_subsets
        acq_models[k].subset_num = k 

        # compute masks 
        mask = acq_models[k].direct(im_one)
        masks.append(mask)

    ###########################################################################
    # Set-up F, G and K
    ###########################################################################

    logging.info('Set-up F, G and K')
    data_fits = [KullbackLeibler(b=sinogram, eta=addfact, mask=mask.as_array(), use_numba=True) for mask in masks]
    if args.reg == "FGP_TV":
        r_alpha = args.reg_strength
        r_iters = 100
        r_tolerance = 1e-7
        r_iso = 0
        r_nonneg = 1
        r_printing = 0
        device = 'gpu'
        G = FGP_TV(r_alpha, r_iters, r_tolerance,
                r_iso, r_nonneg, r_printing, device)
        if args.precond==1:
            FGP_TV.proximal = precond_proximal
    elif args.reg == "None":
        G = IndicatorBox(lower=0)
    elif args.reg == "Explicit-TV":
        raise ValueError("Not implemented at the moment")
    else:
        raise ValueError("Unknown regularisation")

    F = BlockFunction(*data_fits)
    K = BlockOperator(*acq_models)


    ###########################################################################
    # Set-up step-sizes and probabilities
    ###########################################################################

    logging.info('Set-up step-sizes')
    # uniform probabilities
    prob = [1/num_subsets] * num_subsets
    # primal-dual balancing parameter
    if args.precond == 0:
        # XXX mysterious axpby set-up
        use_axpby = True
        # compute the norm of each component
        postfix = "a{}_n{}_d{}".format(
            int(args.acf), 
            int(args.normf), 
            int(args.dtpucf))
        normKs = get_proj_norms(K, num_subsets, postfix, args.folder_output)
        # let spdhg do its default implementation
        sigmas = None
        tau = None
        gamma = args.pd_par
    else:
        logging.info('Preconditioning is on, no need to compute op norms')
        # XXX mysterious axpby set-up
        use_axpby = False
        tau = 1/args.pd_par * get_tau(K, prob)
        sigmas = args.pd_par * get_sigmas(K)
        gamma = None
    
    ###########################################################################
    # Set-up SPDHG
    ###########################################################################
    logging.info('Set-up SPDHG')

    # number of iterations
    num_iter = args.nepoch * num_subsets
    num_save = args.nsave * num_subsets
    num_obj = args.nobj * num_subsets

    spdhg = SPDHG(            
                f=F, 
                g=G, 
                operator=K,
                tau=tau,
                sigma=sigmas,
                gamma=gamma,
                prob=prob,
                use_axpby=use_axpby,
                norms=normKs,
                max_iteration=num_iter,         
                update_objective_interval=num_obj,
                log_file=logfile,
                )

    output_name = 'spdhg_reg_{}_alpha{}_nsub{}_precond{}_gamma{}_a{}_n{}_d{}_r{}_s{}.npy'.format(
            args.reg, args.reg_strength, 
            num_subsets, args.precond, args.pd_par, int(args.acf), 
            int(args.normf), int(args.dtpucf), int(args.randoms), int(args.scatter)
            )

    psave_callback = partial(
        save_callback, num_save, args.nifti, args.folder_output, output_name, num_iter)

    ###########################################################################
    # Run SPDHG
    ###########################################################################

    logging.info('Run SPDHG')
    spdhg.run(num_iter, verbose=2, print_interval=1, callback=psave_callback)

    ###########################################################################
    # Save metadata
    ###########################################################################

    logging.info('save metadata')
    metadata_dict = {}
    # parameters of the mathematical reconstruction
    metadata_dict['regularizer'] = args.reg
    metadata_dict['regularization  strength'] = args.reg_strength
    metadata_dict['number of subsets'] = num_subsets
    metadata_dict['primal-dual parameter gamma'] = args.pd_par
    # parameters of the physical reconstruction
    metadata_dict['attenuation correction'] = args.acf
    metadata_dict['normalization'] = args.normf
    metadata_dict['dead-time and pile-up correction'] = args.dtpucf
    metadata_dict['randoms'] = args.randoms
    metadata_dict['scatter'] = args.scatter
    # save
    np.save(args.folder_output + '/metadata', metadata_dict)

def load_data(data_dir, name):
    # Load Matlab data outputted by Duetto
    # and convert it to 3D Numpy
    # 'data_dir': directory path
    # 'name': for example 'f1b1'
    
    dict = {}

    for key in ['acf', 'dtPuc', 'prompts', 'randoms', 'scatter']:
        if os.path.exists('{}/{}_{}.mat'.format(data_dir, key, name)):
            # load from Matlab format to Numpy
            array = scipy.io.loadmat('{}/{}_{}.mat'.format(data_dir, key, name))['data']
            dict[key] = np.transpose(array, (1, 2, 0))
        
    for key in ['norm']:
        # load from Matlab format to Numpy
        array = scipy.io.loadmat('{}/{}.mat'.format(data_dir, key))['data']
        # Numpy to ODL
        dict[key] = np.transpose(array, (1, 2, 0))
        
    for key in ['decay', 'duration']:
        # load from Matlab format to Numpy
        dict[key] = scipy.io.loadmat('{}/{}_{}.mat'.format(data_dir, key, name))['data']
      
    for key in ['ir3d','pifa']:
        if os.path.exists('{}/{}_{}.mat'.format(data_dir, key, name)):
            # load from Matlab format to Numpy
            array = scipy.io.loadmat('{}/{}_{}.mat'.format(data_dir, key, name))['data']
            dict[key] = np.flip(np.transpose(array, (2,1,0)),axis=1)
        
    return dict


# XXX
# XXX solve me
def acq_data_from_scanner_name(scanner_name):
    return NotImplemented


def precond_proximal(self, x, tau, out=None):

    """Modify proximal method to work with preconditioned tau"""
    pars = {'algorithm': FGP_TV,
            'input': np.asarray(x.as_array()/tau.as_array(),
                                dtype=np.float32),
            'regularization_parameter': self.lambdaReg,
            'number_of_iterations': self.iterationsTV,
            'tolerance_constant': self.tolerance,
            'methodTV': self.methodTV,
            'nonneg': self.nonnegativity,
            'printingOut': self.printing}

    res = regularisers.FGP_TV(pars['input'],
                                    pars['regularization_parameter'],
                                    pars['number_of_iterations'],
                                    pars['tolerance_constant'],
                                    pars['methodTV'],
                                    pars['nonneg'],
                                    self.device)[0]
    if out is not None:
        out.fill(res)
    else:
        out = x.copy()
        out.fill(res)
    out *= tau
    return out 

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
                      num_iter, iteration, x):
    """Save callback function.
        File should be saved at "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations)
    """
    completed_iterations = iteration
    if completed_iterations % save_interval == 0 or \
            completed_iterations == num_iter:
        if nifti==0:
            x.write("{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
        else:
            reg.NiftiImageData(x).write(
                "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))


    
        
if __name__ == "__main__":
    main()
        
