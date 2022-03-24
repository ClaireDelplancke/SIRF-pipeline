"""
Author: Claire Delplancke, University of Bath
Member of the PET++ project: https://petpp.github.io
"""

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



def main():

    #############################################################################
    #               Arguments                                                   #
    #############################################################################

    parser = argparse.ArgumentParser(description="GE PET reconstruction")
    
    # input / output
    parser.add_argument("--scanner", type=str, 
                        default='GE Discovery 690')
    parser.add_argument("--span", type=int, 
                        default=2, help='GE scanners need 2')
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

    # model parameters
    parser.add_argument("--reg", help="Regularisation", 
                        default='FGP-TV',  type=str, 
                        choices = ['FGP-TV' ,'FGP_TV' ,'None', 'Explicit-TV', 'CIL-TV'])
    parser.add_argument("--reg_strength", help="Parameter of regularisation", 
                        default=0.1, type=float)
    parser.add_argument("--lor", 
                        help="Number of tangential LOR. Increases computational time",
                        default=10)
    parser.add_argument("--numSegsToCombine", 
                        help = "Rebin all sinograms, with a given number of segments to combine. Decreases computational time.",
                        type=int, default = 1)
    parser.add_argument("--numViewsToCombine",
                        help = "Rebin all sinograms, with a given number of view to combine. Decreases computational time.",
                        type=int, default = 1)
    parser.add_argument('--fast', action='store_true',
                        help="Decreases computational time by rebinning data and halving reconstructed image dimensons")

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
                        default=100, type=int)
    parser.add_argument("--nobj", help="Frequency to which compute objective", 
                        default=50, type=int)
    parser.add_argument("--nifti", type=int, default=0, help="Save reconstruction in nifti format")
    args = parser.parse_args()

    ###########################################################################
    # Create output folders
    ###########################################################################

    folder_log = '{}/logs'.format(os.getcwd())
    for folder in [args.folder_output, folder_log]:
        if not os.path.exists(folder):
            logging.info('Create output or log folder')
            os.makedirs(folder)
    os.chdir(args.folder_output)

    ###########################################################################
    # Create logger
    ###########################################################################

    # logging options
    logfile = '{}/log_{}.log'.format(folder_log, datetime.now().strftime("%Y_%m_%d_%H:%M"))
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
    # Set-up projection parameters
    ###########################################################################
    
    logging.info('Set-up projection parameters')

    # Set-up acq data template using scanner name
    scanner_name = args.scanner
    span = args.span
    pet.AcquisitionData.set_storage_scheme('memory')
    raw_acq_data_template = pet.AcquisitionData(scanner_name,span=span)
    
    if args.fast:
        logging.warning("The '--fast' option is on, supercedes '--lor', '--nxny', '--numSegsToCombine' and '--numViewsToCombine' ")
        lor = 2
        nxny = int(np.ceil(raw_acq_data_template.create_uniform_image(0.0).shape[1]/2))
        num_segs_to_combine = 5
        num_views_to_combine = 4
    else:
        lor = args.lor
        nxny = args.nxny
        num_segs_to_combine = args.numSegsToCombine
        num_views_to_combine = args.numViewsToCombine

    ###########################################################################
    # Create SIRF objects
    ###########################################################################

    logging.info('Create SIRF objects')
    
    # Create and fill SIRF acq data objects
    sinogram_raw = raw_acq_data_template.clone()
    multfact_raw = raw_acq_data_template.clone()
    addfact_raw = raw_acq_data_template.clone() 
    sinogram_raw.fill(data_dict['prompts'])
    multfact_raw.fill(multfact_data)
    addfact_raw.fill(addfact_data)

    # Rebin
    sinogram, multfact, addfact = pre_process_sinogram(
        [sinogram_raw, multfact_raw, addfact_raw], 
        num_segs_to_combine,
        num_views_to_combine)

    # Initial image
    image_template = raw_acq_data_template.create_uniform_image(0.0, xy=nxny)


    ###########################################################################
    # Set-up acquisition model
    ###########################################################################

    logging.info('Set-up acquisition model')

    # number of subsets
    num_subsets = args.nsubsets 
    # set-up acquisition model
    acq_models = [pet.AcquisitionModelUsingRayTracingMatrix() for k in range(num_subsets)]
    # create masks
    im_one = image_template.clone()
    im_one.fill(1.)
    masks = []
    asm = pet.AcquisitionSensitivityModel(multfact)

    # Loop over physical subsets
    for k in range(num_subsets):
        # Set up
        acq_models[k].set_num_tangential_LORs(lor)
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
    r_alpha = args.reg_strength
    r_iters = 100
    r_tolerance = 1e-7
    if args.reg == "FGP_TV" or args.reg == "FGP-TV":
        logging.warning("With the FGP_TV option, the gradient is defined as the finite difference operator (voxel-size not taken into account)")
        r_iso = 1
        r_nonneg = 1
        device = 'gpu'
        G = FGP_TV(r_alpha, r_iters, r_tolerance,
                r_iso, r_nonneg, device)
        if args.precond==1:
            raise ValueError("Precond option not compatible with FGP-TV regularizer")
            FGP_TV.proximal = precond_proximal
        # redefines check_input which gives error
        FGP_TV.check_input = FGP_TV_check_input
    elif args.reg == "CIL-TV":
        logging.warning("With the CIL-TV option, the gradient is defined as the finite difference operator divided by the voxel-size in each direction")
        G = r_alpha * TotalVariation(r_iters, r_tolerance, lower=0)
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
        normKs = None

    
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

    # output_name = 'spdhg_reg_{}_alpha{}_nsub{}_precond{}_gamma{}_a{}_n{}_d{}_r{}_s{}'.format(
    #         args.reg, args.reg_strength, 
    #         num_subsets, args.precond, args.pd_par, int(args.acf), 
    #         int(args.normf), int(args.dtpucf), int(args.randoms), int(args.scatter)
    #         )
    output_name = 'spdhg_reg_{}_nsub{}_precond{}_a{}_n{}_d{}_r{}_s{}'.format(
        args.reg, 
        num_subsets, args.precond, int(args.acf), 
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
    # Save metadata and obj values
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
    np.save('{}/{}_metadata'.format(args.folder_output,output_name), metadata_dict)
    
    logging.info('save objective values')
    np.save('{}/{}_objective'.format(args.folder_output, output_name), spdhg.objective)

    ###########################################################################
    # End of main()
    ###########################################################################

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
            x.write("{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
        else:
            reg.NiftiImageData(x).write(
                "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
    
        
if __name__ == "__main__":
    main()
        
