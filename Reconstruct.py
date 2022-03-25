"""
Author: Claire Delplancke, University of Bath, PET++ project
2022
https://clairedelplancke.github.io
https://petpp.github.io
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

# Redefines CIL_TV to allow warm-start

def CIL_TV__init__(self,
                max_iteration=100, 
                tolerance = None, 
                correlation = "Space",
                backend = "c",
                lower = -np.inf, 
                upper = np.inf,
                isotropic = True,
                split = False,
                info = False,
                warmstart = False):
    

    super(TotalVariation, self).__init__(L = None)
    # Regularising parameter = alpha
    self.regularisation_parameter = 1.
    
    # Iterations for FGP_TV
    self.iterations = max_iteration
    
    # Tolerance for FGP_TV
    self.tolerance = tolerance
    
    # Total variation correlation (isotropic=Default)
    self.isotropic = isotropic
    
    # correlation space or spacechannels
    self.correlation = correlation
    self.backend = backend        
    
    # Define orthogonal projection onto the convex set C
    self.lower = lower
    self.upper = upper
    self.tmp_proj_C = IndicatorBox(lower, upper).proximal
                    
    # Setup GradientOperator as None. This is to avoid domain argument in the __init__     

    self._gradient = None
    self._domain = None

    self.pptmp = None
    self.pptmp1 = None
    
    # Print stopping information (iterations and tolerance error) of FGP_TV  
    self.info = info

    # splitting Gradient
    self.split = split

    # warm-start
    self.warmstart  = warmstart
    if self.warmstart:
        self.hasstarted = False

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
        self.tmp_x.axpby(-self.regularisation_parameter*tau, 1.0, x, out=self.tmp_x)
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

#############################################################################
#               Reconstruction class                                        #
#############################################################################

class Reconstruct(object):

    def __init__(self, raw_args=None):

        parser = argparse.ArgumentParser(description="PET reconstruction")
        
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
        parser.add_argument("--no_warm_start", action='store_true', 
                            help="disables warm-start in CIL-TV")
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
        parser.add_argument("--precond", action='store_true',
                            help="Preconditioning flag") 
        
        # output parameters
        parser.add_argument("--nsave", help="Frequency to which save iterative recos", 
                            default=100, type=int)
        parser.add_argument("--nobj", help="Frequency to which compute objective", 
                            default=50, type=int)
        parser.add_argument("--nifti", action='store_true', help="Save reconstruction in nifti format")
        self.args = parser.parse_args()

    def create_output_folders(self):
        """Create output folders"""

        self.folder_log = '{}/logs'.format(os.getcwd())
        for folder in [self.args.folder_output, self.folder_log]:
            if not os.path.exists(folder):
                print('Create output or log folder')
                os.makedirs(folder)
        os.chdir(self.args.folder_output)

    def load_data_method(self):
        """Load data"""

        self.data_dict = load_data(self.args.folder_input, self.args.duetto_prefix)

    def define_corrective_factors(self):
        """Define additive and multiplicative correction factors"""

        self.data_shape = self.data_dict['prompts'].shape
        self.multfact_data = np.ones(self.data_shape)
        if self.args.acf == True:
            self.multfact_data *= self.data_dict['acf']
        if self.args.normf == True:
            self.multfact_data *= 1/self.data_dict['norm']
        if self.args.dtpucf == True:
            self.multfact_data *= 1/self.data_dict['dtPuc']
        self.addfact_data = np.zeros(self.data_shape)
        if self.args.randoms == True:
            self.addfact_data += self.data_dict['randoms']
        if self.args.scatter == True:
            self.addfact_data += self.data_dict['scatter']

    def set_up_projection_parameters(self):
        """Set-up projection parameters"""

        # Set-up acq data template using scanner name
        self.scanner_name = self.args.scanner
        self.span = self.args.span
        pet.AcquisitionData.set_storage_scheme('memory')
        self.raw_acq_data_template = pet.AcquisitionData(self.scanner_name,span=self.span)
        
        if self.args.fast:
            print("The '--fast' option is on, supercedes '--lor', '--nxny', '--numSegsToCombine' and '--numViewsToCombine' ")
            self.lor = 2
            self.nxny = int(np.ceil(self.raw_acq_data_template.create_uniform_image(0.0).shape[1]/2))
            self.num_segs_to_combine = 5
            self.num_views_to_combine = 4
        else:
            self.lor = self.args.lor
            self.nxny = self.args.nxny
            self.num_segs_to_combine = self.args.numSegsToCombine
            self.num_views_to_combine = self.args.numViewsToCombine

    def create_sirf_objects(self):
        """Create SIRF objects"""

        # Create and fill SIRF acq data objects
        sinogram_raw = self.raw_acq_data_template.clone()
        multfact_raw = self.raw_acq_data_template.clone()
        addfact_raw = self.raw_acq_data_template.clone() 
        sinogram_raw.fill(augment_dim(self.data_dict['prompts']))
        multfact_raw.fill(augment_dim(self.multfact_data))
        addfact_raw.fill(augment_dim(self.addfact_data))

        # Rebin
        self.sinogram, self.multfact, self.addfact = pre_process_sinogram(
            [sinogram_raw, multfact_raw, addfact_raw], 
            self.num_segs_to_combine,
            self.num_views_to_combine)

        # Initial image
        self.image_template = self.raw_acq_data_template.create_uniform_image(0.0, xy=self.nxny)

    def set_up_acq_model(self):
        '''Set-up acquisition model'''

        # number of subsets
        self.num_subsets = self.args.nsubsets 
        # set-up acquisition model
        self.acq_models = [pet.AcquisitionModelUsingRayTracingMatrix() for k in range(self.num_subsets)]
        # create masks
        im_one = self.image_template.clone()
        im_one.fill(1.)
        self.masks = []
        asm = pet.AcquisitionSensitivityModel(self.multfact)

        # Loop over physical subsets
        for k in range(self.num_subsets):
            # Set up
            self.acq_models[k].set_num_tangential_LORs(self.lor)
            self.acq_models[k].set_acquisition_sensitivity(asm)
            self.acq_models[k].set_up(self.sinogram, self.image_template)    
            self.acq_models[k].num_subsets = self.num_subsets
            self.acq_models[k].subset_num = k 

            # compute masks 
            mask = self.acq_models[k].direct(im_one)
            self.masks.append(mask)

    def set_up_FGK(self):
        """Set-up F, G and K"""

        data_fits = [KullbackLeibler(b=self.sinogram, eta=self.addfact, mask=mask.as_array(), use_numba=True) for mask in self.masks]
        r_alpha = self.args.reg_strength
        r_tolerance = 1e-7
        if self.args.reg == "FGP_TV" or self.args.reg == "FGP-TV":
            print("With the FGP_TV option, the gradient is defined as the finite difference operator (voxel-size not taken into account)")
            r_iters = 100
            r_iso = 1
            r_nonneg = 1
            device = 'gpu'
            self.G = FGP_TV(r_alpha, r_iters, r_tolerance,
                    r_iso, r_nonneg, device)
            if self.args.precond:
                raise ValueError("Precond option not compatible with FGP-TV regularizer")
                FGP_TV.proximal = precond_proximal
            # XXX redefines check_input which gives error
            FGP_TV.check_input = FGP_TV_check_input
        elif self.args.reg == "CIL-TV":
            print("With the CIL-TV option, the gradient is defined as the finite difference operator divided by the voxel-size in each direction")
            if self.args.no_warm_start:
                r_iters = 100
                self.G = r_alpha * TotalVariation(r_iters, r_tolerance, lower=0)
            else:
                TotalVariation.__init__ = CIL_TV__init__
                TotalVariation.proximal = CIL_TV_proximal
                r_iters = 5
                self.G = r_alpha * TotalVariation(r_iters, r_tolerance, lower=0, warmstart=True)

        elif self.args.reg == "None":
            self.G = IndicatorBox(lower=0)
        elif self.args.reg == "Explicit-TV":
            raise ValueError("Not implemented at the moment")
        else:
            raise ValueError("Unknown regularisation")

        self.F = BlockFunction(*data_fits)
        self.K = BlockOperator(*self.acq_models)

    def set_up_step_sizes_and_prob(self):
        """Set-up step-sizes and probabilities"""

        # uniform probabilities
        self.prob = [1/self.num_subsets] * self.num_subsets
        # primal-dual balancing parameter
        if not self.args.precond:
            # XXX mysterious axpby set-up
            self.use_axpby = True
            # compute the norm of each component
            self.postfix = "a{}_n{}_d{}".format(
                int(self.args.acf), 
                int(self.args.normf), 
                int(self.args.dtpucf))
            self.normKs = get_proj_norms(self.K, self.num_subsets, self.postfix, self.args.folder_output)
            # let spdhg do its default implementation
            self.sigmas = None
            self.tau = None
            self.gamma = self.args.pd_par
        else:
            # XXX mysterious axpby set-up
            self.use_axpby = False
            self.tau = 1/self.args.pd_par * get_tau(self.K, self.prob)
            self.sigmas = self.args.pd_par * get_sigmas(self.K)
            self.gamma = None
            self.normKs = None

    def set_up_SPDHG(self,log_file=None):
        """Set-up SPDHG"""

        # number of iterations
        self.num_iter = self.args.nepoch * self.num_subsets
        num_save = self.args.nsave * self.num_subsets
        num_obj = self.args.nobj * self.num_subsets

        self.spdhg = SPDHG(            
                    f=self.F, 
                    g=self.G, 
                    operator=self.K,
                    tau=self.tau,
                    sigma=self.sigmas,
                    gamma=self.gamma,
                    prob=self.prob,
                    use_axpby=self.use_axpby,
                    norms=self.normKs,
                    max_iteration=self.num_iter,         
                    update_objective_interval=num_obj,
                    log_file=log_file,
                    )

        # output_name = 'spdhg_reg_{}_alpha{}_nsub{}_precond{}_gamma{}_a{}_n{}_d{}_r{}_s{}'.format(
        #         args.reg, args.reg_strength, 
        #         num_subsets, args.precond, args.pd_par, int(args.acf), 
        #         int(args.normf), int(args.dtpucf), int(args.randoms), int(args.scatter)
        #         )
        if self.args.reg == 'CIL-TV':
            self.output_name = 'spdhg_reg_{}_warmstart{}_nsub{}_precond{}_a{}_n{}_d{}_r{}_s{}'.format(
                self.args.reg, int(not self.args.no_warm_start),
                self.num_subsets, int(self.args.precond), int(self.args.acf), 
                int(self.args.normf), int(self.args.dtpucf), int(self.args.randoms), int(self.args.scatter)
                )
        else:
            self.output_name = 'spdhg_reg_{}_nsub{}_precond{}_a{}_n{}_d{}_r{}_s{}'.format(
                self.args.reg, 
                self.num_subsets, int(self.args.precond), int(self.args.acf), 
                int(self.args.normf), int(self.args.dtpucf), int(self.args.randoms), int(self.args.scatter)
                )

        self.psave_callback = partial(
            save_callback, num_save, int(self.args.nifti), self.args.folder_output, self.output_name, self.num_iter)

    def run_SPDHG(self):
        '''Run SPDHG'''

        self.spdhg.run(self.num_iter, verbose=2, print_interval=1, callback=self.psave_callback)

    def save_metadata_and_objective(self):
        """Save metadata and obj values"""

        metadata_dict = {}
        # parameters of the mathematical reconstruction
        metadata_dict['regularizer'] = self.args.reg
        metadata_dict['regularization  strength'] = self.args.reg_strength
        metadata_dict['number of subsets'] = self.num_subsets
        metadata_dict['primal-dual parameter gamma'] = self.args.pd_par
        # parameters of the physical reconstruction
        metadata_dict['attenuation correction'] = self.args.acf
        metadata_dict['normalization'] = self.args.normf
        metadata_dict['dead-time and pile-up correction'] = self.args.dtpucf
        metadata_dict['randoms'] = self.args.randoms
        metadata_dict['scatter'] = self.args.scatter
        # save
        np.save('{}/{}_metadata'.format(self.args.folder_output,self.output_name), metadata_dict)
        np.save('{}/{}_objective'.format(self.args.folder_output, self.output_name), self.spdhg.objective)



    
        
if __name__ == "__main__":

    reconstruct = Reconstruct()

    # Create folders
    reconstruct.create_output_folders()

    # Create logger
    logfile = '{}/log_{}.log'.format(reconstruct.folder_log, datetime.now().strftime("%Y_%m_%d_%H:%M"))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(logfile, 'a'))

    # Load data
    logging.info('Load data')
    reconstruct.load_data_method()

    # Define additive and multiplicative correction factors
    logging.info('Define correction factors')
    reconstruct.define_corrective_factors()

    # Set up projection parameters
    logging.info('Set-up projection parameters')
    if reconstruct.args.fast:
        logging.warning("The '--fast' option is on, supercedes '--lor', '--nxny', '--numSegsToCombine' and '--numViewsToCombine' ")
    reconstruct.set_up_projection_parameters()

    # Create SIRF objects
    logging.info('Create SIRF objects')
    reconstruct.create_sirf_objects()

    # Set-up acquisition model
    logging.info('Set-up acquisition model')
    reconstruct.set_up_acq_model()

    # Set-up F, G and K
    logging.info('Set-up F, G and K')
    if reconstruct.args.reg == "FGP_TV" or reconstruct.args.reg == "FGP-TV":
        logging.warning("With the FGP_TV option, the gradient is defined as the finite difference operator (voxel-size not taken into account)")
    elif reconstruct.args.reg == "CIL-TV":
        logging.warning("With the CIL-TV option, the gradient is defined as the finite difference operator divided by the voxel-size in each direction")
    reconstruct.set_up_FGK()

    # Set-up step-sizes and probabilities
    logging.info('Set-up step-sizes and probabilities')
    reconstruct.set_up_step_sizes_and_prob()

    # Set-up SPDHG
    logging.info('Set-up SPDHG')
    reconstruct.set_up_SPDHG(log_file=logfile)

    # Run SPDHG
    logging.info('Run SPDHG')
    reconstruct.run_SPDHG()

    # Save metadata and obj values
    logging.info('save objective values')
    reconstruct.save_metadata_and_objective()







        
