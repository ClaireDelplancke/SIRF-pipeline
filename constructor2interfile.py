# %% Import libraries

import os
import sirf.STIR as pet
import argparse
import runpy
runpy.run_path('Reconstruct.py')
from Reconstruct import Reconstruct

# %%
def main(raw_args=None):

    '''
    Read constructor sinograms in .mat format, process and save to Interfile format

    Usage:
    python constructor2interfile.py --folder_input path1 \
		                        --folder_output path2  \
                                --scanner 'Discovery MI4'
    '''

    #############################################################################
    #               Arguments                                                   #
    #############################################################################

    parser = argparse.ArgumentParser(description="Convert sinograms from constructor format to Interfile")
    
    # input / output
    parser.add_argument("--scanner", type=str, 
                        default='GE Discovery 690',
                        choices=['GE Signa PET/MR', 'Discovery MI4', 'GE Discovery 690'])
    parser.add_argument("--span", type=int, 
                        default=2, help='GE scanners need 2')
    parser.add_argument("--folder_input", type=str)
    parser.add_argument("--duetto_prefix", default='f1b1', type=str)
    parser.add_argument("--folder_output", type=str)
    parser.add_argument("--nxny", help="Number of voxels", 
                        type=int)
    parser.add_argument("--numSegsToCombine", 
                        help = "Rebin all sinograms, with a given number of segments to combine.",
                        type=int, default = 1)
    parser.add_argument("--numViewsToCombine",
                        help = "Rebin all sinograms, with a given number of view to combine.",
                        type=int, default = 1)
    parser.add_argument('--fast', action='store_true',
                        help="Rebins data")
    parser.add_argument('--faast', action='store_true',
                        help="Rebins data and halves default reconstructed image dimensions")
    args = parser.parse_args(raw_args)

    #############################################################################
    #               Output folder                                               #
    #############################################################################

    if not os.path.exists(args.folder_output):
        os.makedirs(args.folder_output)
    os.chdir(args.folder_output)

    #############################################################################
    #               Process data                                                #
    #############################################################################

    recon_tool = Reconstruct(raw_args)
    recon_tool.load_data_method()
    recon_tool.define_corrective_factors()
    recon_tool.set_up_projection_parameters()
    recon_tool.create_sirf_objects()

    #############################################################################
    #               Save in interfile format                                    #
    #############################################################################

    recon_tool.sinogram.write('{}/acquisition_data.hs'.format(args.folder_output))
    recon_tool.multfact.write('{}/multiplicative_factors.hs'.format(args.folder_output))
    recon_tool.addfact.write('{}/additive_factors.hs'.format(args.folder_output))
    recon_tool.image_template.write('{}/image_template.hv'.format(args.folder_output))

if __name__ == "__main__":
    main()



