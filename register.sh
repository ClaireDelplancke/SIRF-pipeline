# register a GE reference image (MR or CT) to a PET reconstruction
# first argument is the path to folder containing DICOMs of the MR or CT
# second argument is the path to the .nii PET reconstruction
# this will create two .nii files in the DICOM folder: reoriented.nii and registered.nii

if [ $# -ne 2 ]; then
  echo "Needs two arguments"
  exit 1
fi

dicom_folder=$1
pet_reco=$2

# reorient GE ref image
python reorient_GE2SIRF.py --dicom_path $dicom_folder

# register
reg_aladin -ref $pet_reco -flo ${dicom_folder}/reoriented.nii -res ${dicom_folder}/registered.nii -rigOnly -speeeeed