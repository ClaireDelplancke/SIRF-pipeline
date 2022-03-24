"""
Author: Claire Delplancke, University of Bath, PET++ project
2022
https://clairedelplancke.github.io
https://petpp.github.io
"""

python Reconstruct.py --folder_input 'folder absolute path' \
				--folder_output 'folder absolute path'  \
				--nepoch 200 \
				--nsave 100 \
				--nobj 10 \
				--reg 'FGP-TV' \
                --nsubsets 1 \
				--fast