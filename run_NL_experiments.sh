#!/bin/bash

# Use "rm test*db*" if you want to try another model, creating a cache.

# python experiment_set_of_VSE.py --IPOT \
# 	--img_type global \
# 	--text_type global

# rm test*db*
# python experiment_set_of_VSE_complete_sentence.py --IPOT \
# 	--model_name "RN50x64" \
#  	--img_type partial \
#  	--text_type partial 

# python experiment_set_of_VSE_complete_sentence.py --IPOT \
#  	--img_type global \
#  	--text_type partial 

# python experiment_set_of_VSE_complete_sentence.py --IPOT \
#  	--img_type partial \
#  	--text_type global

# python experiment_set_of_VSE_complete_sentence.py --IPOT \
#  	--img_type hybrid \
#  	--text_type partial

# python experiment_set_of_VSE_complete_sentence.py --IPOT \
#  	--img_type hybrid \
#  	--text_type hybrid 

# python experiment_set_of_VSE.py --IPOT \
# 	--img_type global \
# 	--text_type partial \
# 	--save_dir img_global_text_partial

# python experiment_set_of_VSE.py --IPOT \
# 	--img_type partial \
# 	--text_type global \
# 	--save_dir text_global_img_partial

#python experiment_set_of_VSE.py --IPOT \
#	--img_type hybrid \
#	--text_type hybrid

# ------------------------------

# python experiment_set_of_VSE.py --IPOT \
# 	--img_type partial \
# 	--text_type partial \
# 	--dataset_type train
