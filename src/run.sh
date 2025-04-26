#!/bin/bash
# get clone detection result for the project (i.e., glide) based on nicad
python './0_clone_detection.py' glide nicad
# purify the clone result of the project (i.e., glide)
#python '1_purify_clone_detection_paratypes.py' glide
# get clone detection result for the project (i.e., glide) based on nicad
#python '2_build_genealogy.py' glide nicad
# get metrics of the project (i.e., glide)
#python '3_extract_metrics.py' glide
# combine all the metrics together of the project (i.e., glide)
#python '4_combine_metrics.py' glide
# extract the quality labels of the project (i.e., glide)
#python '5_extract_quality_labels.py' glide
# combine metrics with label of the project (i.e., glide)
#python '6_combine_metrics_label.py' glide
# conduct model fine-tuning
#python '7_model_training_parallel.py'