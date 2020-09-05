INPUT_DIR=/media/bjw/Disk/Dataset/kitti_odometry/sequences/09/image_2
OUTPUT_DIR=results/
DISP_NET=/Users/raghavamodhugu/Desktop/IIITH/Trajectory_prediction/argoverse_tracks_generation/SC-SfMLearner-Release/Pretrained_models/resnet18/dispnet_model_best.pth.tar
ArgoPath=/Users/raghavamodhugu/Downloads/Argoverse_samples/argoverse-tracking/sample
log=74750688-7475-7475-7475-474752397312

python3 run_inference_argo.py --pretrained $DISP_NET --resnet-layers 18 --output-dir $OUTPUT_DIR --output-depth --argoverse_data_path $ArgoPath --argoverse_log $log
