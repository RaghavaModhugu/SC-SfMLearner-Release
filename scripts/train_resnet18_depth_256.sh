#DATA_ROOT=/media/bjw/Disk
TRAIN_SET=/ssd_scratch/cvit/raghava.modhugu/sequences/
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b12 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name resnet18_depth_256 \
--folder-type sequence \
--pretrained-disp=/home/raghava.modhugu/SC-SfMLearner-Release/Pretrained_models/dispnet_model_best.pth.tar \
--pretrained-pose=/home/raghava.modhugu/SC-SfMLearner-Release/Pretrained_models/exp_pose_model_best.pth.tar
