DATA_ROOT=/ssd_scratch/cvit/raghava.modhugu
TRAIN_SET=$DATA_ROOT/sequences
python train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name resnet50_pose_256_argoverse \
--pretrained-disp=/home/raghava.modhugu/SC-SfMLearner-Release/Pretrained_models/resnet50/dispnet_model_best.pth.tar \
--pretrained-pose=/home/raghava.modhugu/SC-SfMLearner-Release/Pretrained_models/resnet50/exp_pose_model_best.pth.tar