# nmsw
python train-test.py \
    model=NSWNet3D \
    dataset_name=Word \
    gpu_id=0 \
    ++model.num_infernce_patches=[5/30/300]\
    ckpt_path=[your/checkpoint/path]\

# baseline:s_w
python train-test.py \
    model=GlobalLocalSeg3D \
    dataset_name=Word \
    gpu_id=0 \
    ++model.sampling_stretegy.sampling_stretegy=s_w\
    ++model.local_backbone_name=FasterUNet\
    ++model.global_backbone_name=FasterUNet\
    ++model.local_ckpt_path=[your/checkpoint/path]\
    ++model.global_ckpt_path=[your/checkpoint/path]
    
# baseline:random_fg
python train-test.py \
    model=GlobalLocalSeg3D \
    dataset_name=Word \
    gpu_id=0 \
    ++model.num_infernce_patches=[5/30/300]\
    ++model.sampling_stretegy.sampling_stretegy=random_fg\
    ++model.local_backbone_name=FasterUNet\
    ++model.global_backbone_name=FasterUNet\
    ++model.local_ckpt_path=[your/checkpoint/path]\
    ++model.global_ckpt_path=[your/checkpoint/path]

# baseline:zoom_out
python train-test.py \
    model=GlobalLocalSeg3D\
    dataset_name=Word\
    gpu_id=0\
    ++model.sampling_stretegy.sampling_stretegy=zoom_in\
    ++model.local_backbone_name=FasterUNet\
    ++model.global_backbone_name=FasterUNet\
    ++model.local_ckpt_path=[your/checkpoint/path]\
    ++model.global_ckpt_path=[your/checkpoint/path]
            