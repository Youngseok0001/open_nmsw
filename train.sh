# nmsw
python train-test.py \
    model=NSWNet3D \
    dataset_name=Word \
    gpu_id=0 \
    ++model.local_backbone_name=FasterUNet \
    ++model.num_train_topk_patches=4\
    ++model.num_train_random_patches=1\

# local
python train-test.py \
    model=LocalSeg3D \
    dataset_name=Word \
    gpu_id=0 \
    ++model.local_backbone_name=FasterUNet \
    ++model.num_patches=5

# global
python train-test.py \
    model=GlobalSeg3D \
    dataset_name=Word \
    gpu_id=0 \
    ++model.global_backbone_name=FasterUNet    