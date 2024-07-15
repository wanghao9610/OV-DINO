import itertools

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detrex.data import DetrDatasetMapper
from detrex.data.datasets import register_coco_ovd_instances
from omegaconf import OmegaConf

dataloader = OmegaConf.create()

# if you follow the coco format, you can use the following code.
# if you want to define it by yourself, you can change it on ovdino/detrex/data/datasets/custom_ovd.py.
# register_coco_ovd_instances(
#     "custom_train_ovd_unipro",  # dataset_name
#     {},  # custom_data_info
#     "/path/to/train.json",  # annotations_jsonfile
#     "/path/to/train/images",  # image_root
#     80,  # number_of_classes, default: 80
#     "full",  # template, default: full
# )
# register_coco_ovd_instances(
#     "custom_val_ovd_unipro",
#     {},
#     "/path/to/val.json",
#     "/path/to/val/images",
#     80,
#     "full",
# )
# register_coco_ovd_instances(
#     "custom_test_ovd",
#     {},
#     "/path/to/test.json",
#     "/path/to/test/images",
#     80,
#     "identity",
# )

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="custom_train_ovd_unipro"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(
                    480,
                    512,
                    544,
                    576,
                    608,
                    640,
                    672,
                    704,
                    736,
                    768,
                    800,
                ),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="custom_val_ovd_unipro", filter_empty=False
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
