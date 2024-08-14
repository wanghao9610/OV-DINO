import os

from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

from .coco_ovd import register_coco_ovd_instances

_PREDEFINED_SPLITS = {
    # image_root, json_file, num_sampled_classes, template
    "coco_2017_train_ovd_unipro": (
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
        80,
        "full",
    ),
    "coco_2017_val_ovd_unipro": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        80,
        "full",
    ),
    "coco_2017_train_ovd": (
        "coco/train2017",
        "coco/annotations/instances_train2017.json",
        80,
        "identity",
    ),
    "coco_2017_val_ovd": (
        "coco/val2017",
        "coco/annotations/instances_val2017.json",
        80,
        "identity",
    ),
}


def register_all_coco_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
        template,
    ) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datas`.
        register_coco_ovd_instances(
            key,
            _get_coco_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            num_sampled_classes,
            template=template,
            test_mode=True if "val" in key else False,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_instances(_root)
