import os

from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta

from .custom_ovd import register_coco_ovd_instances

_PREDEFINED_SPLITS = {
    # image_root, json_file, num_sampled_classes, template
    "custom_train_ovd_unipro": (
        "custom/train",
        "custom/annotations/train.json",
        80,
        "full",
    ),
    "custom_val_ovd_unipro": (
        "custom/val",
        "custom/annotations/val.json",
        80,
        "full",
    ),
    "custom_test_ovd": (
        "custom/test",
        "custom/annotations/test.json",
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
        # Assume pre-defined datasets live in `./datasets`.
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
