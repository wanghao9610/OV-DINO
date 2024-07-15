import os

from detectron2.data.datasets.lvis import get_lvis_instances_meta

from .lvis_ovd import register_lvis_ovd_instances

_PREDEFINED_SPLITS = {
    # image_root, json_file, num_sampled_classes
    # NOTE: omite unipro to save memory.
    "lvis_v1_train_ovd": (
        "lvis/",
        "lvis/annotations/lvis_v1_train.json",
        40,
    ),
    "lvis_v1_val_ovd": (
        "lvis/",
        "lvis/annotations/lvis_v1_val_inserted_image_name.json",
        40,
    ),
    "lvis_v1_minival_ovd": (
        "lvis/",
        "lvis/annotations/lvis_v1_minival_inserted_image_name.json",
        40,
    ),
}


def register_all_lvis_instances(root):
    for key, (
        image_root,
        json_file,
        num_sampled_classes,
    ) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_lvis_ovd_instances(
            key,
            get_lvis_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            num_sampled_classes,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvis_instances(_root)
