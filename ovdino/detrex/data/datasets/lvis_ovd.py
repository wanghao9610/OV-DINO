# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detrex.utils import rank0_print
from fvcore.common.timer import Timer
from tqdm import tqdm

from .utils import clean_words_or_phrase

"""
This file contains functions to parse LVIS-format annotations into dicts in the
"Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_lvis_ovd_json", "register_lvis_ovd_instances"]


def load_lvis_ovd_json(
    json_file,
    image_root,
    dataset_name=None,
    extra_annotation_keys=None,
    num_sampled_classes=-1,
    chunk_categories=True,
):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    if dataset_name is not None:
        meta = get_lvis_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info(
        "Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file)
    )

    if extra_annotation_keys:
        logger.info(
            "The following extra annotation keys will be loaded: {} ".format(
                extra_annotation_keys
            )
        )
    else:
        extra_annotation_keys = []

    # for lvis, we split the category to chunks following GLIP.
    def split_categories_to_chunks(cat_names, chunk_size):
        cat_names_chunks = []
        cnt = 0
        for i in range(0, len(cat_names), chunk_size):
            cur_chunk = cat_names[i : i + chunk_size]
            cat_names_chunks.append(cur_chunk)
            cnt += len(cur_chunk)

        assert cnt == len(cat_names)
        return cat_names_chunks

    cat_names = meta.get("thing_classes")
    if chunk_categories:
        cat_names_chunks = split_categories_to_chunks(cat_names, num_sampled_classes)

    dataset_dicts = []

    for img_dict, anno_dict_list in tqdm(imgs_anns, desc=f"Loading {dataset_name}"):
        # for (img_dict, anno_dict_list) in imgs_anns:
        # category_start_index: the shiftd category index for split chunk.
        if chunk_categories:
            category_start_index = 0
            for i, cat_names_chunk in enumerate(cat_names_chunks):
                record = {}
                record["file_name"] = os.path.join(image_root, img_dict["file_name"])
                record["height"] = img_dict["height"]
                record["width"] = img_dict["width"]
                record["not_exhaustive_category_ids"] = img_dict.get(
                    "not_exhaustive_category_ids", []
                )
                record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
                image_id = record["image_id"] = img_dict["id"]

                objs = []
                for anno in anno_dict_list:
                    # Check that the image_id in this annotation is the same as
                    # the image_id we're looking at.
                    # This fails only when the data parsing logic or the annotation file is buggy.
                    assert anno["image_id"] == image_id
                    obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
                    # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
                    # variable will have a field with COCO-specific category mapping.
                    if (
                        dataset_name is not None
                        and "thing_dataset_id_to_contiguous_id" in meta
                    ):
                        obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                            anno["category_id"]
                        ]
                    else:
                        obj["category_id"] = (
                            anno["category_id"] - 1
                        )  # Convert 1-indexed to 0-indexed
                    segm = anno["segmentation"]  # list[list[float]]
                    # filter out invalid polygons (< 3 points)
                    valid_segm = [
                        poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    assert len(segm) == len(
                        valid_segm
                    ), "Annotation contains an invalid polygon with < 3 points"
                    assert len(segm) > 0
                    obj["segmentation"] = segm
                    for extra_ann_key in extra_annotation_keys:
                        obj[extra_ann_key] = anno[extra_ann_key]
                    objs.append(obj)
                record["annotations"] = objs
                record["data_type"] = "detection"
                cat_names = [
                    clean_words_or_phrase(cat_name) for cat_name in cat_names_chunk
                ]

                record["category_names"] = cat_names
                record["category_start_index"] = category_start_index
                record["chunk_index"] = i
                category_start_index += len(cat_names)
                dataset_dicts.append(record)
        else:
            record = {}
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["not_exhaustive_category_ids"] = img_dict.get(
                "not_exhaustive_category_ids", []
            )
            record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.
                assert anno["image_id"] == image_id
                obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
                # LVIS data loader can be used to load COCO dataset categories. In this case `meta`
                # variable will have a field with COCO-specific category mapping.
                if (
                    dataset_name is not None
                    and "thing_dataset_id_to_contiguous_id" in meta
                ):
                    obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                        anno["category_id"]
                    ]
                else:
                    obj["category_id"] = (
                        anno["category_id"] - 1
                    )  # Convert 1-indexed to 0-indexed
                segm = anno["segmentation"]  # list[list[float]]
                # filter out invalid polygons (< 3 points)
                valid_segm = [
                    poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
                ]
                assert len(segm) == len(
                    valid_segm
                ), "Annotation contains an invalid polygon with < 3 points"
                assert len(segm) > 0
                obj["segmentation"] = segm
                for extra_ann_key in extra_annotation_keys:
                    obj[extra_ann_key] = anno[extra_ann_key]
                objs.append(obj)
            record["annotations"] = objs

            cat_names = [clean_words_or_phrase(cat_name) for cat_name in cat_names]
            record["category_names"] = cat_names

            dataset_dicts.append(record)

    rank0_print(
        f"Loaded {len(dataset_dicts)} data points from {dataset_name}\nSample: {cat_names}"
    )

    return dataset_dicts


def register_lvis_ovd_instances(
    name,
    metadata,
    json_file,
    image_root,
    num_sampled_classes=-1,
):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(
        name,
        lambda: load_lvis_ovd_json(
            json_file,
            image_root,
            name,
            num_sampled_classes=num_sampled_classes,
        ),
    )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )


if __name__ == "__main__":
    """
    Test the LVIS json dataset loader.

    Usage:
        python -m detectron2.data.datasets.lvis \
            path/to/json path/to/image_root dataset_name vis_limit
    """
    import sys

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import numpy as np
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    from PIL import Image

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_lvis_ovd_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "lvis-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts[: int(sys.argv[4])]:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
