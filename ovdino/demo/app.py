# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import sys

import gradio as gr

sys.path.insert(0, "./")  # noqa
from demo.predictors import OVDINODemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.utils.logger import setup_logger
from detrex.data.datasets import clean_words_or_phrase

# constants
WINDOW_NAME = "COCO detections"


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="detrex demo for visualizing customized inputs"
    )
    parser.add_argument(
        "--config-file",
        default="projects/dino/configs/dino_r50_4scale_12ep.py",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--min_size_test",
        type=int,
        default=800,
        help="Size of the smallest side of the image during testing. Set to zero to disable resize in testing.",
    )
    parser.add_argument(
        "--max_size_test",
        type=float,
        default=1333,
        help="Maximum size of the side of the image during testing.",
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="RGB",
        help="The format of the loading images.",
    )
    parser.add_argument(
        "--metadata_dataset",
        type=str,
        default="coco_2017_val",
        help="The metadata infomation to be used. Default to COCO val metadata.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    demo = OVDINODemo(
        model=model,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        img_format=args.img_format,
        metadata_dataset=args.metadata_dataset,
    )

    def gradio_predict(image, text):
        category_names = text.split(" ")
        category_names = [
            clean_words_or_phrase(cat_name) for cat_name in category_names
        ]
        _, visualized_output = demo.run_on_image(
            image, category_names, args.confidence_threshold
        )

        return visualized_output.get_image()[:, :, ::-1]

    image = gr.inputs.Image(shape=(512, 512))

    title_markdown = """
    # 🦖 OV-DINO: Unifiedcabulary Detection with Language-Aware Selective Fusion
    """

    messeage_markdown = """
    This is a demo of **OV-DINO: Unifiedcabulary Detection with Language-Aware Selective Fusion**. 
    
    The model takes image and text as input, and outputs the detection results.
    The text input is a list of categorie names, the input category_names are separated by spaces, and the words of single class are connected by underline (_).
    
    Paper: https://arxiv.org/abs/2407.07844
    
    Code: https://github.com/wanghao9610/OV-DINO
    """

    gr.Interface(
        description=title_markdown,
        fn=gradio_predict,
        inputs=["image", "text"],
        outputs=[
            gr.outputs.Image(
                type="pil",
                # label="grounding results"
            ),
        ],
        examples=[
            ["./demo/imgs/000000001584.jpg", "bus person license_plate"],
            ["./demo/imgs/000000004495.jpg", "person tv couch chair"],
            [
                "./demo/imgs/000000009483.jpg",
                "person keyboard table computer_monitor computer_mouse",
            ],
            [
                "./demo/imgs/000000017714.jpg",
                "cup spoon pizza knife fork bowl",
            ],
        ],
        article=messeage_markdown,
    ).launch()
