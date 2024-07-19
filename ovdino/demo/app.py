# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import os
import sys

import gradio as gr
import numpy as np

sys.path.insert(0, "./")  # noqa
from demo.predictors import OVDINODemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from detrex.data.datasets import clean_words_or_phrase

this_dir = os.path.dirname(os.path.abspath(__file__))


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

    def gradio_predict(image, text, score_thr):
        category_names = text.split(", ")
        category_names = [
            clean_words_or_phrase(cat_name) for cat_name in category_names
        ]
        image = np.array(image)
        _, visualized_output = demo.run_on_image(image, category_names, score_thr)

        return visualized_output.get_image()[:, :, ::-1]

    examples = [
        [
            os.path.join(this_dir, "./imgs/000000001584.jpg"),
            "person, bus, bicycle",
        ],
        [
            os.path.join(this_dir, "./imgs/000000004495.jpg"),
            "person, tv, couch, whiteboard",
        ],
        [
            os.path.join(this_dir, "./imgs/000000009483.jpg"),
            "person, keyboard, table, computer monitor, computer mouse",
        ],
        [
            os.path.join(this_dir, "./imgs/000000017714.jpg"),
            "table, cup, spoon, pizza, knife, fork, dish",
        ],
    ]

    coco_categories = ", ".join(
        MetadataCatalog.get(args.metadata_dataset).thing_classes
    )

    with gr.Blocks(title="OV-DINO") as app:
        with gr.Row():
            gr.Markdown(
                """
                <div align="center">
                <h1>🦖OV-DINO </h1>
                <h2>Unified Open-Vocabulary Detection with Language-Aware Selective Fusion</h2>
                
                [[`Paper`](https://arxiv.org/abs/2407.07844)] [[`Code`](https://github.com/wanghao9610/OV-DINO)]
                </div>
                """
            )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    image = gr.Image(type="pil", label="input image")
                input_text = gr.Textbox(
                    lines=7,
                    label="Enter the classes to be detected, separated by comma",
                    value=coco_categories,
                    elem_id="textbox",
                )
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")
                score_thr = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    interactive=True,
                    label="Score Threshold",
                )
            with gr.Column(scale=7):
                output_image = gr.Image(type="pil", label="output image")

        gr.Examples(examples=examples, inputs=[image, input_text], examples_per_page=10)
        submit.click(
            fn=gradio_predict,
            inputs=[image, input_text, score_thr],
            outputs=[output_image],
        )
        clear.click(
            lambda: [None, coco_categories, 0.5],
            inputs=None,
            outputs=[image, input_text, score_thr],
        )
        app.launch()
