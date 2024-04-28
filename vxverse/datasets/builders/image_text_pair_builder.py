import os
import logging
import warnings

from vxverse.common.registry import registry
from vxverse.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vxverse.datasets.datasets.gqa_datasets import GQADataset



@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/balanced_val.yaml",
    }

    








