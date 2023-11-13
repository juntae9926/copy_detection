# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import random
from typing import Callable, Dict, Optional
from torchvision.datasets.folder import default_loader
from PIL import Image, ImageFilter
from torchvision import transforms

from sscd.datasets.image_folder import get_image_paths
from sscd.datasets.isc.descriptor_matching import (
    knn_match_and_make_predictions,
    match_and_make_predictions,
)
from sscd.datasets.isc.io import read_ground_truth
from sscd.datasets.isc.metrics import evaluate, Metrics

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class GaussianBlur():
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def pretrain_transform():
    transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.RandomApply([GaussianBlur([.1, 5.])], p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.8), 
                                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform

def val_transform():
    transform = transforms.Compose([transforms.Resize((288, 288)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform

class DISCTrainDataset:
    """A data module describing datasets used during training."""

    def __init__(
        self,
        data_dir,
    ):
        self.data_dir = data_dir
        self.files = get_image_paths(self.data_dir)
        self.transform = pretrain_transform()
        self.loader = default_loader

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        img = self.loader(self.files[idx])
        img_q, img_k = self.transform(img), self.transform(img)
        # record = {"input": img, "instance_id": idx}
        return img_q, img_k
    
    
class DISCEvalDataset:
    """DISC2021 evaluation dataset."""

    SPLIT_REF = 0
    SPLIT_QUERY = 1
    SPLIT_TRAIN = 2

    def __init__(
        self,
        path: str,
    ):

        query_path = os.path.join(path, "final_queries")
        ref_path = os.path.join(path, "references") # references
        gt_path = os.path.join(path, "gt_1k.csv")
        self.files, self.metadata = self.read_files(ref_path, self.SPLIT_REF)
        query_files, query_metadata = self.read_files(query_path, self.SPLIT_QUERY)
        self.files.extend(query_files)
        self.metadata.extend(query_metadata)
        self.gt = read_ground_truth(gt_path)
        self.transform = val_transform()

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        img = default_loader(filename)
        if self.transform:
            img = self.transform(img)
        sample = {"input": img, "instance_id": idx}
        sample.update(self.metadata[idx])
        return sample

    def __len__(self):
        return len(self.files)

    @classmethod
    def read_files(cls, path, split):
        files = get_image_paths(path)
        names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        metadata = [
            dict(name=name, split=split, image_num=int(name[1:]), target=-1)
            for name in names
        ]
        return files, metadata

    def retrieval_eval(
        self, embedding_array, targets, split, **kwargs
    ) -> Dict[str, float]:
        query_mask = split == self.SPLIT_QUERY
        ref_mask = split == self.SPLIT_REF
        query_ids = targets[query_mask]
        query_embeddings = embedding_array[query_mask, :]
        ref_ids = targets[ref_mask]
        ref_embeddings = embedding_array[ref_mask, :]
        return self.retrieval_eval_splits(
            query_ids, query_embeddings, ref_ids, ref_embeddings, **kwargs
        )

    def retrieval_eval_splits(
        self,
        query_ids,
        query_embeddings,
        ref_ids,
        ref_embeddings,
        use_gpu=False,
        k=10,
        global_candidates=False,
        **kwargs
    ) -> Dict[str, float]:
        query_names = ["Q%05d" % i for i in query_ids]
        ref_names = ["R%06d" % i for i in ref_ids]
        if global_candidates:
            predictions = match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                num_results=k * len(query_names),
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        else:
            predictions = knn_match_and_make_predictions(
                query_embeddings,
                query_names,
                ref_embeddings,
                ref_names,
                k=k,
                ngpu=-1 if use_gpu else 0,
                **kwargs,
            )
        results: Metrics = evaluate(self.gt, predictions)
        return {
            "uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0,
        }
