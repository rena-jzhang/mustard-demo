import sys

import numpy as np
import pandas as pd

# sys.path.append("/results/twoertwe/emro")

from dataloader import get_partitions


for dataset, labels in (
    # regression
    ("mosi", ("sentiment",)),
    ("mosei", ("sentiment", "happiness")),
    ("sewa", ("arousal", "valence")),
    ("recola", ("arousal", "valence")),
    # ("iemocap", ("arousal", "valence")),
    ("umeme", ("arousal", "valence")),
    # # classification (4 classes)
    # ("tpot", ("constructs",)),
    # # classification (5 classes)
    # ("vreed", ("av",)),
):
    for label in labels:
        # get 5-fold
        for fold, partition in get_partitions(
            f"{dataset}/{label}", batch_size=-1
        ).items():
            # training, validation, test sets
            for name, data in partition.items():
                assert len(data.iterator) == 1
                features = pd.DataFrame(
                    data.iterator[0].pop("x")[0],
                    columns=data.properties["x_names"],
                )
                labels_metadata = pd.DataFrame(
                    {key: value[0][:, 0] for key, value in data.iterator[0].items()}
                )
                pd.concat([labels_metadata, features], axis=1).to_csv(
                    f"{dataset}_{label}_{fold}_{name}.csv", index=False
                )
