#!/usr/bin/env python3
import sys
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from python_tools import caching
from python_tools.features import (
    aggregate_to_intervals,
    apply_masking,
    suggest_statistics,
    synchronize_sequences,
)
from python_tools.generic import map_parallel
from python_tools.ml.data_loader import DataLoader
from python_tools.ml.markers import get_markers
from python_tools.ml.pytorch_tools import dict_to_batched_data
from python_tools.ml.split import stratified_splits
from sklearn.decomposition import PCA


def align(
    name_intervals: tuple[str, pd.DataFrame],
    folder: Path,
    skip_opensmile_stats: bool = False,
) -> pd.DataFrame:
    name, intervals = name_intervals
    index = intervals.index
    intervals = intervals.loc[:, ["meta_begin", "meta_end"]].values + [-0.5, 0.5]

    # load raw features
    data = {}
    for key in ("opensmile_eGeMAPSv02", "opensmile_vad_opensource", "openface"):
        tmp = caching.read_hdfs(folder / key / name)
        if "df" in tmp:
            data[key] = tmp["df"]
        else:
            for subfeature in tmp:
                data[f"{key}_{subfeature}"] = tmp[subfeature]
    if skip_opensmile_stats:
        data.pop("opensmile_eGeMAPSv02_csvoutput")
    data = synchronize_sequences(data, intersect=False)

    # simple stats and some markers
    data = apply_masking(data)
    stats = suggest_statistics(data.columns.to_list())
    feature = (
        aggregate_to_intervals(data, intervals, stats)
        .drop(columns=["begin", "end"])
        .set_index(index)
    )
    tmp = []
    for begin, end in intervals:
        interval_data = data.loc[begin:end]
        if interval_data.empty:
            tmp.append(None)
        else:
            tmp.append(get_markers(interval_data))
    nan = {key: float("NaN") for key in next(x for x in tmp if x is not None)}
    tmp = [nan if x is None else x for x in tmp]
    tmp = pd.DataFrame(tmp, index=index)
    result = (
        pd.concat(
            [feature, tmp.loc[:, tmp.columns.difference(feature.columns)]],
            axis=1,
        )
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    return result.rename(
        columns={x: f"vision_{x}" for x in result.columns if x.startswith("openface")}
        | {x: f"acoustic_{x}" for x in result.columns if x.startswith("opensmile")}
        | {"duchenne_smile_ratio": "vision_duchenne_smile_ratio"}
    )


class MOSI:
    def __init__(self) -> None:
        self.folder = Path(
            f"/projects/dataset_processed/{self.__class__.__name__.replace('_', '-')}/twoertwein/"
        )

        self.minilm = pd.read_hdf(self.folder / "all_minilm_l12_v2.hdf").rename(
            columns={
                "begin": "meta_begin",
                "end": "meta_end",
                "clip": "meta_clip",
                "meta_id": "name",
            }
        )
        self.minilm["meta_id"] = pd.Categorical(self.minilm["name"]).codes
        if "file" in self.minilm.columns:  # for IEMOCAP and UMEME
            self.minilm["file"] = self.minilm["file"].apply(
                lambda x: x.split(".", 1)[0]
            )
            self.minilm["name"] = self.minilm["file"]
        if "meta_begin" not in self.minilm:
            self.minilm["meta_begin"] = 0.0
            self.minilm["meta_end"] = 60.0
        self.minilm = self.minilm.drop(
            columns=self.minilm.columns[self.minilm.dtypes == object].difference(
                ["name"]
            )
        ).copy()
        self.minilm = self.minilm.rename(
            columns={
                x: f"language_{x}"
                for x in self.minilm.columns
                if x.startswith(("all_", "liwc_"))
            }
        )
        self.features = pd.concat([self.minilm, self.align_features()], axis=1).astype(
            np.float32, errors="ignore"
        )
        self.features["meta_id"] = self.features["meta_id"].astype(int)

    def get_official_validation_test_set(self) -> tuple[list[str], list[str]]:
        if type(self) != MOSI:
            return [], []

        sys.path.append(
            str(
                Path(
                    "~/git/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI/"
                ).expanduser()
            )
        )
        import cmu_mosi_std_folds

        return (
            cmu_mosi_std_folds.standard_valid_fold,
            cmu_mosi_std_folds.standard_test_fold,
        )

    def align_features(self) -> pd.DataFrame:
        folder = Path(__file__).parent.resolve()
        cached_file = folder / "cache" / f"{self.__class__.__name__}.hdf"
        cached = caching.read_hdfs(cached_file)
        if cached:
            return cached["df"]

        features = []
        names = self.minilm["name"].unique()
        intervals = []
        for name in names:
            intervals.append(
                self.minilm.loc[self.minilm["name"] == name, ["meta_begin", "meta_end"]]
            )
        features = pd.concat(
            map_parallel(
                partial(
                    align,
                    folder=self.folder,
                    skip_opensmile_stats=isinstance(self, UMEME),
                ),
                zip(names, intervals),
                workers=7,
            ),
            axis=0,
        )
        features = features.fillna(features.mean())
        caching.write_hdfs(cached_file, {"df": features})
        return features

    def get_subjects(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
    ) -> list[str]:
        validation, test = self.get_official_validation_test_set()
        # always use official test split
        if name == "test" and test:
            return test

        # use official validation set during for first fold
        if validation and fold == 0:
            if name == "validation":
                return validation
            assert name == "training"
            return np.setdiff1d(
                self.features["name"].unique(), validation + test
            ).tolist()

        # create a stratified split for the remaining cases
        average = self.features.groupby("name")[dimension].mean()
        average = average.loc[average.index.difference(test)]

        if validation:
            groups = np.ones(round(average.shape[0] / len(validation)))
        else:
            groups = np.ones(5)
        indices = stratified_splits(average.values, groups)
        test = indices == fold
        validation = indices == (fold + 1) % 5
        training = ~(test | validation)
        match name:
            case "training":
                return average.index[training].tolist()
            case "validation":
                return average.index[validation].tolist()
            case "test":
                return average.index[test].tolist()

    def get_loader(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
        batch_size: int,
    ) -> DataLoader:
        subjects = self.get_subjects(name, fold, dimension)
        data = self.features.loc[self.features["name"].isin(subjects)]
        x_names = np.asarray(
            sorted(
                [
                    x
                    for x in data.columns
                    if x.startswith(
                        ("acoustic", "language", "vision", "ecg", "eda", "mocap")
                    )
                ]
            )
        )
        y_names = np.array([dimension])
        meta = {"x_names": x_names, "y_names": y_names}
        data = {
            "x": data.loc[:, x_names].values.astype(np.float32),
            "y": data.loc[:, y_names].values.astype(np.float32),
            **{
                key: data.loc[:, [key]].values
                for key in data.columns
                if key.startswith("meta_")
            },
        }
        return DataLoader(
            dict_to_batched_data(data, batch_size=batch_size),
            properties=meta,
        )


class SEWA(MOSI):
    def get_official_validation_test_set(self) -> tuple[list[str], list[str]]:
        return (
            [],
            [x for x in self.features["name"].unique() if x.startswith("Devel_")],
        )


class AVEC16_RECOLA(MOSI):
    ...


class MOSEI(MOSI):
    def get_official_validation_test_set(self) -> tuple[list[str], list[str]]:
        sys.path.append(
            str(
                Path(
                    "~/git/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/"
                ).expanduser()
            )
        )
        import cmu_mosei_std_folds

        return (
            cmu_mosei_std_folds.standard_valid_fold,
            cmu_mosei_std_folds.standard_test_fold,
        )


class UMEME(MOSI):
    def get_subjects(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
    ) -> list[str]:
        # speaker-split
        mapping = self.features.groupby("meta_id")["name"].unique()
        names = self.features["name"]
        self.features["name"] = self.features["meta_id"]
        subjects = super().get_subjects(name, fold, dimension)
        subjects = sum([mapping[session].tolist() for session in subjects], [])
        self.features["name"] = names
        return subjects



class IEMOCAP(MOSI):
    def get_subjects(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
    ) -> list[str]:
        # session-split
        self.features["name_"] = self.features["name"]
        self.features["name"] = self.features["name"].str[:5]
        mapping = self.features.groupby("name")["name_"].unique()
        subjects = super().get_subjects(name, fold, dimension)
        subjects = sum([mapping[session].tolist() for session in subjects], [])
        self.features["name"] = self.features.pop("name_")
        return subjects


class TPOT(MOSI):
    def __init__(self) -> None:
        super().__init__()

        folder = Path(__file__).parent.resolve()
        self.partitions = caching.read_pickle(folder / "panam.pickle")[0]

    def get_subjects(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
    ) -> list[str]:
        # split used in ICMI paper
        ids = np.unique(
            np.concatenate(
                [x["meta_id"][0] for x in self.partitions[fold][name].iterator], axis=0
            )
        )
        ids = [(f"TPOT_{abs(x):04d}", x) for x in ids]
        self.features[dimension] = self.features["constructs"]
        return sorted([f"{x}_{1 if y >0 else 2}_2" for x, y in ids])

    def get_loader(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
        batch_size: int,
    ) -> DataLoader:
        loader = super().get_loader(name, fold, dimension, batch_size)
        loader.properties["y_names"] = np.array(
            ["aggressive", "dysphoric", "other", "positive"]
        )
        for batch in loader.iterator:
            batch["y"][0] = batch["y"][0].astype(int)
        return loader


class VREED:
    def __init__(self) -> None:
        self.features = pd.read_csv(
            "/projects/dataset_processed/VREED/twoertwein/data.csv", index_col=0
        ).fillna(0)
        self.features["meta_id"] = self.features.index // 12

    def get_subjects(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
    ) -> list[int]:
        average = self.features.groupby("meta_id")[dimension].mean()

        indices = stratified_splits(average.values, np.ones(5))
        test = indices == fold
        validation = indices == (fold + 1) % 5
        training = ~(test | validation)
        match name:
            case "training":
                return average.index[training].tolist()
            case "validation":
                return average.index[validation].tolist()
            case "test":
                return average.index[test].tolist()

    def get_loader(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
        batch_size: int,
    ) -> DataLoader:
        subjects = self.get_subjects(name, fold, dimension)
        data = self.features.loc[self.features["meta_id"].isin(subjects)]

        x_names = sorted(
            [x for x in data.columns if x.startswith(("vision", "ecg", "eda"))]
        )
        data = {
            "x": data.loc[:, x_names].values.astype(np.float32),
            "y": data["av"].values[:, None],
            "meta_id": data["meta_id"].values[:, None],
        }
        return DataLoader(
            dict_to_batched_data(data, batch_size=batch_size),
            properties={
                "x_names": np.array(x_names),
                "y_names": np.array(["HAHV", "LAHV", "LALV", "HALV", "Baseline"]),
            },
        )


class TOY:
    def get_loader(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
        batch_size: int,
    ) -> DataLoader:
        rng = np.random.default_rng(fold)
        n = 500
        data = rng.multivariate_normal([0] * 7, np.eye(7), size=n).astype(np.float32)
        data = (data - data.mean(axis=0, keepdims=True)) / data.std(
            axis=0, keepdims=True
        )
        data = PCA(n_components=data.shape[1]).fit_transform(data)
        data = (data - data.mean(axis=0, keepdims=True)) / data.std(
            axis=0, keepdims=True
        )
        data[:, 6] = data[:, 6] * 0
        y = data.sum(axis=1, keepdims=True)
        A = data[:, [0, 3, 4, 6]]
        B = data[:, [1, 3, 5, 6]]
        C = data[:, [2, 4, 5, 6]]
        x_names = (
            "acoustic_unique",
            "acoustic_language",
            "acoustic_vision",
            "acoustic_all",
            "language_unique",
            "language_acoustic",
            "language_vision",
            "language_all",
            "vision_unique",
            "vision_acoustic",
            "vision_language",
            "vision_all",
        )
        data = pd.DataFrame(np.concatenate([A, B, C], axis=1), columns=x_names)
        return DataLoader(
            dict_to_batched_data(
                {
                    "x": data.loc[:, sorted(data.columns)].values,
                    "y": y,
                    "meta_id": np.ones((n, 1), dtype=int),
                }
                | {f"meta_{name}": data.loc[:, [name]].values for name in x_names},
                batch_size=batch_size,
            ),
            properties={
                "x_names": np.array(x_names),
                "y_names": np.array(["toy"]),
            },
        )


class NATOY:  # non-additive toy example
    def get_loader(
        self,
        name: Literal["training", "validation", "test"],
        fold: Literal[0, 1, 2, 3, 4],
        dimension: str,
        batch_size: int,
    ) -> DataLoader:
        rng = np.random.default_rng(fold)
        n = 500
        data = rng.multivariate_normal([0] * 4, np.eye(4), size=n).astype(np.float32)
        data = (data - data.mean(axis=0, keepdims=True)) / data.std(
            axis=0, keepdims=True
        )
        data = PCA(n_components=data.shape[1]).fit_transform(data)
        data = np.concatenate(
            [
                data * [1, 1, 1, 1],
                data * [-1, 1, 1, 1],
                data * [1, -1, 1, 1],
                data * [1, 1, -1, 1],
                data * [1, 1, 1, -1],
            ],
            axis=0,
        )
        data = (
            (data - data.mean(axis=0, keepdims=True))
            / data.std(axis=0, keepdims=True)
            / 1.5
        )
        data = data.clip(min=-3, max=3)  # otherwise clipped by transformer
        data = data.astype(np.float32)
        y = data[:, [0]] * data[:, [1]] + data[:, [2]] * data[:, [3]]
        A = data[:, [0, 2]]
        B = data[:, [0, 3]]
        C = data[:, [1]]
        x_names = (
            "acoustic_a_r_av_lv",
            "acoustic_b_u_al",
            "language_a_r_av_lv",
            "language_b_u_al",
            "vision_a_av_lv",
        )
        data = pd.DataFrame(np.concatenate([A, B, C], axis=1), columns=x_names)
        return DataLoader(
            dict_to_batched_data(
                {
                    "x": data.loc[:, sorted(data.columns)].values,
                    "y": y,
                    "meta_id": np.ones((data.shape[0], 1), dtype=int),
                }
                | {f"meta_{name}": data.loc[:, [name]].values for name in x_names},
                batch_size=batch_size,
            ),
            properties={
                "x_names": np.array(x_names),
                "y_names": np.array(["toy"]),
            },
        )


def get_partitions(dimension: str, batch_size: int) -> dict[int, dict[str, DataLoader]]:
    klass_str, dimension = dimension.split("/", 1)
    klass_str = klass_str.upper().replace("RECOLA", "AVEC16_RECOLA")
    loader = getattr(sys.modules[__name__], klass_str)()
    return {
        fold: {
            name: loader.get_loader(name, fold, dimension, batch_size)
            for name in ("training", "test", "validation")
        }
        for fold in range(5)
    }
