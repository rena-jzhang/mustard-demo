#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues = false
"""A function designed to validate hyper-parameters."""
import os
import shutil
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
from dask import delayed, graph_manipulation
from dask.delayed import Delayed

from python_tools import caching, generic
from python_tools.ml.default.transformations import (
    revert_transform as _revert_transform,
)
from python_tools.ml.metrics import average_list_dict_values
from python_tools.ml.model import Model
from python_tools.typing import (
    DataLoader,
    MetricWrappedFun,
    TransformDefineFun,
    TransformRevertFun,
    TransformSetFun,
)

TransFormRevertPartialFun = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
_DEFAULT_TMP_FOLDER: Final = Path("/scratch")


def evaluator(
    *,
    models: tuple[Model, ...] = (),
    partitions: dict[int, dict[str, DataLoader]],
    metric_fun: MetricWrappedFun,
    metric: str,
    metric_max: bool,
    concat_folds: bool = False,
    folder: Path,
    parameters: tuple[dict[str, Any], ...] = (),
    transform_parameter: tuple[dict[str, Any], ...] = (),
    learn_transform: TransformDefineFun,
    apply_transform: TransformSetFun,
    revert_transform: TransformRevertFun = _revert_transform,
    tmp_folder: Path = _DEFAULT_TMP_FOLDER,
    debug: str = "",
    **kwargs: Any,
) -> pd.DataFrame:
    """Framework to evaluate hyper-parameters.

    Args:
        models: Tuple of models (see python_tools.ml.model).
        partitions: A dictionary (fold number) of dictionaries ('training',
            'validation', 'test', ...). The values are instances of
            python_tools.ml.data_loader. If 'validation' is not present,
            'training' will be used for validation.
        metric_fun: A function returning a dictionary of the performance metrics.
        metric: The metric used for choosing the best model. Only the best model
            (according to the validation set) is evaluated on the test set.
        metric_max: Whether the maximum or the minimum is the best for the metric.
        concat_folds: Whether to concatenate the data from all folds to calculate
            metrics or whether to calculate metrics for each fold and then average.
        folder: Write all intermrediate and final files to this folder.
        parameters: Tuple of dictionaries representing possible parameters for the
            models.
        transform_parameter: Same as parameters but for model-independent
            transformations of the data Transformation parameters are not validated,
            each fold has exactly one transformation.
        learn_transform: Designed to initialize a data-driven data pre-processing
            (scaling features, PCA, Lasso).
        apply_transform: Applies the learned transform to the actual partition.
        revert_transform:  Reverts the potentially scaled predictions to the original
            scale.
        tmp_folder: If enough space, will use this folder for temporary file.
        debug: Run only the model with the given hash.
        **kwargs: Are forwarded to python_tools.generic.run_distributed.

    Returns:
        A DataFrame summarizing the gridsearch.
    """
    # init
    assert models
    assert len(models) == len(parameters)
    assert len(partitions) == len(transform_parameter)
    folder = generic.fullpath(folder)
    folder.mkdir(exist_ok=True, parents=True)
    kwargs.setdefault("setup", select_gpu)
    if not tmp_folder.is_dir():
        tmp_folder = folder

    if skip_evaluation(folder, parameters, partitions, metric, metric_max):
        return pd.read_csv(folder / "overview.csv", index_col=0)

    # split partitions: training and validation vs test and additional
    partitions_test = {}
    training = ("training", "validation")
    partitions = partitions.copy()
    for name, partition in partitions.items():
        partitions_test[name] = {
            key: partition[key] for key in partition if key not in training
        }
        partitions[name] = {key: partition[key] for key in partition if key in training}

    # temporary folder
    with tempfile.TemporaryDirectory(
        dir=tmp_folder,
    ) as fast_tmp, tempfile.TemporaryDirectory(dir=folder) as tmp:
        # build graph
        task = build_graph(
            parameters,
            models,
            partitions,
            revert_transform,
            folder,
            transform_parameter,
            learn_transform,
            apply_transform,
            metric_fun,
            metric,
            concat_folds,
            metric_max,
            partitions_test,
            (Path(fast_tmp), Path(tmp)),
            debug=debug,
        )

        # run graph
        best_index, overviews, best_pairs, parameter_columns = generic.run_distributed(
            [task],
            **kwargs,
        )[0]
    overview = pd.DataFrame(overviews)
    assert overview["name"].nunique() == overview.shape[0]

    # rank parameters
    rank_parameters(overview, parameter_columns, metric, metric_max=metric_max)

    # write predictions and delete old predictions
    for file in folder.glob("*_predictions.pickle"):
        file.unlink()
    prefix = overview.loc[best_index, "name"]
    for key in best_pairs:
        caching.write_pickle(folder / f"{prefix}_{key}_predictions", best_pairs[key])

    overview.to_csv(folder / "overview.csv")

    if not debug:
        clean(folder, overview["name"].tolist())
    return overview


def clean(folder: Path, hashes: list[str]) -> None:
    """Remove unused files.

    These files might be from a different gridsearch.

    Args:
        folder: The folder containing the gridsearch data.
        hashes: The model hashes for the current gridsearch.
        folds: The number of folds.
    """
    for file in folder.iterdir():
        name = file.name

        # old tmp* files
        if file.is_dir():
            if name.startswith("tmp"):
                shutil.rmtree(file)
            continue

        # skip
        if (
            # the overview file
            name == "overview.csv"
            # files related to partitions
            or (name.startswith("partition_") and len(name.split("_")) == 2)
            # files related to current model
            or any(model in name for model in hashes)
        ):
            continue

        file.unlink()


def run_model(
    model: Model,
    name_parameter: str,
    name_partition: str,
    parameter: dict[str, Any],
    partition_revert: tuple[dict[str, DataLoader], TransFormRevertPartialFun],
    folder: Path,
    metric_fun: MetricWrappedFun,
    tmp_folders: tuple[Path, Path],
) -> Path:
    """Run an individual model and returns its data.

    Args:
        model: The model wrapper.
        name_parameter: Hash of the model.
        name_partition: Name of the partition.
        parameter: Parameters for the model.
        partition_revert: The learned transform.
        folder: Folder where to store to/load from results.
        metric_fun: Function returning several metrics.
        tmp_folders: Write temporary files to these folders.

    Returns:
        Path to the saved model output.
    """
    partition, revert = partition_revert

    # training as validation
    if "training" in partition and "validation" not in partition:
        partition["validation"] = partition["training"]

    # load model
    basename = folder / f"partition_{name_partition}_{name_parameter}"
    trained, output = model.load(basename)
    print(basename.name, trained)

    if trained is None:
        # or init, train, and save model
        try:
            trained, output = model.fit(
                partition["training"],
                partition["validation"],
                metric_fun,
                basename,
                **parameter,
            )
        except Exception as err:
            err.add_note(basename.name)
            raise
        model.save(basename, trained, output, **parameter)

    # summary
    result = {
        "name": name_parameter,
        "parameters": {
            key: str(fun_to_string(parameter[key])) for key in sorted(parameter)
        },
        "model": model,
        "output": output,
        "pairs": {
            key: revert(model.predict(trained, value, key))
            for key, value in partition.items()
        },
    }

    # write result to disk, return path
    tmp_folder = determine_temporary_storage(*tmp_folders)
    return caching.write_pickle(tmp_folder / f"{basename.name}_run_model", result)


def fun_to_string(fun: object) -> Any:
    """Convert a function-like object to a string.

    Args:
        fun: A function.

    Returns:
        The same object or a string representing the function.
    """
    if hasattr(fun, "__name__"):
        fun = fun.__name__
    if hasattr(fun, "func") and hasattr(fun.func, "name"):
        fun = fun.func.name
    if hasattr(fun, "func") and hasattr(fun.func, "__name__"):
        return fun.func.__name__
    return fun


def dict_to_string(dictionary: dict[str, Any]) -> str:
    """Convert a dictionary to a string.

    Args:
        dictionary: Any dictionary.

    Returns:
        A compact string representing the dictionary.
    """
    parts = []
    for key in sorted(dictionary):
        value = dictionary[key]
        if isinstance(value, np.ndarray):
            value = "ndarray"
        value = fun_to_string(value)
        if isinstance(value, dict):
            value = dict_to_string(value)
        parts.append(f"{key}_{value}")
    return "_".join(parts).lower().replace(".", "-")


def build_graph(
    parameters: tuple[dict[str, Any], ...],
    models: tuple[Model, ...],
    partitions_: dict[int, dict[str, DataLoader]],
    revert_transform: TransformRevertFun,
    folder: Path,
    transform_parameter: tuple[dict[str, Any], ...],
    learn_transform: TransformDefineFun,
    apply_transform: TransformSetFun,
    metric_fun: MetricWrappedFun,
    metric: str,
    concat_folds: bool,
    metric_max: bool,
    test_partitions_: dict[int, dict[str, DataLoader]],
    tmp_folders: tuple[Path, Path],
    debug: str = "",
) -> Delayed:
    """Build the computation graph.

    Args:
        parameters: Tuple of dictionaries representing possible parameters for the
            models.
        models: Tuple of models (see python_tools.ml.model).
        partitions_: A dictionary (fold number) of dictionaries ('training',
            'validation', 'test', ...). The values are instances of
            python_tools.ml.data_loader. If 'validation' is not present, 'training'
            will be used for validation.
        revert_transform: Reverts the potentially scaled predictions to the original
            scale.
        folder: Write all intermrediate and final files to this folder.
        transform_parameter: Same as parameters but for model-independent
            transformations of the data Transformation parameters are not validated,
            each fold has exactly one transformation.
        learn_transform: Designed to initialize a data-driven data pre-processing
            (scaling features, PCA, Lasso).
        apply_transform: Applies the learned transform to the actual partition.
        metric_fun: A function returning a dictionary of the performance metrics.
        metric: The metric used for choosing the best model. Only the best model
            (according to the validation set) is evaluated on the test set.
        concat_folds: Whether to concatenate the data from all folds to calculat
            metrics or whether to calculate metrics for each fold and then average.
        metric_max: Whether the maximum or the minimum is the best for the metric.
        test_partitions_: Same as partitions_ but "test" and non-training and
            non-validation keys.
        tmp_folders: Write temporary files to these folders. Prefer the first
           folder if enough space is available.
        debug: Run only the model with the given hash.

    Returns:
        The delayed object representing the gridsearch.
    """
    graph: dict[str, Delayed] = {}
    partitions = {
        key: {
            name: delayed(loader, pure=True, traverse=False)
            for name, loader in value.items()
        }
        for key, value in partitions_.items()
    }
    test_partitions = {
        key: {
            name: delayed(loader, pure=True, traverse=False)
            for name, loader in value.items()
        }
        for key, value in test_partitions_.items()
    }
    # add transformation
    for (pname, partition), parameter in zip(
        partitions.items(),
        transform_parameter,
        strict=True,
    ):
        graph[shorten_name(f"node_partition_{pname}")] = delayed(
            run_transform,
            pure=True,
            nout=2,
        )(
            folder / f"partition_{pname}",
            learn_transform,
            apply_transform,
            revert_transform,
            partition,
            parameter,
        )

    # buffer to heuristically determine what the best score is so far
    best_score = [float("inf")]
    if metric_max:
        best_score = [-float("inf")]
    best_score_delayed = delayed(best_score)

    # add models
    overviews: list[Delayed] = []
    for model, parameter in zip(models, parameters, strict=True):
        name_parameter = generic.hashname(
            f"model_{model}_{dict_to_string(parameter)}".lower().replace(".", "-"),
        )
        if debug:
            if name_parameter != debug:
                continue
            parameters = (parameter,)
            models = (model,)
        fold_nodes: list[Delayed] = []
        for pname in partitions:
            parent = shorten_name(f"node_partition_{pname}")
            fold_nodes.append(
                delayed(run_model, pure=True)(
                    model,
                    name_parameter,
                    pname,
                    parameter,
                    graph[parent],
                    folder,
                    metric_fun,
                    tmp_folders,
                ),
            )
        # merge folds
        overviews.append(
            delayed(print_results, pure=True, nout=3)(
                metric_fun,
                concat_folds,
                best_score_delayed,
                metric,
                metric_max,
                tmp_folders,
                *fold_nodes,
            ),
        )

    # merge overviews and find best model (might be the last node)
    graph["merge_results"] = delayed(merge_results, pure=True, nout=4)(
        metric,
        metric_max,
        *overviews,
    )
    if sum(len(partition) for partition in test_partitions.values()) == 0:
        return graph["merge_results"]

    # test
    # transformations
    for (pname, partition), parameter in zip(
        test_partitions.items(),
        transform_parameter,
        strict=True,
    ):
        graph[shorten_name(f"node_test_partition_{pname}")] = graph_manipulation.bind(
            delayed(run_transform, pure=True, nout=2),
            graph["merge_results"],
        )(
            folder / f"partition_{pname}",
            learn_transform,
            apply_transform,
            revert_transform,
            partition,
            parameter,
        )

    # run the best model
    fold_nodes = []
    for pname in test_partitions:
        parent = shorten_name(f"node_test_partition_{pname}")
        fold_nodes.append(
            delayed(run_best_model, pure=True)(
                models,
                parameters,
                graph[parent],
                folder,
                metric_fun,
                graph["merge_results"][0],  # type: ignore[index]
                pname,
                tmp_folders,
            ),
        )

    # merge test folds
    graph["test_results"] = delayed(print_results, pure=True, nout=3)(
        metric_fun,
        concat_folds,
        [best_score[0]],
        metric,
        metric_max,
        tmp_folders,
        *fold_nodes,
    )

    # merge training and test
    return delayed(merge_valdiation_results, pure=True, nout=4)(
        graph["merge_results"],
        graph["test_results"],
    )


def run_best_model(
    models: tuple[Model, ...],
    parameters: tuple[dict[str, Any], ...],
    partition_revert: tuple[dict[str, DataLoader], TransFormRevertPartialFun],
    folder: Path,
    metric_fun: MetricWrappedFun,
    best_index: int,
    partition_name: str,
    tmp_folders: tuple[Path, Path],
) -> Path:
    """Determine the best model and run it.

    Args:
        models: All model wrappers.
        parameters: All model parameters.
        partition_revert: The learned transform.
        folder: Folder where to store to/load from results.
        metric_fun: Function returning several metrics.
        best_index: Indicate the index of the best model.
        partition_name: Name of the partition.
        tmp_folders: Write temporary files to these folders.

    Returns:
        The output of the best model.
    """
    # determine best model
    model = models[best_index]
    parameter = parameters[best_index]
    name_parameter = generic.hashname(
        f"model_{model}_{dict_to_string(parameter)}".lower().replace(".", "-"),
    )

    return run_model(
        model,
        name_parameter,
        partition_name,
        parameter,
        partition_revert,
        folder,
        metric_fun,
        tmp_folders,
    )


def print_results(
    metric_fun: MetricWrappedFun,
    concat_folds: bool,
    best_score: list[float],
    metric: str,
    metric_max: bool,
    tmp_folders: tuple[Path, Path],
    *summaries_files: Path,
) -> tuple[dict[str, Any], list[str], Path | None]:
    """Aggregate all folds for one model.

    Args:
        metric_fun: Function to calculate metrics.
        concat_folds: Whether to concatenate folds before calculating metrics.
        best_score: Contains the best metric so far. Heuristic to avoid some disk IO.
        metric: Name of the scoring metric.
        metric_max: Whether higher values of the metric are best.
        tmp_folders: Write temporary files to these folders.
        *summaries_files: Paths to the individual fold files.

    Returns:
        The overview data.
        Which columns in overview represent parameter names.
        If the fold might be the best fold, path to its prediction data.
    """
    # read and delete summary files
    summaries: list[dict[str, Any]] = []
    for path in summaries_files:
        data = caching.read_pickle(path)
        assert data is not None, path
        summaries.append(data)
        path.unlink(missing_ok=True)

    # add parameters
    result: dict[str, Any] = {
        key: str(value) for key, value in summaries[0]["output"].items()
    }
    result["name"] = summaries[0]["name"]
    model = summaries[0]["model"]
    result["model"] = str(model)
    result["parameters"] = summaries[0]["parameters"]
    result.update(
        {f"parameter_{key}": value for key, value in result["parameters"].items()},
    )
    parameter_columns = [f"parameter_{key}" for key in result["parameters"]]

    # add metrics
    pairs = {}
    for partition in summaries[0]["pairs"]:
        pairs[partition] = [summary["pairs"][partition] for summary in summaries]
        scores = (
            model.metric(pairs[partition], metric_fun)
            if concat_folds
            else average_list_dict_values(
                [model.metric([x], metric_fun) for x in pairs[partition]],
            )
        )

        for key in scores:
            result[f"{partition}_{key}"] = scores[key]

    # add mean optimization
    if "0_optimization" in summaries[0]["output"]:
        result["optimization"] = np.mean(
            [summary["output"]["0_optimization"] for summary in summaries],
        )

    # check whether the current fold might be the best fold, this does not lock!
    metric_name = f"validation_{metric}"
    if metric_name in result:
        score = result[metric_name]
        is_best = (metric_max and score > best_score[0]) or (
            not metric_max and score < best_score[0]
        )
        if is_best:
            best_score[0] = score
        else:
            return result, parameter_columns, None

    # write result to disk, return path
    tmp_folder = determine_temporary_storage(*tmp_folders)
    return (
        result,
        parameter_columns,
        caching.write_pickle(tmp_folder / f"{result['name']}_print_results", pairs),
    )


def merge_valdiation_results(
    merge_results_output: tuple[
        int,
        list[dict[str, Any]],
        dict[str, list[Any]],
        set[str],
    ],
    print_output: tuple[dict[str, Any], list[str], Path | None],
) -> tuple[int, list[dict[str, Any]], dict[str, list[Any]], set[str]]:
    """Combine previously aggregated training&valdaition results with test results.

    Args:
        merge_results_output: The training&validation data.
        print_output: The test data.

    Returns:
        The same as merge_results.
    """
    best_index, overviews, best_pairs, parameter_columns = merge_results_output
    overview, _, pairs_path = print_output
    assert pairs_path is not None

    overviews[best_index].update(overview)
    best_pairs.update(caching.read_pickle(pairs_path))
    pairs_path.unlink(missing_ok=True)
    return best_index, overviews, best_pairs, parameter_columns


def _lambda_score(
    item: tuple[dict[str, Any], list[str], Path | None],
    name: str = "",
) -> float:
    return item[0][name]


def merge_results(
    metric: str,
    metric_max: bool,
    *summaries: tuple[dict[str, Any], list[str], Path | None],
) -> tuple[int, list[dict[str, Any]], dict[str, list[Any]], set[str]]:
    """Combine summaries from inidivudal models.

    Args:
        metric: Name of the scoring metric.
        metric_max: Whether higher vales of the metric are best.
        *summaries: Output form print_output.

    Returns:
        The index of the best model.
        The overview data.
        The prediction data.
        Column name in overview that incidate parameter names.
    """
    # create a nice summary
    overviews = []
    best_pairs_path = Path()
    best_index = -1
    name = f"validation_{metric}"
    best_metric = (
        max(summaries, key=partial(_lambda_score, name=name))[0][name]
        if metric_max
        else min(summaries, key=partial(_lambda_score, name=name))[0][name]
    )

    # create summary
    parameter_columns = set()
    for i, data in enumerate(summaries):
        overview, parameter_columns_, pairs_path = data
        parameter_columns.update(parameter_columns_)
        overviews.append(overview)

        if pairs_path is None:
            pass
        elif overview[name] == best_metric:
            # find best predictions
            best_pairs_path = pairs_path
            best_index = i
        else:
            pairs_path.unlink(missing_ok=True)
    assert best_index >= 0
    best_pairs: dict[str, list[Any]] = caching.read_pickle(best_pairs_path)
    best_pairs_path.unlink(missing_ok=True)

    return best_index, overviews, best_pairs, parameter_columns


def run_transform(
    name: Path,
    learn: TransformDefineFun,
    apply: TransformSetFun,
    revert: TransformRevertFun,
    partition: dict[str, DataLoader],
    parameter: dict[str, Any],
) -> tuple[dict[str, DataLoader], TransFormRevertPartialFun]:
    """Learns and applies a transform.

    Args:
        name: Where to save/load a transformation to/from.
        learn: Function to learn a transform (on the training set).
        apply: Function to apply a transform.
        revert: Function to revert a transform.
        partition: The dataloader.
        parameter: Parameters for the transform.

    Returns:
        The dataloaders with the transform applied.
        Function to revert the transform.
    """
    # try to load transform
    transform = caching.read_pickle(name)
    if transform is None:
        # learn transform
        transform = learn(partition["training"], **parameter)
        # save transformation
        caching.write_pickle(name, transform)

    # apply transform
    partition = {key: apply(partition[key], transform) for key in partition}

    return partition, partial(revert, transform=transform)


def rank_parameters(
    overview: pd.DataFrame,
    parameter_columns: set[str],
    metric: str,
    *,
    metric_max: bool = True,
) -> dict[str, list[float]]:
    """Ranks the parameters (integrates over all other parameters).

    Args:
        overview: Pandas dataframe containing the parameter and metrics.
        parameter_columns: List of columns which represent parameters.
        metric: Name of the column with the metric ('validation_' is added).
        metric_max: Whether the maximum or minimum of the metric is best.

    Returns:
        Dictionary for each parameter with values in descending order.
    """
    rank: dict[str, list[float]] = {}
    metric = f"validation_{metric}"
    overview = overview.fillna("")
    mean_performance: pd.Series
    for parameter in sorted(parameter_columns):
        # parameter which has at least 2 values
        values = overview[parameter].unique()
        if values.size < 2:
            continue

        # collect mean metrics
        mean_performance = (
            overview.groupby(parameter)[metric]
            .mean()
            .sort_values(ascending=not metric_max)
        )

        # add values in descending order
        rank[parameter] = []
        for index in mean_performance.index:
            performance = mean_performance[index]
            rank[parameter].append(performance)
            print(parameter.replace("parameter_", ""), index, performance)
    return rank


def find_rows_with_matching_parameters(
    overview: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.Series:
    """Find rows in overview with matching parameters.

    Args:
        overview: The overview (might be from to_csv)
        parameters: The parameters.

    Returns:
        A series indexing into the overview.
    """
    index = pd.Series(True, index=overview.index)
    for name, value in parameters.items():
        column_name = f"parameter_{name}"
        if column_name not in overview.columns:
            continue
        # find type (might be converted to string)
        dtype = type(overview.loc[~overview[column_name].isna(), column_name].iloc[0])
        # convert to same type
        overview[column_name] = overview[column_name].astype(dtype)
        if dtype == str:
            overview.loc[overview[column_name] == "nan", column_name] = ""
        # compare
        index &= overview[column_name] == dtype(value)
    return index


def select_gpu(dask_worker: Any) -> None:
    """Assign a GPU to each worker."""
    gpus = torch.cuda.device_count()
    if gpus < 2:
        return

    os.environ["MY_CUDA_VISIBLE_DEVICES"] = str(dask_worker.name % gpus)


def shorten_name(name: str) -> str:
    """Shortens a string.

    Args:
        name: The potentially long string.

    Returns:
        The shortened string.
    """
    if len(name) > 256:
        return f"shortened_{generic.hashname(name)}"
    return name


def determine_temporary_storage(fast_tmp: Path, fallback: Path) -> Path:
    """Queries a space-limited but fast directory.

    Args:
        fast_tmp: The fast but space-limited directory.
        fallback: The slower alternative.

    Returns:
        The temporary folder.
    """
    # should have at least 5GB
    if generic.available_disk_space(fast_tmp) < 5:
        return fallback
    return fast_tmp


def skip_evaluation(
    folder: Path,
    parameters: tuple[dict[str, Any], ...],
    partitions: dict[int, dict[str, DataLoader]],
    metric: str,
    metric_max: bool,
) -> bool:
    """Return whether the evaluation can probably be skipped.

    Args:
        folder: The folder with all the saved files.
        parameters: Tuple of all parameters.
        partitions: The dict of dicts containing the data loaders.
        metric: Metric to determine the best model with.
        metric_max: Whether higher values are better for the metric.

    Returns:
        Whether to skip or not.
    """
    # overview has to exist
    overview_file = folder / "overview.csv"
    if not overview_file.is_file():
        return False
    # partition_?.pickle have to exist
    for partition in partitions:
        transform_file = folder / f"partition_{partition}.pickle"
        if not transform_file.is_file():
            return False
    # _{}_predictions.pickle have to exist
    for dataset in next(iter(partitions.values())):
        matches = list(folder.glob(f"*_{dataset}_predictions.pickle"))
        if len(matches) != 1 or not matches[0].is_file():
            return False
    # same metric to choose test set
    overview = pd.read_csv(overview_file, index_col=0)
    columns = overview.columns
    if f"validation_{metric}" not in columns:
        return False
    if any(x.startswith("test_") for x in columns):
        if f"test_{metric}" not in columns:
            return False
        best_index = overview[f"test_{metric}"].idxmax()
        scores = overview[f"validation_{metric}"]
        validation_score = scores.loc[best_index]
        if metric_max and scores.max() > validation_score:
            return False
        if not metric_max and scores.min() < validation_score:
            return False
    # same number of parameter combinations
    if overview.shape[0] != len(parameters):
        return False
    # parameters in overview have to match
    parameter_strings = [
        str({key: str(fun_to_string(parameter[key])) for key in sorted(parameter)})
        for parameter in parameters
    ]
    return (overview["parameters"] == parameter_strings).all()
