from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from python_tools import caching, features
from python_tools.generic import map_parallel
from python_tools.time_series import elan_to_interval_dataframe
# from sentence_transformers import SentenceTransformer
from python_tools.ml.metrics import concat_dicts

folder = Path("/projects/panam/workspace/twoertwein/TPOT2")


def extract(file: Path) -> None:
    name = file.with_suffix(".hdf").name
    # openface+opensmile
    if file.suffix == ".flac":
        cache = {"opensmile_eGeMAPSv02": folder / f"meta/opensmile_eGeMAPSv02/{name}"}
        cache = {
            "opensmile_vad_opensource": folder / f"meta/opensmile_vad_opensource/{name}"
        }
    else:
        cache = {"openface": folder / f"meta/openface/{name}"}
    features.extract_features(video=file, audio=file, caches=cache)


def extract_liwc(liwc: pd.DataFrame, text: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(0.0, columns=liwc.columns, index=text.index)
    text = text.str.lower().str.strip(" .,-'").to_list()
    texts = [
        [token for word in words.split(" ") for token in word.split(",") if token]
        for words in text
    ]
    for itext, text in enumerate(texts):
        # in LIWC dictionary
        index = pd.Index(text)
        can_lookup = index.isin(liwc.index)
        datum = liwc.loc[index[can_lookup]].sum(axis=0)
        n_tokens = can_lookup.sum()
        # tokenize and then lookup
        for word in index[~can_lookup].to_list():
            tokens = pd.Index(nltk.word_tokenize(word.strip(" .,-')(?.[]")))
            can_lookup = tokens.isin(liwc.index)
            datum = datum + liwc.loc[tokens[can_lookup]].sum(axis=0)
            n_tokens += can_lookup.sum()
        result.iloc[itext] = datum / n_tokens
    return result.fillna(0)

def load_liwc() -> pd.DataFrame:
    liwc = pd.read_csv(
        "/projects/dataset_processed/TPOT/twoertwein/liwc_mapping.csv",
        index_col=0,
        engine="python",
        converters={i: lambda x: 1.0 if x == "X" else 0.0 for i in range(75)},
        encoding_errors="ignore",
    ).drop(columns=["Unnamed: 74"])
    return liwc.loc[liwc.sum(axis=1) > 0]

if __name__ == "__main__":
    # map_parallel(extract, (folder / "audio").glob("*.flac"), workers=7)
    # map_parallel(extract, (folder / "videos").glob("*.mp4"), workers=7)

    # load LIWC
    liwc = load_liwc()

    # load annotations from one of the experiments
    annotations = concat_dicts(
        caching.read_pickle(
            Path(
                "../modality_experiments/dimension=constructs_res_det=True/497381248a4d3267a88ad8b7fc70d98fe7eb26b7b0036ea022ec4927229c6558_test_predictions.pickle"
            )
        ),
        keys=(
            "Y",
            "meta_begin",
            "meta_end",
            "meta_id",
            "meta_Evidence",
            "meta_Visual",
            "meta_Language",
            "meta_Acoustic",
            "meta_Interlocutor",
        ),
    )
    annotations = pd.DataFrame({key: value[:, 0] for key, value in annotations.items()})
    annotations["meta_id"] = annotations["meta_id"].apply(
        lambda x: f"TPOT_{abs(x):04d}_{2 if x < 0 else 1}_2"
    )

    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    data = []
    # UPitt says rater KH has more percise onset annotation
    eafs = list((folder / "labels").glob("*.eaf"))
    eaf_names = [eaf.with_suffix("").name for eaf in eafs]
    eafs = [
        eaf
        for eaf in eafs
        if "KH" in eaf.name
        or (f"{'_'.join(eaf.name.split('_')[:4])}_KH" not in eaf_names)
    ]
    Y_NAMES = ["Aggressive", "Dysphoric", "Other", "Positive"]
    sentences = []
    for eaf in eafs:
        labels = annotations.loc[
            annotations["meta_id"] == "_".join(eaf.name.split("_")[:4])
        ].copy()
        labels["constructs"] = labels.pop("Y")
        labels.index = pd.IntervalIndex.from_arrays(
            labels.pop("meta_begin"), labels.pop("meta_end")
        )
        """
        # load labels
        labels = (
            elan_to_interval_dataframe(eaf, remove_empty_tiers=False)
            .fillna(False)
            .loc[:, Y_NAMES]
            .sort_index()
        )
        labels = (labels * [0, 1, 2, 3]).sum(axis=1).to_frame("constructs")
        # shift by 1 second, until the next onset but atmost 30s long
        labels.index = pd.IntervalIndex.from_arrays(
            labels.index.left - 1,
            np.minimum(
                labels.index.left + 29,
                np.concatenate(
                    [labels.index.left[1:] - 1, [labels.index.left[-1] + 29]], axis=0
                ),
            ),
        )
        """

        # find words
        file = folder / "transcript" / f'{"_".join(eaf.name.split("_")[:2])}_0_2.eaf'
        if not file.is_file():
            continue
        words = elan_to_interval_dataframe(file)
        if "_2_" in eaf.name:
            if "Adolescent" not in words.columns:
                continue
            words = words["Adolescent"]
        else:
            words = words["Parent"]
        words = words[words.notna() & words.apply(lambda x: isinstance(x, str))]
        """
        labels = labels.loc[
            (labels.index.left >= words.index.left.min())
            & (labels.index.right <= words.index.right.max())
        ]
        """
        labels["sentence"] = ""

        for index, row in labels.iterrows():
            word_index = (words.index.right >= index.left) & (
                words.index.left <= index.right
            )
            labels.loc[index, "sentence"] = " ".join(words.loc[word_index].to_list())
        labels["meta_id"] = eaf.with_suffix("").name[:-3]
        labels["meta_begin"] = labels.index.left
        labels["meta_end"] = labels.index.right

        sentences.append(labels["sentence"])
        liwc_features = extract_liwc(liwc, labels["sentence"])
        # embeddings = pd.DataFrame(
        #     model.encode(labels.pop("sentence").str.lower().tolist()),
        #     index=labels.index,
        # )

        data.append(
            pd.concat(
                [
                    labels,
                    # embeddings.add_prefix("all_minilm_l12_v2_"),
                    liwc_features.add_prefix("liwc_"),
                ],
                axis=1,
            ).reset_index(drop=True)
        )
    pd.concat(sentences, ignore_index=True).to_csv("panam_sentences.csv")
    data = pd.concat(data, axis=0).reset_index(drop=True)

    # remove some sessions marked as bad
    bad = (
        "TPOT_9524",
        "TPOT_8988",
        "TPOT_8693",
        "TPOT_2872",
        "TPOT_2620",
        "TPOT_4429",
        "TPOT_6896",
    )
    # keep sessions with both parent and child
    ids = data["meta_id"].unique()
    bad = bad + tuple(
        "_".join(x.split("_")[:2])
        for x in ids
        if "_1_" in x and x.replace("_1_", "_2_") not in ids
    )
    data = data.loc[data["meta_id"].apply(lambda x: x[:9] not in bad)]

    """
    # load annotations from one of the experiments
    annotations = concat_dicts(
        caching.read_pickle(
            Path(
                "../modality_experiments/dimension=constructs_res_det=True/497381248a4d3267a88ad8b7fc70d98fe7eb26b7b0036ea022ec4927229c6558_test_predictions.pickle"
            )
        ),
        keys=(
            "meta_begin",
            "meta_end",
            "meta_id",
            "meta_Evidence",
            "meta_Visual",
            "meta_Language",
            "meta_Acoustic",
            "meta_Interlocutor",
        ),
    )
    annotations = pd.DataFrame({key: value[:, 0] for key, value in annotations.items()})
    tiers = [
        "meta_Evidence",
        "meta_Visual",
        "meta_Language",
        "meta_Acoustic",
        "meta_Interlocutor",
    ]
    annotations = annotations.loc[annotations.loc[:, tiers].notna().any(axis=1)]
    annotations["meta_id"] = annotations["meta_id"].apply(
        lambda x: f"TPOT_{abs(x):04d}_{2 if x < 0 else 1}_2"
    )
    data.loc[:, tiers] = float("NaN")
    for _, row in annotations.iterrows():
        compare = data.loc[data["meta_id"] == row["meta_id"], "meta_begin"]
        distance = (compare - row["meta_begin"]).abs().min()
        if distance < 0.5:
            index = (compare - row["meta_begin"]).abs().idxmin()
            for tier in tiers:
                data.loc[index, tier] = row[tier]
    """

    caching.write_hdfs(folder / "meta/all_minilm_l12_v2.hdf", {"df": data})
