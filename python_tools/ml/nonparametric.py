#!/usr/bin/env python3
"""Non-parametric models."""
from math import ceil, log, pi, sqrt
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional

import torch
from beartype import beartype

from python_tools.ml import neural, neural_wrapper


class DWACActivation(neural.LossModule):
    """Deep Weighted Averaging Classifiers/Regressors."""

    regression: Final[bool]
    exclude_same_id: Final[bool]
    clustered_similarity: Final[bool]
    cache_keys: Final[tuple[str, ...]]

    data: Dict[str, List[torch.Tensor]]
    cache: Dict[str, torch.Tensor]
    # for type checkers
    output_size: torch.Tensor
    x_t: torch.Tensor
    ids: torch.Tensor
    x_t_weight: torch.Tensor
    y: torch.Tensor

    @beartype
    def __init__(
        self,
        *,
        regression: bool,
        output_size: tuple[int, ...],
        embedding_size: int,
        exclude_same_id: bool = False,
        similarity: Literal[
            "rbf_similarity",
            "cosine_similarity",
            "frechet_similarity",
        ] = "rbf_similarity",
        cache_keys: tuple[str, ...] = ("x_t_norm",),
        clustered_similarity: bool = False,
        log_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Deep Weighted Averaging Classifiers/Regressors.

        Args:
            regression: Whether it is used for regression or classification.
            output_size: List of number of classes for classification. Or a
                one-element list to indicate multivariate regression.
            embedding_size: Size of the embedding.
            exclude_same_id: Whether to exclude comparisons during training
                to the same ID.
            similarity: Which similarity function to use.
            cache_keys: Which fields of the similarity function should be cached.
            clustered_similarity: Whether it is a clustered similarity function.
            log_scale: Scale factor for the similarity function.
            kwargs: Forwarded to LossModule.
        """
        super().__init__(training_validation=True, **kwargs)
        if similarity == "frechet_similarity":
            clustered_similarity = True
            cache_keys = ("unique_ids_t", "means_t", "covs_t", "id_t_index")
        if kwargs.get("attenuation", ""):
            embedding_size -= 1
        self.regression = regression
        self.exclude_same_id = exclude_same_id
        self.clustered_similarity = clustered_similarity

        self.batch_norm = torch.nn.BatchNorm1d(embedding_size, affine=False)

        self.length_scale = torch.nn.Parameter(torch.FloatTensor([log_scale]))
        self.similarity = getattr(neural, similarity)
        self.cache_keys = cache_keys

        # aggregate the training set
        self.data = {}
        # ... and then moved into save-able buffer
        self.register_buffer("x_t", None)
        self.register_buffer("ids", None)
        self.register_buffer("x_t_weight", None)
        self.register_buffer("y", None)
        self.register_buffer("output_size", torch.LongTensor(output_size))
        # JIT doesn't support setattr: cannot save cached statistics
        self.cache = {}

        self._reset()

    def _reset(self) -> None:
        self.x_t = torch.zeros(0)
        self.ids = torch.zeros(0, dtype=torch.long)
        self.x_t_weight = torch.zeros(0)
        self.y = torch.zeros(0, dtype=torch.float32 if self.regression else torch.int64)
        self.data = {"x_t": [], "ids": [], "x_t_weight": [], "y": []}
        self.cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """House keeping.

        We want to use the entire training set as reference set for
        validation/testing. The embedding of the training set is changing during
        training, we have a second training phase without updating the model.
        """
        assert y is not None
        meta["meta_embedding"] = x
        x = self.batch_norm(x)

        x_t, y_, x_t_weight, ids_t, same = self._data_managment(x, meta, y, dataset)

        y_hat = self.dwac_transform(
            x,
            x_t,
            y_,
            meta["meta_id"].view(-1),
            x_t_weight,
            ids_t,
            same,
        )
        meta["meta_y_hat"] = y_hat
        return y_hat, meta

    def _data_managment(
        self,
        x: torch.Tensor,
        meta: Dict[str, torch.Tensor],
        y: torch.Tensor,
        dataset: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        # reset every time a new training starts
        if dataset == "training" and len(self.data["x_t"]) == 0:
            self._reset()

        # freeze collected training data during first validation run
        # and make it saveable/transferable
        if dataset == "validation" and len(self.data["x_t"]) > 0:
            self.x_t = torch.cat(self.data["x_t"], dim=1)
            self.data["x_t"].clear()

            self.y = torch.cat(self.data["y"], dim=1)
            self.data["y"].clear()

            self.ids = torch.cat(self.data["ids"], dim=0)
            self.data["ids"].clear()

            if self.data["x_t_weight"][0].shape[0] != 0:
                self.x_t_weight = torch.cat(self.data["x_t_weight"], dim=1)
            self.data["x_t_weight"].clear()

        # training: current batch as reference
        same = True
        if dataset == "training":
            x_t = x.t()
            x_t_weight = (
                meta["meta_weights"].t().clamp(min=1e-8, max=1.0)
                if "meta_weights" in meta
                else torch.zeros(0)
            )
            y_ = y.view(1, y.shape[0], y.shape[1])
            ids_t = meta["meta_id"]
            # save it if the model is not changing
            if not self.training:
                self.data["x_t"].append(x_t.detach())
                self.data["y"].append(y_)
                assert meta["meta_id"].dtype == torch.long, meta["meta_id"].dtype
                self.data["ids"].append(meta["meta_id"])
                self.data["x_t_weight"].append(x_t_weight.detach())
        else:
            # training data as reference
            same = False
            x_t = self.x_t
            y_ = self.y
            x_t_weight = self.x_t_weight
            ids_t = self.ids

        return x_t, y_, x_t_weight, ids_t, same

    def dwac_transform(
        self,
        x: torch.Tensor,
        x_t: torch.Tensor,
        y: torch.Tensor,
        ids: torch.Tensor,
        x_t_weight: torch.Tensor,
        ids_t: torch.Tensor,
        same: bool,
    ) -> torch.Tensor:
        """Actual implementation of 'Deep Weighted Averaging Classifiers'."""
        # calculate distances to reference
        length_scale = torch.exp(self.length_scale)
        similarity, cache = self.similarity(
            x,
            x_t,
            length_scale,
            self.cache,
            ids=ids,
            ids_t=ids_t,
            same=same,
        )
        if not self.training and not same:
            for key in self.cache_keys:
                self.cache[key] = cache[key].detach()

        id_index_reverse: Optional[torch.Tensor] = None
        if self.clustered_similarity:
            id_index_reverse = torch.nonzero(ids.view(-1, 1) == cache["unique_ids"])[
                :,
                1,
            ]
            ids = cache["unique_ids"].view(-1)
            ids_t = cache["unique_ids"].view(-1, 1)
            y = y[:, cache["id_t_index"]]
            if x_t_weight.shape[0] > 0:
                x_t_weight = x_t_weight[cache["id_index"]]
        if x_t_weight.shape[0] > 0:
            similarity = similarity * x_t_weight

        # ignore comparisons to self
        if self.training and not self.exclude_same_id:
            # self = sample
            similarity = similarity * (
                1.0 - torch.eye(similarity.shape[0], device=x.device, dtype=x.dtype)
            )
        elif self.training:
            # self = identifier
            similarity = similarity * (ids.view(1, -1) != ids.view(-1, 1)).to(
                dtype=similarity.dtype,
                device=similarity.device,
            )
        similarity = similarity.clamp(min=1e-8)

        # derive labels
        if self.regression:
            similarity = similarity.mm(y[0]) / similarity.sum(dim=1, keepdim=True)
        else:
            mask = torch.zeros(
                similarity.shape[1],
                self.output_size.sum(),
                device=x.device,
                dtype=x.dtype,
            )
            index = torch.arange(mask.shape[0], dtype=torch.long)
            for ilabel in range(self.output_size.shape[0]):
                multilabel_offset = self.output_size[:ilabel].sum()
                mask[index, y[0, :, ilabel] + multilabel_offset] = 1.0
            # similarity to each class
            similarity = torch.mm(similarity, mask)
            # Compatible with CrossEntropy
            similarity = torch.log(similarity)

        if self.clustered_similarity and id_index_reverse is not None:
            return similarity[id_index_reverse]
        return similarity


class GPVFEActivation(neural.LossModule):
    """Variational learning of inducing variables in sparse Gaussian processes."""

    eps: Final[float]

    # for type checkers
    k_m_n_k_n_m: torch.Tensor
    k_m_n_y: torch.Tensor
    k_m_m: torch.Tensor
    k_m_m_cholesky: torch.Tensor
    k_m_m_noise_inv_k_m_n_k_n_m_cholesky: torch.Tensor
    iterative_init: torch.Tensor
    init_position: torch.Tensor
    x_t_norm: torch.Tensor
    N_log_2_pi: torch.Tensor
    alpha: torch.Tensor
    length_scale: torch.nn.Parameter
    observation_noise: torch.nn.Parameter

    @beartype
    def __init__(
        self,
        *,
        inducing_points: int = 2000,
        embedding_size: int = 10,
        iterative_init: int = 100,
        similarity: Literal["rbf_similarity", "cosine_similarity"] = "rbf_similarity",
        eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        """Create a module representing a GP-VFE.

        Args:
            inducing_points: Number of inducing points.
            embedding_size: Size of the embedding, i.e., same size as the output of the
                MLP.
            iterative_init: If the inducing points should be randomly initialized from
                the import training set, this parameter is the total number of samples.
                Otherwise, it should be negative.
            similarity: Which similarity measure to use. Self-similarity has to be 1.0!
            eps: Added to the diagonal to ensure that a matrix is PSD.
            kwargs: Forwarded to neural.LossModule.
        """
        super().__init__(training_validation=True, **kwargs)

        self.register_buffer("iterative_init", torch.LongTensor([iterative_init]))
        self.register_buffer("init_position", torch.LongTensor([0]))
        self.similarity = getattr(neural, similarity)
        self.batch_norm = torch.nn.BatchNorm1d(embedding_size, affine=False)
        self.eps = eps

        # Parameters
        self.x_t = torch.nn.Parameter(
            (torch.rand(embedding_size, inducing_points) - 0.5) / sqrt(embedding_size),
        )
        self.length_scale = torch.nn.Parameter(torch.FloatTensor([0.0]))
        self.observation_noise = torch.nn.Parameter(torch.FloatTensor([0.0]))

        # Buffers
        self.register_buffer(
            "k_m_n_k_n_m",
            torch.zeros(inducing_points, inducing_points),
        )
        self.register_buffer("k_m_n_y", torch.zeros(inducing_points, 1))
        self.register_buffer("alpha", torch.zeros(0))
        self.register_buffer(
            "k_m_m_cholesky",
            -torch.ones(embedding_size, embedding_size),
        )
        self.register_buffer(
            "k_m_m_noise_inv_k_m_n_k_n_m_cholesky",
            -torch.ones(embedding_size, embedding_size),
        )
        self.register_buffer("k_m_m", -torch.ones(embedding_size, embedding_size))
        self.register_buffer("x_t_norm", -torch.ones(embedding_size, inducing_points))
        self.register_buffer("N_log_2_pi", torch.ones(1) * log(2 * pi) * iterative_init)

    def _reset(self) -> None:
        # clear all buffers
        self.k_m_n_k_n_m.fill_(0)
        self.k_m_n_y.fill_(0)
        self.alpha = torch.zeros(0)
        self.k_m_m_noise_inv_k_m_n_k_n_m_cholesky = -torch.ones_like(
            self.k_m_m_noise_inv_k_m_n_k_n_m_cholesky,
        )
        self.k_m_m = -torch.ones_like(self.k_m_m)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert y is not None
        x = self.batch_norm(x)
        meta["meta_embedding"] = x

        if self.training:
            self._reset()
        else:
            # end of initialization
            self.iterative_init.fill_(-1)

        # take random embedding from the batch (only during the first epoch)
        if self.iterative_init > 0 and self.init_position < self.x_t.shape[1]:
            init_position = self.iterative_init.cpu().item()
            size = min(
                max(
                    1,
                    int(
                        (float(x.shape[0]) / self.iterative_init.cpu().item())
                        * self.x_t.shape[1],
                    ),
                ),
                x.shape[0],
            )
            init_position = self.init_position.cpu().item()
            index = torch.randperm(x.shape[0], dtype=torch.int64)[:size]
            with torch.no_grad():
                self.x_t[:, init_position : init_position + size] = (
                    x[index, :].detach().t()
                )
            self.init_position += size

        noise = torch.exp(self.observation_noise)
        length_scale = torch.exp(self.length_scale)
        cache = {}
        if self.k_m_m[0, 0] != -1:
            cache["x_t_norm"] = self.x_t_norm
        k_x_m, cache = self.similarity(x, self.x_t, length_scale, cache)
        x_t_norm = cache["x_t_norm"]
        self.x_t_norm = x_t_norm.detach()

        k_m_m_cholesky = self.k_m_m_cholesky
        k_m_m = self.k_m_m
        if k_m_m[0, 0] == -1:
            # get k_m_m and L
            k_m_m = self.similarity(
                self.x_t.t(),
                self.x_t,
                length_scale,
                {"x_norm": x_t_norm.t(), "x_t_norm": x_t_norm},
            )[0]
            k_m_m = k_m_m + self.eps * torch.eye(
                k_m_m.shape[0],
                dtype=x.dtype,
                device=x.device,
            )
            k_m_m_cholesky = neural.cholesky(k_m_m)
            self.k_m_m = k_m_m.detach()
            self.k_m_m_cholesky = k_m_m_cholesky.detach()

        alpha = self.alpha
        k_m_m_noise_inv_k_m_n_k_n_m_cholesky = self.k_m_m_noise_inv_k_m_n_k_n_m_cholesky
        if k_m_m_noise_inv_k_m_n_k_n_m_cholesky[0, 0] == -1:
            # find variational mu
            if dataset == "training":
                k_m_n_k_n_m = k_x_m.t().mm(k_x_m)
                k_m_n_y = k_x_m.t().mm(y)
                if not self.training:
                    # accumulate
                    self.k_m_n_k_n_m += k_m_n_k_n_m.detach()
                    self.k_m_n_y += k_m_n_y.detach()
            else:
                # test time
                k_m_n_k_n_m = self.k_m_n_k_n_m
                k_m_n_y = self.k_m_n_y
            noise_inv = 1.0 / noise
            # eq 10
            k_m_m_noise_inv_k_m_n_k_n_m_cholesky = neural.cholesky(
                k_m_m + noise_inv * k_m_n_k_n_m,
            )
            self.k_m_m_noise_inv_k_m_n_k_n_m_cholesky = (
                k_m_m_noise_inv_k_m_n_k_n_m_cholesky.detach()
            )
            mu = noise_inv * k_m_m.mm(
                torch.cholesky_solve(k_m_n_y, k_m_m_noise_inv_k_m_n_k_n_m_cholesky),
            )

            # save alpha
            alpha = torch.cholesky_solve(mu, k_m_m_cholesky)
            self.alpha = alpha.detach()

        # find mean
        mean = k_x_m.mm(alpha)  # eq 6

        # find covariance
        meta["meta_loss_q"] = k_x_m.mm(torch.cholesky_solve(k_x_m.t(), k_m_m_cholesky))
        if dataset == "unknown":
            meta["meta_sigma_2"] = (
                1
                - meta["meta_loss_q"].diag()
                + k_x_m.mm(
                    torch.cholesky_solve(
                        k_x_m.t(),
                        k_m_m_noise_inv_k_m_n_k_n_m_cholesky,
                    ),
                ).diag()
            ).detach()

        # clear caches
        if self.training:
            self._reset()
        elif dataset == "training":
            self.k_m_m_noise_inv_k_m_n_k_n_m_cholesky[0, 0] = -1

        meta["meta_y_hat"] = mean
        meta["meta_loss_observation_noise"] = noise
        meta["meta_loss_N_log_2_pi"] = self.N_log_2_pi
        return mean, meta

    @torch.jit.export
    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = meta.pop("meta_loss_q")
        noise = meta.pop("meta_loss_observation_noise")
        q_noise = q + noise * torch.eye(q.shape[0], device=q.device)
        q_noise_cholesky = neural.cholesky(q_noise)

        elbow = meta.pop("meta_loss_N_log_2_pi")
        # complexity penalty
        elbow = elbow + neural.choleksy_sum_log_det(
            q_noise_cholesky.unsqueeze(0),
            n_chunks=1,
        )
        # data fit
        elbow = elbow + ground_truth.t().mm(
            torch.cholesky_solve(ground_truth, q_noise_cholesky),
        )
        # trace term
        elbow = elbow + (1 - q).trace() / noise

        return elbow * 0.5


class SelectiveTransferMachine(neural.LossModule):
    """Neural Selective Transfer Machine.

    "Selective transfer machine for personalized facial action unit detection"
    in the context of neural networks.

    Differences/notes:
    - s is determined using uLSIF [1] for each batch independently
    - the embedding from the last layer ("meta_embedding") is used for the similarity
    - for batches larger than 'sample_size', VFKMM [2] is used
    - learnable parameters of the kernel are leanred through backprop
    - lambda is annealed: important for un-initialized models

    [1] "A Least-squares Approach to Direct Importance Estimation"
    [2] "Efficient Sampling- Based Kernel Mean Matching"
    """

    embedding_name: Final[str]
    sample_size: Final[int]
    tolerance: Final[float]
    epochs_no_loss: Final[int]
    epoch_loss_ramp_up: Final[int]
    keep_existing_skew: Final[bool]
    loss_lambda: Final[float]

    cache: Dict[str, torch.Tensor]
    data: Dict[str, torch.Tensor]

    @beartype
    def __init__(
        self,
        *,
        model_fun: type[neural.LossModule] = neural.MLP,
        model_state: Optional[Path] = None,
        embedding_name: str = "meta_embedding",
        embedding_size: int,
        data: tuple[torch.Tensor, dict[str, torch.Tensor]],
        similarity: Literal[
            "rbf_similarity",
            "cosine_similarity",
            "frechet_similarity",
        ] = "rbf_similarity",
        cache_keys: tuple[str, ...] = ("x_t_norm",),
        clustered_similarity: bool = False,
        log_scale: float = 0.0,
        loss_lambda: float = 0.1,
        epochs_no_loss: int = 20,
        epoch_loss_ramp_up: int = 100,
        sample_size: int = 512,
        tolerance: float = 0.1,
        keep_existing_skew: bool = False,
        **kwargs: Any,
    ) -> None:
        """Neural implementation of Selective Transfer Machine.

        Args:
            model_fun: Function returning a model.
            model_state: Optionally, initialize model from pre-trained weights.
            embedding_name: Which embedding to use for the similarity.
            embedding_size: How large the embedding is.
            data: Known data for the test person.
            similarity: Which similarity function to use.
            cache_keys: Which fields of the similarity function should be cached.
            clustered_similarity: Whether it is a clustered similarity function.
            log_scale: Scale factor for the similarity function.
            loss_lambda: How much influence the loss term of STM has,
            epochs_no_loss: Ignore the loss term in STM for the first n epochs.
            epoch_loss_ramp_up: After ignoring the loss term, slowly increase the
                loss term for n epochs.
            sample_size: If batch size is smaller than this, us VF-KMM.
            tolerance: Tolerance to accept solutions for the relaxed problem.
            keep_existing_skew: Whether to re-skew the similarity weights to keep
                the existing skew.
            kwargs: Forwarded to LossModule.
        """
        model_kwargs = neural.pop_with_prefix(kwargs, "model_")
        model_kwargs["input_size"] = kwargs.pop("input_size")
        model_kwargs["output_size"] = kwargs.pop("output_size")
        super().__init__(**kwargs)

        self.data = data[1].copy()
        self.register_buffer("x", data[0].clone())
        self.embedding_name = embedding_name
        self.embedding: torch.Tensor = torch.zeros(0)
        self.sample_size = sample_size
        self.tolerance = tolerance
        self.epochs_no_loss = epochs_no_loss
        self.epoch_loss_ramp_up = epoch_loss_ramp_up
        self.keep_existing_skew = keep_existing_skew

        # init model
        self.model = model_fun(**model_kwargs, **kwargs)
        if model_state is not None:
            self.model = neural_wrapper.load_model_state(self.model, model_state)
        self.training_validation = self.model.training_validation

        # similarity
        if similarity == "frechet_similarity":
            clustered_similarity = True
            cache_keys = ("unique_ids_t", "means_t", "covs_t", "id_t_index")
        self.cache = {}
        self.length_scale = torch.nn.Parameter(torch.FloatTensor([log_scale]))
        self.similarity = getattr(neural, similarity)
        self.cache_keys = cache_keys
        self.clustered_similarity = clustered_similarity
        self.loss_lambda = loss_lambda

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_hat, meta = self.model(x, meta, y=y, dataset=dataset)
        embedding = meta[self.embedding_name]
        embedding_is_constant = (x.shape == embedding.shape) and (x == embedding).all()

        # similarity to test data
        if (self.training and not embedding_is_constant) or self.embedding.numel() == 0:
            with torch.no_grad():
                self.embedding = self.model(self.x, self.data, dataset="training")[1][
                    self.embedding_name
                ].t()
        similarity, cache = self.similarity(
            embedding.detach(),
            self.embedding.detach(),
            torch.exp(self.length_scale),
            self.cache,
            ids=meta["meta_id"],
            ids_t=self.data["meta_id"],
            same=False,
        )

        for key in self.cache_keys:
            self.cache[key] = cache[key].detach()

        self_similarity, cache = self.similarity(
            embedding.detach(),
            embedding.detach().t(),
            torch.exp(self.length_scale),
            {
                key.replace("_t", ""): cache[key.replace("_t", "")]
                for key in self.cache_keys
            },
            ids=meta["meta_id"],
            ids_t=meta["meta_id"],
            same=True,
        )
        if self.clustered_similarity:
            index = torch.nonzero(meta["meta_id"].view(-1, 1) == cache["unique_ids"])[
                :,
                1,
            ]
            similarity = similarity[index, :]
            self_similarity = self_similarity[torch.meshgrid(index, index)]

        # determine whether how strong the loss component of STM should be
        loss_lambda = self.loss_lambda
        if self.epoch < self.epochs_no_loss:
            loss_lambda = 0.0
        elif self.epoch < self.epochs_no_loss + self.epoch_loss_ramp_up:
            fraction = (
                float(self.epoch.cpu().item() - self.epochs_no_loss)
                / self.epoch_loss_ramp_up
            )
            loss_lambda *= fraction

        # solve relaxed problem (can be negative) and then clamp&scale
        loss = torch.zeros(1, dtype=x.dtype, device=x.device)
        if self.training:
            assert y is not None
            with torch.no_grad():
                loss = self.loss(y_hat, y, meta, take_mean=False, loss=None)
        if not self.training and dataset != "unknown":
            # need weighting only during training and for post-hoc inspection
            weights = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        elif self.sample_size >= x.shape[0]:
            weights = torch.cholesky_solve(
                (similarity.mean(dim=1, keepdim=True) - loss_lambda * loss)
                * x.shape[0],
                torch.linalg.cholesky(
                    self_similarity.double()
                    + 1e-3
                    * torch.eye(
                        self_similarity.shape[0],
                        device=x.device,
                        dtype=torch.double,
                    ),
                ).float(),
            )
        else:
            samplings = int(
                max(
                    round(
                        log(self.tolerance)
                        / (self.sample_size * log(1 - 1.0 / x.shape[0])),
                    ),
                    ceil(x.shape[0] / self.sample_size),
                ),
            )
            samples = torch.randint(
                0,
                x.shape[0],
                (samplings, self.sample_size),
                device=x.device,
                dtype=torch.int64,
            )
            self_similarities_cholesky = torch.linalg.cholesky(
                self_similarity[
                    samples.unsqueeze(2).expand(
                        samplings,
                        self.sample_size,
                        self.sample_size,
                    ),
                    samples.unsqueeze(1).expand(
                        samplings,
                        self.sample_size,
                        self.sample_size,
                    ),
                ].double()
                + 1e-3
                * torch.eye(self.sample_size, device=x.device, dtype=torch.double),
            ).float()
            linear_term = (
                similarity.mean(dim=1, keepdim=True) - loss_lambda * loss
            ) * self.sample_size
            weights = (
                torch.cholesky_solve(
                    linear_term[samples.view(-1)].view(-1, self.sample_size, 1),
                    self_similarities_cholesky,
                )
                .clamp(min=1e-8)
                .view(-1)
            )
            # re-order and average
            counts = torch.bincount(samples.view(-1), minlength=x.shape[0])
            weights = torch.sparse_coo_tensor(
                torch.vstack(
                    [
                        samples.view(-1),
                        torch.repeat_interleave(
                            torch.arange(samplings, device=x.device),
                            self.sample_size,
                        ),
                    ],
                ),
                weights,
                size=(x.shape[0], samplings),
            )
            weights = torch.sparse.mm(
                weights.coalesce(),
                torch.ones(samplings, 1, device=x.device, dtype=x.dtype),
            ) / counts.view(-1, 1)
            # assume that similar examples have a similar weighting
            zero = counts == 0
            non_zero = ~zero
            with torch.no_grad():
                non_self_similarity = self_similarity[zero][:, non_zero]
                non_weighting = non_self_similarity.mm(
                    weights[non_zero],
                ) / non_self_similarity.sum(dim=1, keepdim=True)
            weights = weights.index_put(indices=(zero,), values=non_weighting)
        weights = weights.clamp(min=1e-8)
        weights = (weights / weights.mean()).clamp(min=0.1, max=10.0)

        # keep the same class skew as before
        if self.keep_existing_skew and self.training:
            # only for 1D discrete targets (nominal&ordinal)
            assert y is not None
            y_flat = y.view(-1)
            if y_flat.dtype != torch.long:
                y_flat = (y_flat * 10).round()
            re_weighting = torch.empty_like(weights).detach()
            for label in torch.unique(y_flat):
                index = y_flat == label
                re_weighting[index] = (
                    index.to(dtype=(weights.dtype)).mean() / weights[index].mean()
                )
            weights = weights * re_weighting
            weights = (weights / weights.mean()).clamp(min=0.1, max=10.0)

        meta["meta_sample_weight"] = weights / weights.mean()
        return y_hat, meta
