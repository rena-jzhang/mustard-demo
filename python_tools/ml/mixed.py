#!/usr/bin/env python3
"""PyTorch implementations of mixed effect models."""
from collections import defaultdict
from collections.abc import Generator
from math import log, pi
from typing import Any, Dict, Final, Optional

import torch
from beartype import beartype

from python_tools import generic
from python_tools.ml import neural


class LinearMixedEffects(neural.LossModule):
    """An activation function that learns a random effects.

    Note:
        This layer is meant to be used  after a linear layer that represents
        the fixed effects.

    Same as
    - "Neural networks for longitudinal studies in Alzheimer's disease"
    - "Mixed effects neural networks (MeNets) with applications to gaze estimation"

    """

    truncate: Final[int]
    only_bias: Final[bool]
    reml: Final[bool]
    device: Final[str]
    add_bias: Final[bool]
    log_2_pi: Final[float]

    # buffers
    random_effect_ids: torch.Tensor
    random_effects: torch.Tensor
    fixed_effects: torch.Tensor
    sigma_2: torch.Tensor
    d_sigma: torch.Tensor

    @beartype
    def __init__(
        self,
        *,
        embedding_size: int = -1,
        output_size: int = 1,
        truncate: int = 4096,
        add_bias: bool = True,
        only_bias: bool = False,
        number_of_cluster: int = -1,
        dtype: torch.dtype = torch.float64,
        reml: bool = False,
        device: str = "cpu",
        iterations: int = 0,
        **kwargs: Any,
    ) -> None:
        """Linear mixed effect model.

        Args:
            embedding_size: Size of the random-effect input.
            output_size: Number of independent LMEs.
            truncate: Maximal number of observations per cluster (larger clusters
                are randomly truncated). Use a negative number to not truncate clusters.
            add_bias: Adds a bias column to the random features.
            only_bias: Only estimate a random bias term.
            number_of_cluster: How many cluster there are.
            dtype: Which dtype to use when calculating the Cholesky decomposition.
            reml: Whether to optimize for the restricted maximum likelihood.
            device: Which device to use for all internal calculations.
            iterations: Unused parameter. Present for compatibility with DeepModel.
            kwargs: Keywords forwarded to neural.LossModule.

        Note:
            The random covariates are normalized with a BatchNorm1d.
        """
        super().__init__(**kwargs)
        if only_bias:
            add_bias = True
            embedding_size = 0
        self.truncate = truncate
        self.only_bias = only_bias
        self.reml = reml
        self.device = device

        # parameters
        self.norm = torch.nn.BatchNorm1d(embedding_size, affine=False)

        self.add_bias = add_bias
        if self.add_bias:
            embedding_size = embedding_size + 1
        self.register_buffer(
            "d_sigma",
            torch.eye(embedding_size).repeat(output_size, 1, 1),
        )
        self.register_buffer("sigma_2", -torch.ones(output_size))

        # keep random effects
        assert number_of_cluster > 0
        self.register_buffer(
            "random_effects",
            torch.zeros(output_size, number_of_cluster, embedding_size),
        )
        self.register_buffer("random_effect_ids", -torch.ones(number_of_cluster))

        # decomposition
        self.dtype = dtype

        self.log_2_pi = log(2 * pi)

    @staticmethod
    def _get_clusters(meta_id: torch.Tensor) -> dict[int, torch.Tensor]:
        full_clusters: dict[int, list[torch.Tensor]] = {}
        for index in torch.unique(meta_id).view(-1, 1) == meta_id.view(1, -1):
            index = torch.nonzero(index)
            key = int(index.shape[0])
            if key not in full_clusters:
                full_clusters[key] = []
            full_clusters[key].append(index)
        results: dict[int, torch.Tensor] = {}
        for key, value in full_clusters.items():
            results[int(key)] = torch.squeeze(torch.stack(value), dim=-1)
        return results

    def estimate_random_effects(
        self,
        y: torch.Tensor,
        ids: torch.Tensor,
        random: torch.Tensor,
    ) -> None:
        random = self._preprocess(
            random.to(device=self.norm.running_var.device, non_blocking=True),
        )

        # remove observations from too large clusters
        full_clusters: Optional[Dict[int, torch.Tensor]] = None
        if self.truncate > 1:
            clusters = self._get_clusters(ids)
            keep = torch.ones(y.shape[0], dtype=torch.bool, device=self.device)
            for observations, groups in clusters.items():
                if self.truncate >= observations:
                    continue
                index = torch.randperm(observations)[: observations - self.truncate]
                keep[groups[:, index].view(-1)] = False
            if not keep.all():
                y = y[keep]
                ids = ids[keep]
                random = random[keep]
                full_clusters = None
            else:
                full_clusters = clusters

        # take 1 M step of EM
        self._em(
            y.to(device=self.device, non_blocking=True),
            ids.to(device=self.device, non_blocking=True),
            random.to(device=self.device, non_blocking=True),
            full_clusters=full_clusters,
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # norm
        if not self.only_bias:
            x = self.norm(x)

        # add bias
        if self.add_bias:
            x = torch.cat(
                [x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)],
                dim=1,
            )
        if self.only_bias:
            return x[:, -1:]
        return x

    def forward(
        self,
        y_fixed: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert y is not None
        meta["meta_y_hat_fixed"] = y_fixed
        x = self._preprocess(meta["meta_embedding"])

        # get predictions
        if "meta_random_effects" in meta:
            # use provided random effects
            random_effects = (
                meta["meta_random_effects"]
                .view(x.shape[0], x.shape[1], y_fixed.shape[1])
                .to(device=x.device, dtype=x.dtype, non_blocking=True)
            )
        elif "meta_cluster_weights" in meta:
            # use provided weighting
            weights = (
                meta["meta_cluster_weights"]
                .view(x.shape[0], -1, y_fixed.shape[1])
                .to(device=x.device, dtype=x.dtype, non_blocking=True)
            )
            assert (weights >= 0).all()
            weights = weights / weights.sum(dim=-1, keepdim=True)
            random_effects = (
                weights.permute(2, 0, 1).bmm(self.random_effects).permute(1, 2, 0)
            )
        else:
            # use random effects for known subjects
            random_effects = torch.zeros(
                x.shape[0],
                x.shape[1],
                y_fixed.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
            matches = torch.nonzero(self.random_effect_ids.cpu() == meta["meta_id"])
            random_effects[matches[:, 0]] = self.random_effects[
                :,
                matches[:, 1],
            ].permute(1, 2, 0)

        meta["meta_embedding_random_effects"] = random_effects.view(x.shape[0], -1)
        y_hats = x.unsqueeze(1).bmm(random_effects).squeeze(1) + y_fixed

        meta["meta_y_hat"] = y_hats
        return y_hats, meta

    @torch.jit.export
    def add_new_random_effects(
        self,
        x: torch.Tensor,
        y_y_fixed: torch.Tensor,
        ids: torch.Tensor,
    ) -> None:
        """Add random effects of new people to the model.

        Does not adjust the covariance matrix and error variance.
        """
        assert not (ids.view(-1, 1) == self.random_effect_ids.view(1, -1)).any()

        # save internal state
        saved_state = {
            "d_sigma": self.d_sigma.clone(),
            "sigma_2": self.sigma_2.clone(),
            "random_effects": self.random_effects.clone(),
            "random_effect_ids": self.random_effect_ids.clone(),
        }

        # run EM with only the new data
        self.estimate_random_effects(y_y_fixed, ids, x)

        # combine states
        self.d_sigma[:] = saved_state["d_sigma"]
        self.sigma_2[:] = saved_state["sigma_2"]
        self.random_effects = torch.cat(
            [saved_state["random_effects"], self.random_effects],
            dim=1,
        )
        self.random_effect_ids = torch.cat(
            [saved_state["random_effect_ids"], self.random_effect_ids],
            dim=0,
        )

    def _em(
        self,
        residual: torch.Tensor,
        ids: torch.Tensor,
        random: torch.Tensor,
        full_clusters: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        device = self.sigma_2.device
        lmes = residual.shape[1]
        random_effects = torch.empty(
            lmes,
            random.shape[0],
            random.shape[1],
            device=random.device,
            dtype=random.dtype,
        )

        # initialize sigma_2 to MSE and the covariance of random effects to the identity
        if (self.sigma_2 == -1).all():
            self.sigma_2 = (
                residual.detach()
                .to(dtype=self.dtype)
                .pow(2)
                .mean(dim=0)
                .to(dtype=random.dtype, device=device, non_blocking=True)
            )
            self.d_sigma /= self.sigma_2.view(-1, 1, 1)
        d_sigma = self.d_sigma.to(device=self.device, non_blocking=True)
        sigma_2 = self.sigma_2.to(device=self.device).clamp(min=1e-5)

        # extract clusters
        if full_clusters is None:
            full_clusters = self._get_clusters(ids)

        # accumulate sums
        sigma_2_sum = torch.zeros_like(sigma_2)
        d_sigma_sum = torch.zeros_like(d_sigma)
        sum_x_vinv_x: Optional[torch.Tensor] = None
        if self.reml:
            sum_x_vinv_x = torch.zeros(
                (lmes, random.shape[1], random.shape[1]),
                device=self.device,
                dtype=random.dtype,
            )

        # process clusters
        cluster_count: int = 0
        future: Optional[torch.jit.Future[torch.Tensor]] = None
        vs_cholesky: list[torch.Tensor] = []
        for cluster_size, clusters in full_clusters.items():
            cluster_count += clusters.shape[0]
            current_random = random[clusters].repeat(lmes, 1, 1)
            current_d_sigma = torch.repeat_interleave(d_sigma, clusters.shape[0], dim=0)

            # calculate Cholesky of v
            d_sigma_current_random = current_d_sigma.bmm(current_random.transpose(1, 2))
            v = current_random.bmm(d_sigma_current_random)
            v.diagonal(dim1=-2, dim2=-1)[:] += 1
            v_cholesky = neural.cholesky(v, dtype=self.dtype)
            if self.reml:
                vs_cholesky.append(v_cholesky)

            # for restricted ML
            if self.reml:
                future = torch.jit.fork(
                    _lme_x_vinv_x,
                    current_random,
                    lmes,
                    residual[clusters],
                    v_cholesky,
                )

            # infer random effects
            current_residual = torch.cat(torch.chunk(residual[clusters], lmes, dim=-1))
            v_inv_current_residual = torch.cholesky_solve(current_residual, v_cholesky)
            current_random_effects = d_sigma_current_random.bmm(
                v_inv_current_residual,
            ).squeeze(2)

            future_random_effects = torch.jit.fork(
                _lme_random_effects,
                current_random_effects,
                lmes,
                cluster_size,
            )

            # for d_sigma
            future_d_sigma = torch.jit.fork(
                _lme_d_sigma,
                sigma_2,
                current_d_sigma,
                current_random,
                lmes,
                current_random_effects,
                v_cholesky,
            )

            # for sigma_2
            sigma_2_sum += torch.pow(
                current_residual
                - current_random.bmm(
                    current_random_effects.view(current_random_effects.shape[0], -1, 1),
                ),
                2,
            ).view(lmes, -1).sum(dim=1) - sigma_2 * torch.cholesky_inverse(
                v_cholesky,
            ).diagonal(
                dim1=-2,
                dim2=-1,
            ).reshape(
                lmes,
                -1,
            ).sum(
                dim=1,
            )

            # wait for async
            if future is not None and sum_x_vinv_x is not None:
                sum_x_vinv_x += torch.jit.wait(future)

            random_effects[:, clusters.view(-1), :] = torch.jit.wait(
                future_random_effects,
            )

            d_sigma_sum += torch.jit.wait(future_d_sigma)

        sum_x_vinv_x_cholesky = torch.zeros(0)
        if self.reml and sum_x_vinv_x is not None:
            sum_x_vinv_x_cholesky = neural.cholesky(sum_x_vinv_x, dtype=self.dtype)

            for v_cholesky, clusters in zip(vs_cholesky, full_clusters.values()):
                current_random = random[clusters].repeat(lmes, 1, 1)
                current_d_sigma = torch.repeat_interleave(
                    d_sigma,
                    clusters.shape[0],
                    dim=0,
                )

                # for d_sigma
                v_inv_x = torch.cholesky_solve(current_random, v_cholesky)
                current_sum_x_vinv_x_cholesky = torch.repeat_interleave(
                    sum_x_vinv_x_cholesky,
                    clusters.shape[0],
                    dim=0,
                )
                sum_x_vinv_x_inv_x = torch.cholesky_solve(
                    current_random.transpose(1, 2),
                    current_sum_x_vinv_x_cholesky,
                )
                v_inv_x_sum_x_vinv_x_inv_x = v_inv_x.bmm(sum_x_vinv_x_inv_x)
                d_sigma_sum += (
                    current_d_sigma.bmm(current_random.transpose(1, 2))
                    .bmm(v_inv_x_sum_x_vinv_x_inv_x)
                    .bmm(v_inv_x)
                    .bmm(current_d_sigma)
                    .view(lmes, -1, random.shape[1], random.shape[1])
                    .sum(dim=1)
                )

                # for sigma^2
                sigma_2_sum += sigma_2 * v_inv_x_sum_x_vinv_x_inv_x.bmm(
                    torch.cholesky_inverse(v_cholesky),
                ).diagonal(dim1=-2, dim2=-1).reshape(lmes, -1).sum(dim=1)

        # M-step
        self.sigma_2 += sigma_2_sum.to(device=device) / random.shape[0]
        self.sigma_2.clamp_(min=1e-5)
        self.d_sigma += d_sigma_sum.to(device=device) / cluster_count

        # fix symptom, not the cause :(
        degenerated_vars = self.d_sigma.diagonal(dim1=1, dim2=2) < 0
        if degenerated_vars.any():
            for ilme, degenerated_var in enumerate(degenerated_vars):
                if not degenerated_var.any():
                    continue
                self.d_sigma[ilme, degenerated_var | degenerated_var.view(-1, 1)] = 0
                self.d_sigma[ilme].diagonal()[degenerated_var] = 1e-5

        # to save random effects
        ids, index = neural.unique_with_index(ids)
        self.random_effects = random_effects[:, index].to(
            device=device,
            non_blocking=True,
        )
        self.random_effect_ids = ids.to(device=device, non_blocking=True)


def _lme_d_sigma(
    sigma_2: torch.Tensor,
    current_d_sigma: torch.Tensor,
    current_random: torch.Tensor,
    lmes: int,
    current_random_effects: torch.Tensor,
    v_cholesky: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            chunk2.t().mm(chunk2) / sigma_2[ichunk] - chunk1.sum(dim=0)
            for ichunk, (chunk1, chunk2) in enumerate(
                zip(
                    torch.chunk(
                        current_d_sigma.bmm(current_random.transpose(1, 2))
                        .bmm(torch.cholesky_solve(current_random, v_cholesky))
                        .bmm(current_d_sigma),
                        lmes,
                    ),
                    torch.chunk(current_random_effects, lmes),
                ),
            )
        ],
    )


def _lme_random_effects(
    current_random_effects: torch.Tensor,
    lmes: int,
    cluster_size: int,
) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.stack(torch.chunk(current_random_effects, lmes)),
        cluster_size,
        dim=1,
    )


def _lme_x_vinv_x(
    current_random: torch.Tensor,
    lmes: int,
    y: torch.Tensor,
    v_cholesky: torch.Tensor,
) -> torch.Tensor:
    return (
        current_random.transpose(1, 2)
        .bmm(torch.cholesky_solve(current_random, v_cholesky))
        .view(lmes, -1, current_random.shape[-1], current_random.shape[-1])
        .sum(dim=1)
    )


class NeuralMixedEffects(neural.LossModule):
    """Fit non-linear mixed effect models."""

    simulated_annealing_alpha: Final[float]
    dtype: Final[torch.dtype]
    l2_lambda: Final[float]

    p_eta: torch.Tensor
    index_diagonal: Final[torch.Tensor]
    index_full: Final[torch.Tensor]

    @beartype
    def __init__(
        self,
        *,
        clusters: torch.Tensor,
        model_fun: type[neural.LossModule] = neural.MLP,
        fixed_model_fun: type[neural.LossModule] | None = None,
        random_effects: tuple[str, ...] = (),
        random_buffers: tuple[str, ...] = (),
        simulated_annealing_alpha: float = 0.97,
        cluster_count: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        independent: tuple[str, ...] = (),
        l2_lambda: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Instantiate a non-linear mixed effects model.

        Note:
            This implementation follows the optimization procedures of saemix. Instead
            of sampling, this implementation uses gradient descent. After each
            epoch, the sampling statistics (covariance and residuals) are updated.

            The loss function defined by the underlying model (created by model_fun) is
            assumed to be the log of a unit deviance from a distribution in the
            exponential family.

            NME allows specifying two models: one with only fixed effects (optional;
            fixed_model_fun) and one with mixed effects (required; model_fun). If a
            fixed-only model is specified, its output will be provided as the input to
            the mixed model. This can improve performance as the fixed-only model is
            called only one time instead of for each person.

        Args:
            clusters: Array containing the cluster IDs, e.g., ID of people.
            model_fun: Function to instantiate a new model. This function is called for
                each cluster. Non-random parameters and buffers are shared.
            fixed_model_fun: Function containing only fixed effects. The output is the
                input to the mixed model from model_fun.
            cluster_count: How many observations are there for each cluster (same order
                as clusters!)
            random_effects:
                List of parameters that are random effects, e.g., layers.0.bias.
            random_buffers: List of buffers which should not be shared.
            simulated_annealing_alpha: To prevent the variance from becoming zero to
                quickly. When using higher values, such as 0.97, early stopping might
                pick a model that relies too heavily on random effects.
            dtype: Which dtype to use when calculating the Cholesky decomposition of the
                covariance matrix Sigma.
            independent: List of parameters that are assumed to be independent. The
                selected parameters have only a diagonal Sigma component (avoids matrix
                inversion for the provided parameters).
            l2_lambda: If the data is correlated (not IID), this can be used to scale
                the regularization term to keep the random effect smaller.
            **kwargs: Keywords starting with 'model_' are forwarded to model_fun (same
                for 'fixed_model_').
        """
        # separate keywords for the mixed/fixed models
        model_kwargs = neural.pop_with_prefix(kwargs, "model_")
        fixed_model_kwargs = neural.pop_with_prefix(kwargs, "fixed_model_")
        model_kwargs.setdefault("output_size", kwargs.pop("output_size"))
        fixed_model_kwargs.setdefault("input_size", kwargs.pop("input_size"))
        if fixed_model_fun is None:
            model_kwargs.setdefault("input_size", fixed_model_kwargs["input_size"])
        else:
            fixed_size = fixed_model_kwargs.get(
                "output_size",
                model_kwargs.get("input_size", -1),
            )
            fixed_model_kwargs.setdefault("output_size", fixed_size)
            model_kwargs.setdefault("input_size", fixed_size)

        # replace "." with "-"
        random_effects = tuple(name.replace(".", "-") for name in random_effects)
        independent = tuple(name.replace(".", "-") for name in independent)

        super().__init__(**kwargs)

        self.jit_me = False
        self.l2_lambda = l2_lambda
        self.simulated_annealing_alpha = simulated_annealing_alpha
        self.dtype = dtype
        self.register_buffer("cluster_count", cluster_count.view(-1))
        self.during_training = False
        self.losses: list[torch.Tensor] = []

        # instantiate purely fixed model
        self.fixed_model = None
        if fixed_model_fun is not None:
            self.fixed_model = fixed_model_fun(**fixed_model_kwargs)

        # instantiate N models (mixed models)
        assert clusters.numel() == cluster_count.numel()
        self.cluster_names = clusters.view(-1)
        self.models = neural.Ensemble(
            model=model_fun,
            size=clusters.numel(),
            **kwargs,
            **{
                (key if key in ("input_size", "output_size") else f"model_{key}"): value
                for key, value in model_kwargs.items()
            },
        )

        # keep reference of mixed parameters and share fixed parameters
        self.mixed_parameters = defaultdict(list)
        shared_parameters = {}
        shared_buffers = {}
        for model in self.models.models:
            for name, parameter in sorted(model.named_parameters()):
                name = name.replace(".", "-")
                # save references to all random parameters
                if name in random_effects:
                    self.mixed_parameters[name].append(parameter)
                    continue

                # not yet encountered shared parameters
                if name not in shared_parameters:
                    shared_parameters[name] = parameter
                    continue

                # share parameters
                parent = generic.get_object(model, name.split("-")[:-1])
                setattr(
                    parent,
                    name.split("-")[-1],
                    shared_parameters[name],
                )

            # share buffer
            for name, buffer in model.named_buffers():
                if name in random_buffers:
                    continue

                # not yet encountered buffer
                if name not in shared_buffers:
                    shared_buffers[name] = buffer
                    continue

                # share buffer
                parent = generic.get_object(model, name.split(".")[:-1])
                setattr(parent, name.split(".")[-1], shared_buffers[name])

        # separate mixed parameters into fixed and random
        # this is needed because a large L2 norm on "just" the random effects does not
        # work well with gradient decent (it also prevents/slows down the fixed effects)
        self.random = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(
                    torch.zeros_like(torch.stack(values, dim=0)),
                )
                for key, values in self.mixed_parameters.items()
            },
        )
        self.fixed = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(values[0].clone())
                for key, values in self.mixed_parameters.items()
            },
        )

        # get size of random effects
        number_random_effects = 0
        parameter_index = {}
        for name, parameter in self.fixed.items():
            size = parameter.numel()
            parameter_index[name] = list(
                range(number_random_effects, number_random_effects + size),
            )
            number_random_effects += size

        # inverting covariance structure (full or independent)
        full = tuple(sorted(set(self.mixed_parameters).difference(independent)))
        independent = tuple(
            set(independent).intersection(self.mixed_parameters),
        )

        independent_indices = generic.flatten_nested_list(
            [parameter_index[x] for x in independent],
        )
        full_indices = generic.flatten_nested_list([parameter_index[x] for x in full])
        self.register_buffer("sigma_diagonal", torch.ones(len(independent_indices)))
        self.register_buffer("sigma_full", torch.eye(len(full_indices)))
        self.index_diagonal = torch.LongTensor(independent_indices)
        self.index_full = torch.LongTensor(full_indices)

        # variables for 'SAEM' (for M)
        self.register_buffer("sigma_2", torch.ones(1) / 10.0)
        self.register_buffer("sa_sigma", torch.ones(number_random_effects))
        self.register_buffer("p_eta", torch.ones(len(self.models.models)))

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Generator[None, None, tuple[str, torch.nn.Parameter]]:
        """Remove duplicate parameters."""
        for name, parameter in super().named_parameters(prefix=prefix, recurse=recurse):
            if (
                name.startswith("models.models.")
                and name.split(".", 3)[-1].replace(".", "-") in self.random
            ):
                continue
            yield name, parameter

    def can_jit(self) -> bool:
        # to first move all buffers to the device and then partially jit
        self.models = torch.jit.script(self.models)
        if self.fixed_model is not None and self.fixed_model.can_jit():
            self.fixed_model = torch.jit.script(self.fixed_model)
        return False

    def _update_sigma_2_sigma(self) -> None:
        with torch.no_grad():
            # update sigma
            variances, sigmas = neural.cov_block(
                self._get_eta(),
                self.index_diagonal,
                [self.index_full],
            )

            # independent: annealing + invert
            if self.index_diagonal.shape[0] > 0:
                self.sa_sigma[self.index_diagonal] = torch.max(
                    self.sa_sigma[self.index_diagonal] * self.simulated_annealing_alpha,
                    variances,
                )
                self.sigma_diagonal.copy_(
                    1.0 / self.sa_sigma[self.index_diagonal].clamp(min=1e-8),
                )

            # full: annealing + cholesky
            if self.index_full.shape[0] > 0:
                sigma = sigmas[0]
                assert sigma.shape[0] == self.sigma_full.shape[0]
                # annealing
                self.sa_sigma[self.index_full] = torch.max(
                    self.sa_sigma[self.index_full] * self.simulated_annealing_alpha,
                    sigma.diagonal(),
                )
                sigma.diagonal().copy_(self.sa_sigma[self.index_full])
                # cholesky
                self.sigma_full.copy_(neural.cholesky(sigma, dtype=self.dtype))
                assert (self.sigma_full.diag() >= -1e-6).all()
                self.sigma_full.diagonal().clamp_(min=0.0)

            # update sigma_2: no annealing (does not collapse)
            self.sigma_2.copy_(torch.stack(self.losses).sum())
            self.losses = []

    def _get_eta(self) -> torch.Tensor:
        # note: eta can contain fixed parts (eta.mean(dim=0, keepdim=True))
        return torch.cat(
            [random.view(random.shape[0], -1) for random in self.random.values()],
            dim=1,
        )

    def _after_training(self) -> None:
        """Update sigma and sigma_2."""
        if not (not self.training and self.during_training):
            return
        self.during_training = False
        with torch.no_grad():
            self._update_sigma_2_sigma()

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.training:
            self.during_training = True

        # update sigma_2 and sigma
        self._after_training()

        # fixed-only model
        if self.fixed_model is not None:
            x, meta = self.fixed_model(x, meta, y=y, dataset=dataset)

        # use model for known clusters and average for unknown
        if "meta_cluster_weights" in meta:
            weighting = meta["meta_cluster_weights"]
        else:
            weighting = (
                meta["meta_id"].view(-1, 1) == self.cluster_names.view(1, -1)
            ).to(
                dtype=x.dtype,
                device=x.device,
            )
        unknown = weighting.sum(dim=1) == 0
        if unknown.any():
            weighting[unknown] = torch.exp(-0.5 * self.p_eta)

        # set fixed+random effects
        for key, targets in self.mixed_parameters.items():
            mixeds = self.fixed[key] + self.random[key]
            for target, mixed in zip(targets, mixeds):
                target.detach_()
                target.copy_(mixed)
                target.requires_grad_(True)

        return self.models(x, dataset=dataset, meta=meta, y=y, weights=weighting)

    def _get_p_eta(self, *, index: torch.Tensor) -> torch.Tensor:
        eta = self._get_eta()[index]
        eta = eta - eta.mean(dim=0, keepdims=True)
        p_eta = torch.zeros(eta.shape[0], device=eta.device, dtype=eta.dtype)

        # Mixed Effects
        eta_index = eta[:, self.index_diagonal]
        p_eta = (eta_index.pow(2) * self.sigma_diagonal.view(1, -1)).sum(dim=1)

        eta_index = eta[:, self.index_full]
        return p_eta + (
            eta_index.view(eta.shape[0], 1, -1)
            .bmm(
                torch.cholesky_solve(
                    eta_index.view(eta.shape[0], -1, 1),
                    self.sigma_full,
                ),
            )
            .view(-1)
        ).clamp(min=0.0)

    @torch.jit.export
    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # p(y | eta, Theta)
        loss = 0.0
        indices = meta["meta_id"].view(1, -1) == self.cluster_names.view(-1, 1)
        # unknown clusters are processed by the last model
        indices[-1] = ~indices[:-1].any(dim=0)
        counts = indices.sum(dim=1)
        for index, model, count in zip(indices, self.models.models, counts):
            if count == 0:
                continue
            loss = (
                loss
                + model.loss(
                    scores[index, :],
                    ground_truth[index, :],
                    {key: value[index, :] for key, value in meta.items()},
                )
                * count
            )
        if self.training:
            self.losses.append(loss.detach() / self.cluster_count.sum())
        loss = loss / scores.shape[0] / self.sigma_2

        # p(eta, Theta)
        ratios = counts.to(device=scores.device, dtype=scores.dtype) / (
            self.cluster_count * scores.shape[0]
        )
        if self.training:
            cluster_index = ratios != 0
            self.p_eta = torch.index_put(
                self.p_eta,
                (cluster_index,),
                self._get_p_eta(index=cluster_index),
            )
        loss = loss + (self.p_eta * ratios).sum() * self.l2_lambda
        self.p_eta.detach_()

        return loss
