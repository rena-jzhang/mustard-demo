#!/usr/bin/env python3
"""PyTorch implementations of neural models."""
import inspect
from collections.abc import Callable
from typing import Any, Final, Literal, Optional, Union

import torch
from beartype import beartype

from python_tools.ml.pytorch_tools import packedsequence_to_list
from python_tools.typing import LossModule as LossModuleABC


def index_dict(
    meta: dict[str, torch.Tensor],
    index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Index dict with either GPU/CPU index.

    Args:
        meta: The dictionariy with GPU/CPU tensors.
        index: The GPU index.

    Returns:
        The indexed dictionary.
    """
    index_cpu = index.cpu()
    result = {}
    for key, value in meta.items():
        if value.device == index.device:
            result[key] = value[index]
        else:
            result[key] = value[index_cpu]
    return result


@torch.jit.script
def neg_log_likelihood(
    scores: torch.Tensor,
    ground_truth: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    loss = -torch.log(torch.gather(scores, 1, ground_truth.view(-1, 1)).clamp(min=eps))
    if class_weights is not None:
        loss = loss * torch.gather(
            class_weights.view(-1, 1),
            0,
            ground_truth.view(-1, 1),
        )
    return loss.view(-1)


def torch_identity(x: torch.Tensor) -> torch.Tensor:
    return x


@torch.jit.script
def unique_with_index(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """np.unique(tensor, return_index=True) for pytorch.

    github.com/pytorch/pytorch/issues/36748#issuecomment-619514810
    """
    tensor = tensor.view(-1)
    unique, inverse = torch.unique(tensor, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


@torch.jit.script
def cov_block(
    data: torch.Tensor,
    diagonal: torch.Tensor,
    blocks: list[torch.Tensor],
    unbiased: int = 1,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Calculate the block-diagonal (co)-variance.

    data        The data (observations x features)
    diagonal    Indicates which features are independent of all other features.
    block       List of block-wise correlated features.

    Returns:
        A tuple of size two:
        0) The variance of independent features;
        1) List covariance matrices (in the same order as blocks)
    """
    variance = data[:, diagonal].var(dim=0, unbiased=unbiased == 1)
    covariances = [
        torch.cov(data[:, block].t(), correction=unbiased) for block in blocks
    ]

    return variance, covariances


@torch.jit.script
def cholesky(x_bii: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Cholesky decomposition with additive noise on the diagonal."""
    device = x_bii.device
    dtype_ = x_bii.dtype
    size = x_bii.size()
    # convert to double
    x_bii = x_bii.to(dtype=dtype, device="cpu")
    if x_bii.ndim == 2:
        x_bii = x_bii.unsqueeze(0)
    x_bii = (x_bii + x_bii.transpose(1, 2)) / 2

    eye = torch.eye(x_bii.shape[-1], dtype=dtype, device="cpu")
    choleskys = [eye for _ in range(x_bii.shape[0])]
    indices = torch.arange(x_bii.shape[0], dtype=torch.long)

    # add more and more regularization, if needed
    for i in range(10):
        cholesky, info = torch.linalg.cholesky_ex(
            x_bii[indices] + (10 ** (i - 10)) * eye.unsqueeze(0),
        )
        success: list[int] = indices[info == 0].tolist()
        for index, cholesky_i in zip(
            success,
            cholesky[info == 0].to(dtype=dtype, device="cpu"),
        ):
            choleskys[index] = cholesky_i
        indices = indices[info != 0]
        if (info == 0).all():
            break
    if indices.shape[0] > 0:
        eye = eye.to(dtype=dtype, device="cpu")
        for index in indices:
            choleskys[index] = eye
    return (
        torch.stack(choleskys)
        .view(size)
        .to(dtype=dtype_, device=device, non_blocking=True)
    )


def choleksy_sum_log_det(choleksy: torch.Tensor, n_chunks: int = 1) -> torch.Tensor:
    return (
        choleksy.diagonal(dim1=1, dim2=2)
        .log()
        .view(n_chunks, -1, choleksy.shape[-1])
        .sum(dim=(1, 2))
        * 2
    )


@torch.jit.script
def interaction_terms(xs: list[torch.Tensor], append_one: bool = True) -> torch.Tensor:
    """Calculate multiplicative interaction terms all between modalities.

    Note:
        Supports only lists of size 1, 2, and 3.

    Args:
        xs: List of matrices. The first dimension is the batch dimension.
        append_one: Whether to append 1 to keep uni/bi-modal interactions.

    Returns:
        The flattened prodcut of all modalities.
    """
    if len(xs) == 1:
        return xs[0]

    batch_size = xs[0].shape[0]
    if append_one:
        ones = torch.ones(batch_size, 1, dtype=xs[0].dtype, device=xs[0].device)
        xs = [torch.cat([x, ones], dim=1) for x in xs]

    if len(xs) == 2:
        x = torch.einsum("bi,bj->bij", xs[0], xs[1]).view(batch_size, -1)
    else:
        assert len(xs) == 3
        x = torch.einsum("bi,bj,bk->bijk", xs[0], xs[1], xs[2]).view(batch_size, -1)

    if append_one:
        return x[:, :-1]

    return x


@torch.jit.script
def cosine_similarity(
    x: torch.Tensor,
    x_t: torch.Tensor,
    scale: torch.Tensor,
    cache: dict[str, torch.Tensor],
    eps: float = 1e-8,
    ids: Optional[torch.Tensor] = None,
    ids_t: Optional[torch.Tensor] = None,
    same: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cosine Similarity."""
    x_norm: Optional[torch.Tensor] = None
    x_t_norm: Optional[torch.Tensor] = None
    if "x_norm" in cache:
        x_norm = cache["x_norm"]
    if "x_t_norm" in cache:
        x_t_norm = cache["x_t_norm"]

    if same:
        if x_norm is None and x_t_norm is not None:
            x_norm = x_t_norm.t()
        elif x_t_norm is None and x_norm is not None:
            x_t_norm = x_norm.t()

    if x_norm is None:
        x_norm = x.pow(2).sum(dim=1, keepdim=True).clamp(min=eps).sqrt()
    if x_t_norm is None and same:
        x_t_norm = x_norm.t()
    elif x_t_norm is None:
        x_t_norm = x_t.pow(2).sum(dim=0, keepdim=True).clamp(min=eps).sqrt()

    sim = torch.exp(
        ((x.mm(x_t)) / (x_norm * x_t_norm)).clamp(min=-1, max=1) * scale - scale,
    )
    return sim, {"x_t_norm": x_t_norm, "x_norm": x_norm}


@torch.jit.script
def rbf_similarity(
    x: torch.Tensor,
    x_t: torch.Tensor,
    scale: torch.Tensor,
    cache: dict[str, torch.Tensor],
    eps: float = 1e-8,
    ids: Optional[torch.Tensor] = None,
    ids_t: Optional[torch.Tensor] = None,
    same: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Similarity with an RBF kernel."""
    x_norm: Optional[torch.Tensor] = None
    x_t_norm: Optional[torch.Tensor] = None
    if "x_norm" in cache:
        x_norm = cache["x_norm"]
    if "x_t_norm" in cache:
        x_t_norm = cache["x_t_norm"]

    if same:
        if x_norm is None and x_t_norm is not None:
            x_norm = x_t_norm.t()
        elif x_t_norm is None and x_norm is not None:
            x_t_norm = x_norm.t()

    if x_norm is None:
        x_norm = x.pow(2).sum(dim=1, keepdim=True)
    if x_t_norm is None and same:
        x_t_norm = x_norm.t()
    elif x_t_norm is None:
        x_t_norm = x_t.pow(2).sum(dim=0, keepdim=True)

    sim = torch.exp(
        ((2.0 * torch.mm(x, x_t) - x_t_norm - x_norm) * scale).clamp(
            min=torch.log(torch.full([1], eps)).item(),
            max=0.0,
        ),
    )
    return sim, {"x_t_norm": x_t_norm, "x_norm": x_norm}


@torch.jit.script
def sqrtm(x: torch.Tensor, eps: float = 1e-8, exponent: float = 0.5) -> torch.Tensor:
    """Compute the square root of a symmetric positive definite matrix.

    Based on github.com/pytorch/pytorch/issues/25481#issuecomment-576258403
    """
    x = (x + x.transpose(-2, -1)) / 2
    values, vectors = torch.linalg.eigh(x)
    return (
        vectors * torch.pow(values.clamp(min=eps), exponent).unsqueeze(-2)
    ) @ vectors.transpose(-2, -1)


@torch.jit.script
def trace_sqrtm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = (x + x.transpose(-2, -1)) / 2
    values = torch.linalg.eigvalsh(x)
    return values.clamp(min=eps).sqrt().sum(dim=-1)


def _clustered_mean_covs(
    ids: torch.Tensor,
    x: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate mean and covariance for each cluster."""
    unique_ids, id_index = unique_with_index(ids)
    means: list[torch.Tensor] = []
    covs: list[torch.Tensor] = []
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * eps
    for index in unique_ids.view(-1, 1) == ids.view(1, -1):
        x_subset = x[index]
        mean = x_subset.mean(dim=0, keepdim=True)
        x_subset = x_subset - mean
        covs.append(x_subset.t().mm(x_subset) / (x_subset.shape[0] - 1) + eye)
        means.append(mean)

    return unique_ids, id_index, torch.cat(means), torch.stack(covs)


@torch.jit.script
def frechet_distance(
    x: torch.Tensor,
    x_t: torch.Tensor,
    ids: torch.Tensor,
    ids_t: torch.Tensor,
    cache: dict[str, torch.Tensor],
    eps: float = 1e-8,
    same: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pairwise Frechet distance between two sets of clusters."""
    features = x.shape[1]

    # caching
    unique_ids: Optional[torch.Tensor] = None
    unique_ids_t: Optional[torch.Tensor] = None
    means: Optional[torch.Tensor] = None
    means_t: Optional[torch.Tensor] = None
    covs: Optional[torch.Tensor] = None
    covs_t: Optional[torch.Tensor] = None
    id_index: Optional[torch.Tensor] = None
    id_t_index: Optional[torch.Tensor] = None
    if "unique_ids" in cache:
        unique_ids = cache["unique_ids"]
        means = cache["means"]
        covs = cache["covs"]
        id_index = cache["id_index"]
    if "unique_ids_t" in cache:
        unique_ids_t = cache["unique_ids_t"]
        means_t = cache["means_t"]
        covs_t = cache["covs_t"]
        id_t_index = cache["id_t_index"]
    if same and unique_ids is not None and unique_ids_t is None:
        unique_ids_t = unique_ids
        means_t = means
        covs_t = covs
        id_t_index = id_index
    if same and unique_ids is None and unique_ids_t is not None:
        unique_ids = unique_ids_t
        means = means_t
        covs = covs_t
        id_index = id_t_index

    # determine mean and covariance for each clusters
    if unique_ids is None:
        unique_ids, id_index, means, covs = _clustered_mean_covs(
            ids.view(-1, 1),
            x,
            eps,
        )
    assert means is not None
    assert covs is not None
    assert id_index is not None

    if same:
        means_t = means
        covs_t = covs
        unique_ids_t = unique_ids
        id_t_index = id_index
    elif unique_ids_t is None:
        unique_ids_t, id_t_index, means_t, covs_t = _clustered_mean_covs(
            ids_t.view(-1, 1),
            x_t.t(),
            eps,
        )
    assert means_t is not None
    assert covs_t is not None
    assert id_t_index is not None

    # mean terms
    mean_term = (means.unsqueeze(1) - means_t.unsqueeze(0)).pow(2).sum(dim=-1)
    # trace terms
    trace_a = covs.diagonal(dim1=1, dim2=2).sum(dim=1)
    trace_b = covs_t.diagonal(dim1=1, dim2=2).sum(dim=1)
    ab = (
        torch.mm(covs.view(-1, features), covs_t.view(-1, features).t())
        .view(unique_ids.shape[0], features, unique_ids_t.shape[0], features)
        .transpose(dim0=1, dim1=2)
        .reshape(-1, features, features)
    )
    if same:
        indices = torch.triu_indices(unique_ids.shape[0], unique_ids.shape[0], offset=1)
        mask = indices[0] + indices[1] * unique_ids.shape[0]
        trace_ab = torch.diag((trace_a + trace_b) / 2)
        trace_ab[indices[0], indices[1]] = 2 * trace_sqrtm(ab[mask])
        trace_ab = trace_ab + trace_ab.t()
    else:
        trace_ab = 2 * trace_sqrtm(ab).view(unique_ids.shape[0], -1)
    return mean_term + trace_a.unsqueeze(1) + trace_b.unsqueeze(0) - trace_ab, {
        "unique_ids": unique_ids,
        "unique_ids_t": unique_ids_t,
        "means": means,
        "means_t": means_t,
        "covs": covs,
        "covs_t": covs_t,
        "id_index": id_index,
        "id_t_index": id_t_index,
    }


@torch.jit.script
def frechet_similarity(
    x: torch.Tensor,
    x_t: torch.Tensor,
    scale: torch.Tensor,
    cache: dict[str, torch.Tensor],
    eps: float = 1e-8,
    ids: Optional[torch.Tensor] = None,
    ids_t: Optional[torch.Tensor] = None,
    same: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compatible with DWAC."""
    assert ids is not None
    assert ids_t is not None

    distance, cache = frechet_distance(x, x_t, ids, ids_t, cache, eps=eps, same=same)

    # similarity
    return (
        torch.exp(
            (-distance * scale).clamp(
                min=torch.log(torch.full([1], eps)).item(),
                max=0.0,
            ),
        ),
        cache,
    )


@torch.jit.script
def cca(x: torch.Tensor, y: torch.Tensor, r: float = 1e-3) -> torch.Tensor:
    """Calculate the  Canonical Correlation Analysis."""
    n = x.shape[0]
    m_x = x.shape[1]
    m_y = y.shape[1]

    # covariance matrices
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov_xy = (1.0 / (n - 1)) * torch.matmul(x.t(), y)
    cov_xx = (1.0 / (n - 1)) * torch.matmul(x.t(), x) + r * torch.eye(
        m_x,
        device=x.device,
        dtype=x.dtype,
    )
    cov_yy = (1.0 / (n - 1)) * torch.matmul(y.t(), y) + r * torch.eye(
        m_y,
        device=y.device,
        dtype=y.dtype,
    )

    # matrix square root
    if m_x == m_y:
        cov_sqrts = sqrtm(torch.stack([cov_xx, cov_yy]), exponent=-0.5)
        cov_xx_sqrt = cov_sqrts[0]
        cov_yy_sqrt = cov_sqrts[1]
    else:
        cov_xx_sqrt = sqrtm(cov_xx, exponent=-0.5)
        cov_yy_sqrt = sqrtm(cov_yy, exponent=-0.5)

    t_val = torch.matmul(torch.matmul(cov_xx_sqrt, cov_xy), cov_yy_sqrt)
    return trace_sqrtm(torch.matmul(t_val.t(), t_val)) / min(m_x, m_y)


@torch.jit.script
def center_symm_gram(gram: torch.Tensor) -> torch.Tensor:
    """Centers a self-gram matrix.

    Notes:
        Based on github.com/google-research/google-research/blob/master/
            representation_similarity/Demo.ipynb
        The Annals of Statistics, 42(6), 2382-2412

    Args:
        gram: Gram matrix of observations with themselves.

    Returns:
        The centered gram matrix.
    """
    n = gram.shape[0]
    diag_mask = 1 - torch.eye(n, device=gram.device, dtype=gram.dtype)
    gram = gram * diag_mask
    means = gram.sum(dim=0).view(-1) / (n - 2)
    centered_mean = means - means.sum() / (2 * (n - 1))
    gram = gram - centered_mean[:, None] - centered_mean[None, :]
    return gram * diag_mask


@torch.jit.script
def normalized_hsic(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the normalized Hilbert Schmidt Independence Criterion (aka CKA).

    Notes:
        Based on github.com/google-research/google-research/blob/master/
            representation_similarity/Demo.ipynb

    Args:
        gram_x: First self-gram matrix;
        gram_y: Second self-gram matrix.

    Returns:
        The normalized Hilbert Schmidt Independence Criterion.
    """
    gram_x = center_symm_gram(gram_x)
    gram_y = center_symm_gram(gram_y)

    # normalization
    gram_x = gram_x / torch.linalg.norm(gram_x)
    gram_y = gram_y / torch.linalg.norm(gram_y)

    return gram_x.view(-1).dot(gram_y.view(-1))


def pop_with_prefix(dictionary: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Pop keys with the specified prefix from the dictionary into a new dictionary.

    Args:
        dictionary: Dictionary containing keys with a certain prefix.
        prefix: The prefix to look for.

    Returns:
        dictionary with matching keys (prefix is removed).
    """
    return {
        key.removeprefix(prefix): dictionary.pop(key)
        for key in tuple(dictionary)
        if key.startswith(prefix)
    }


# work around for github.com/pytorch/pytorch/issues/62866
def _fun2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x


def _fun3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return x


class LossWrapper(torch.nn.Module):
    flag3: Final[bool]

    def __init__(
        self,
        function: Union[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        super().__init__()
        self.function2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _fun2
        self.function3: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ] = _fun3
        self.flag3 = len(inspect.signature(function).parameters) == 3
        if self.flag3:
            self.function3 = function
        else:
            self.function2 = function

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward data to the wrapped loss function.

        Args:
            y_hat: The predictions.
            y: The ground truth.
            ids: IDs used for clustered scores.

        Returns:
            The scalar loss value.
        """
        if self.flag3:
            return self.function3(y_hat, y, ids)
        return self.function2(y_hat, y)


class LossModule(LossModuleABC):
    """PyTorch modules which integrates a loss function."""

    training_validation: Final[bool]
    attenuation: Final[Literal["", "gaussian", "horserace"]]
    attenuation_lambda: Final[float]

    @beartype
    def __init__(
        self,
        *,
        loss_function: Union[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            str,
        ] = "MSELoss",
        attenuation: Literal["", "gaussian", "horserace"] = "",
        attenuation_lambda: float = -1.0,
        sample_weight: Optional[dict[int, float]] = None,
        training_validation: bool = False,
    ) -> None:
        """Base-class for all neural networks, contains the loss function.

        loss_function   The loss function
        attenuation     Whether to use the loss attenuation mentioned in
                        'gaussian': [1] equation 5
                        'horserace': [2]
        attenuation_lambda  The weighting of reward (horserace) or the regularization
                            (gaussian).
        sample_weight   A dictionary of IDs and weights
        training_validation Whether to run a second training instance without
                            learning.

        [1] "What Uncertainties Do We Need in Bayesian Deep Learning for Computer
            Vision?"
        [2] "Deep Gamblers: Learning to Abstain with Portfolio Theory"
        """
        super().__init__()

        # fix RNG
        torch.manual_seed(1)

        # set loss function
        self.sample_weight = sample_weight
        if isinstance(loss_function, str):
            loss_function = getattr(torch.nn.modules.loss, loss_function)(
                reduction="none",
            )
        assert not isinstance(loss_function, str)
        self.loss_function = LossWrapper(loss_function)

        self.training_validation = training_validation
        weights = getattr(loss_function, "weight", None)
        weights = [1.0] * 100 if weights is None else weights.cpu().tolist()
        self.class_weights = dict(enumerate(weights))

        # min/max for regression
        self.register_buffer("min", torch.zeros(0))
        self.register_buffer("max", torch.zeros(0))

        # make model aware which parameters are frozen
        self.exclude_parameters_prefix: tuple[str, ...] = ()

        # loss attenuation
        if attenuation_lambda < 0:
            if attenuation == "horserace":
                attenuation_lambda = 3.0
            elif attenuation == "gaussian":
                attenuation_lambda = 1.0
            else:
                attenuation_lambda = 0.1
        self.attenuation = attenuation
        self.attenuation_lambda = attenuation_lambda

        # epoch counter (incremented externally, starts at 1)
        self.register_buffer("epoch", torch.ones(1, dtype=torch.int32))

        # assume module can be JITed, mark exceptions manually
        self.jit_me = True

    @torch.jit.export
    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the loss of the predictions.

        scores      The predicted outcomes.
        ground_truth    The ground truth.
        meta        A dictionary with all the meta data.

        Returns:
            The loss.
        """
        # flatten
        if (
            ground_truth.dim() == 2
            and ground_truth.shape[1] == 1
            and ground_truth.dtype == torch.int64
        ):
            ground_truth = ground_truth.view(-1)
        # preparation for attenuated loss
        if self.attenuation == "horserace":
            scores = torch.softmax(scores, dim=1)
        log_sigma_2 = scores  # for jit
        if self.attenuation != "":
            log_sigma_2 = scores[:, -1:]
            scores = scores[:, :-1]

        if loss is None:
            loss = self.loss_function(scores, ground_truth, meta["meta_id"])
        assert isinstance(loss, torch.Tensor)
        if loss.ndim == 1:
            loss = loss.unsqueeze(1)

        if self.attenuation == "gaussian":
            # gaussian attenuation
            assert loss.shape[0] == scores.shape[0]
            log_sigma_2 = log_sigma_2.clamp(min=-5)
            loss = (
                loss * torch.exp(-log_sigma_2) + self.attenuation_lambda * log_sigma_2
            )

        elif self.attenuation == "horserace":
            # horserace
            reservation = log_sigma_2
            weights = torch.tensor(
                [self.class_weights[x.item()] for x in ground_truth.cpu().view(-1)],
                device=scores.device,
                dtype=scores.dtype,
            )
            loss = neg_log_likelihood(
                scores * self.attenuation_lambda + reservation,
                ground_truth,
                class_weights=weights,
            )
            assert loss is not None

        # sample weights
        if "meta_sample_weight" in meta:
            sample_weight = meta["meta_sample_weight"]
            assert sample_weight.shape[0] == loss.shape[0]
            loss = loss * sample_weight

        elif self.sample_weight is not None:
            assert meta["meta_id"].shape[0] == loss.shape[0]
            weights = torch.tensor(
                [
                    self.sample_weight.get(identifier[0], 1.0)
                    for identifier in meta["meta_id"]
                ],
                device=loss.device,
                dtype=loss.dtype,
            ).view(-1, 1)
            loss = loss * weights

        if take_mean:
            return loss.mean()

        return loss

    @torch.jit.ignore
    def device(self) -> torch.device:
        """Return the device the model is on."""
        for parameter in self.parameters():
            return parameter.device
        return torch.device("cpu")

    @torch.jit.ignore
    def disable_gradient_for_excluded(
        self,
        exclude_parameters_prefix: tuple[str, ...],
    ) -> None:
        """Disable gradients for listed parameters.

        Note:
            And re-enables gradients for other parameters.

        Args:
            exclude_parameters_prefix: Parameter names with matching prefixes
                will be disabled.
        """
        self.exclude_parameters_prefix = exclude_parameters_prefix
        for name, parameter in self.named_parameters():
            if name.startswith(exclude_parameters_prefix):
                parameter.requires_grad_(False)
            else:
                parameter.requires_grad_(True)

        # and set in all child modules
        for child in self.children():
            if hasattr(child, "exclude_parameters_prefix"):
                child.exclude_parameters_prefix = exclude_parameters_prefix

    @beartype
    def can_jit(self) -> bool:
        """Return whether this module can be jit'ed."""
        if not self.jit_me:
            return False

        return all(
            not hasattr(child, "jit_me") or child.jit_me for child in self.modules()
        )


class PoolTorch(LossModule):
    """Generic pooling provided by pytorch."""

    dim: Final[int]

    @beartype
    def __init__(self, *, name: str = "mean", dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        self.fun = getattr(torch, name)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = self.fun(x, dim=self.dim, keepdim=True)
        return meta["meta_y_hat"], meta


class PoolMax(PoolTorch):
    """Pool the maximum."""

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = torch.max(x, dim=self.dim, keepdim=True)[0]
        return meta["meta_y_hat"], meta


class PoolLast(PoolTorch):
    """Pool the last state."""

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = torch.torch.unsqueeze(x.select(self.dim, -1), self.dim)
        return meta["meta_y_hat"], meta


class PoolVar(PoolTorch):
    """Calculate the variance (set to 0 if there is no variability)."""

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        variance = torch.var(x, dim=self.dim, keepdim=True)
        index = torch.isnan(variance)
        if index.any():
            variance_ = torch.zeros_like(variance)
            index = ~index
            variance_[index] = variance[index]
            variance = variance_
        meta["meta_y_hat"] = variance
        return variance, meta


class PoolStats(LossModule):
    """Calculate mean and variance."""

    dim: Final[int]

    @beartype
    def __init__(self, *, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
        self.mean = PoolTorch(name="mean", dim=dim)
        self.var = PoolVar(dim=dim)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = torch.cat(
            [self.mean(x, meta)[0], self.var(x, meta)[0]],
            dim=-1,
        )
        return meta["meta_y_hat"], meta


class PoolWeightedMean(LossModule):
    """Weighted mean using softmax over MLP."""

    @beartype
    def __init__(
        self,
        *,
        input_size: int = -1,
        dim: int = 0,
        layer_sizes: Optional[tuple[int, ...]] = None,
        layers: int = -1,
    ) -> None:
        super().__init__()
        assert dim == 0, dim
        self.mlp = MLP(
            input_size=input_size,
            output_size=1,
            activation={"name": "ReLU"},
            final_activation={"name": "linear"},
            layers=layers,
            layer_sizes=layer_sizes,
        )
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = torch.sum(
            self.softmax(self.mlp(x, meta, y=y, dataset=dataset)[0]) * x,
            dim=0,
            keepdim=True,
        )
        return meta["meta_y_hat"], meta


@beartype
def get_pool_module(*, name: str = "", **kwargs: Any) -> LossModule:
    """Return the requested pooling function."""
    if name == "no":
        return LinearActivation(**kwargs)
    if name == "var":
        return PoolVar(**kwargs)
    if name == "max":
        return PoolMax(**kwargs)
    if name == "last":
        return PoolLast(**kwargs)
    if name == "stats":
        return PoolStats(**kwargs)
    if name == "weighted_mean":
        return PoolWeightedMean(**kwargs)
    return PoolTorch(name=name, **kwargs)


@beartype
def get_activation_module(
    *,
    model: LossModule,
    name: str = "",
    **kwargs: Any,
) -> LossModule:
    """Return a module for the activation function.

    model   The model requiring the activation function.
    name    Name of the activation function.
    kwargs  Parameters of the activation function

    Returns:
        A torch.nn.module representing the activation function.
    """
    match name:
        case "linear":
            activation = LinearActivation(**kwargs)
        case "gpvfe":
            import python_tools.ml.nonparametric

            activation = python_tools.ml.nonparametric.GPVFEActivation(**kwargs)
        case "dwac":
            import python_tools.ml.nonparametric

            activation = python_tools.ml.nonparametric.DWACActivation(**kwargs)
        case "lme":
            import python_tools.ml.mixed

            activation = python_tools.ml.mixed.LinearMixedEffects(**kwargs)
        case "WeightedLookup":
            activation = WeightedLookup(**kwargs)
        case _:
            activation = TorchActivation(name=name, **kwargs)
    model.training_validation = activation.training_validation
    return activation


class MLP(LossModule):
    """Creates a generic MLP."""

    @beartype
    def __init__(
        self,
        *,
        input_size: int = -1,
        output_size: int = -1,
        layers: int = -1,
        layer_sizes: Optional[tuple[int, ...]] = None,
        activation: dict[str, Any],
        final_activation: dict[str, Any],
        dropout: float = 0.0,
        layer_fun: Callable[[int, int], torch.nn.Module] = torch.nn.Linear,
        **kwargs: Any,
    ) -> None:
        """Multi-layer perceptron.

        input_size      Input dimension.
        output_size     Output dimension.
        layers          Number of hidden layers (>=0).
        activation      Function applied after hidden layers.
        final_activation Function applied after the last layer.
        dropout         The dropout rate after each layer.
        layer_sizes     If None, linearly interpolate the dimensions. A list of length
                        layer - 1 (it doesn't contain input and output size).
        """
        super().__init__(**kwargs)

        if layer_sizes is None:
            assert layers >= -1, layers
            layer_sizes = tuple(
                torch.linspace(
                    input_size,
                    output_size,
                    layers + 2,
                    dtype=torch.int,
                ).tolist(),
            )
        else:
            layer_sizes = (input_size, *layer_sizes, output_size)
        assert layer_sizes is not None
        if final_activation["name"] == "lme" and len(layer_sizes) > 1:
            final_activation = final_activation.copy()
            final_activation.setdefault("embedding_size", int(layer_sizes[-2]))

        # activation functions
        activation = get_activation_module(model=self, **activation, **kwargs)
        self.final_activation = None
        final_activation = get_activation_module(
            model=self,
            **final_activation,
            **kwargs,
        )
        if not isinstance(final_activation, LinearActivation):
            self.final_activation = final_activation
            self.loss = self.final_activation.loss

        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(layer_fun(int(layer_sizes[i - 1]), int(layer_sizes[i])))
            if layers[-1].bias is not None:
                layers[-1].bias.data.fill_(0)

            if i >= len(layer_sizes) - 1:
                continue

            if dropout > 0.0:
                layers.append(torch.nn.Dropout(p=dropout))

            if isinstance(activation, LinearActivation):
                continue
            assert isinstance(activation, TorchActivation)
            layers.append(activation.activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        x   A tensor.

        Returns:
            A tensor.
        """
        x_embedding = x
        for layer in self.layers:
            x_embedding = x
            x = layer(x)
        meta["meta_embedding"] = x_embedding

        # final activation
        if self.final_activation is None:
            pass
        elif self.attenuation != "":
            score = x[:, -1:]
            x, meta = self.final_activation(x[:, :-1], meta, y=y, dataset=dataset)
            x = torch.cat((x, score), dim=1)
        else:
            x, meta = self.final_activation(x, meta, y=y, dataset=dataset)
        meta["meta_y_hat"] = x
        return x, meta


class LinearActivation(LossModule):
    """Simple linear activation function."""

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = x
        return x, meta


class TorchActivation(LossModule):
    """Wrapper for torch's activation functions."""

    @beartype
    def __init__(self, *, name: str = "", **kwargs: Any) -> None:
        loss_keys = (
            "loss_function",
            "attenuation",
            "attenuation_lambda",
            "sample_weight",
            "training_validation",
        )
        super().__init__(
            **{key: kwargs.pop(key) for key in tuple(kwargs) if key in loss_keys},
        )
        self.activation = getattr(torch.nn, name)(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        meta["meta_y_hat"] = self.activation(x)
        return meta["meta_y_hat"], meta


class WeightedLookup(LossModule):
    lookup_name: Final[str]

    @beartype
    def __init__(self, lookup_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lookup_name = lookup_name

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.softmax(x, dim=-1)
        y = meta[self.lookup_name].to(device=x.device, dtype=x.dtype, non_blocking=True)
        meta["meta_embedding_weighting"] = x
        meta["meta_y_hat"] = (x * y).sum(dim=-1, keepdim=True)
        return meta["meta_y_hat"], meta


class InteractionModel(LossModule):
    """Module version of interaction_terms."""

    append_one: Final[bool]
    view_starts: Final[torch.Tensor]

    @beartype
    def __init__(
        self,
        input_sizes: tuple[int, ...],
        append_one: bool = True,
        **kwargs: Any,
    ) -> None:
        """Model of interaction_terms.

        Args:
            input_sizes: Used to determine where modalities start and end.
            append_one: See interaction_terms.
            kwargs: Forwarded to LossModule.
        """
        super().__init__(**kwargs)
        self.append_one = append_one
        self.view_starts = torch.cumsum(
            torch.LongTensor([0] + [x for x in input_sizes if x > 0]),
            dim=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward-pass through the module.

        Args:
            x: The input matrix. First dimension is the batch dimension.
            meta: Dictionary containing partial computations and meta data.
            y: The ground truth (only used by non-parametric models during training).
            dataset: The dataset name.

        Returns:
            The output and meta.
        """
        meta["meta_y_hat"] = interaction_terms(
            [
                x[:, start : self.view_starts[view + 1]]
                for view, start in enumerate(self.view_starts[:-1])
            ],
            append_one=self.append_one,
        )
        return meta["meta_y_hat"], meta


class AttenuatedModalityExperts(LossModule):
    """Decision fusion for modalities using loss attenuation."""

    competitive: Final[bool]
    latent_gating: Final[bool]
    joint_attenuation: Final[bool]
    view_indices: Final[torch.Tensor]

    @beartype
    def __init__(
        self,
        *,
        input_sizes: tuple[int, ...],
        output_size: int = -1,
        final_activation: dict[str, Any],
        competitive: bool = False,
        combinations: Optional[tuple[tuple[int, ...], ...]] = None,
        latent_gating: bool = False,
        joint_attenuation: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        attenuation = kwargs.pop("attenuation")
        if joint_attenuation is None:
            joint_attenuation = {}
        joint_attenuation = joint_attenuation.copy()
        if latent_gating:
            assert not attenuation
        if joint_attenuation:
            assert latent_gating or attenuation
            output_size -= 1
        model = kwargs.pop("model")
        input_sizes = tuple(x for x in input_sizes if x > 0)
        kwargs.pop("input_size", None)
        if combinations is None:
            combinations = tuple((i,) for i in range(len(input_sizes)))
        combinations = tuple(tuple(sorted(views)) for views in combinations)

        model_kwargs: list[dict[str, Any]] = [{} for _ in range(len(combinations))]
        for key, value in pop_with_prefix(kwargs, "model_").items():
            assert isinstance(value, tuple), f"{key} {type(value)} {value}"
            if len(value) == 1:
                value = value * len(combinations)
            for iview, model_kwarg in enumerate(model_kwargs):
                model_kwarg[key] = value[iview]
                model_kwarg.setdefault(
                    "input_size",
                    sum(input_sizes[view] for view in combinations[iview]),
                )

        super().__init__(**kwargs, attenuation=attenuation)
        self.jit_me = False

        for model_kwarg in model_kwargs:
            model_kwarg.setdefault("final_activation", final_activation)
            model_kwarg.setdefault("output_size", output_size + int(latent_gating))

        self.models = torch.nn.ModuleList(
            [
                model(attenuation=attenuation, **model_kwarg, **kwargs)
                for model_kwarg in model_kwargs
            ],
        )
        self.training_validation = self.models[0].training_validation
        self.competitive = competitive
        self.latent_gating = latent_gating
        self.joint_attenuation = bool(joint_attenuation)
        self.joint_attenuation_model: Optional[MLP] = None
        if self.joint_attenuation:
            if "input_size" not in joint_attenuation and hasattr(
                self.models[0],
                "layers",
            ):
                joint_attenuation["input_size"] = sum(
                    model.layers[-1].weight.shape[1] for model in self.models
                )
            self.joint_attenuation_model = MLP(
                output_size=len(model_kwargs),
                **joint_attenuation,
                **kwargs,
            )

        # pre-compute indices/slices
        view_starts = torch.cumsum(torch.LongTensor((0, *input_sizes)), dim=0)
        view_indices: list[Union[slice, torch.Tensor]] = []
        for views in combinations:
            # use a slice if possible (does not make a copy)
            index = (
                slice(view_starts[views[0]].item(), view_starts[views[-1] + 1].item())
                if len(views) == 1
                or all(
                    view + 1 == views[iview + 1]
                    for iview, view in enumerate(views[:-1])
                )
                else torch.cat(
                    [
                        torch.arange(view_starts[view], view_starts[view + 1])
                        for view in views
                    ],
                    dim=-1,
                )
            )

            view_indices.append(index)
        self.view_indices = view_indices

    def forward(
        self,
        x: Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence],
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # get predictions for each modality
        ys = []
        for iview, indices in enumerate(self.view_indices):
            x_ = (
                x[:, indices]
                if isinstance(x, torch.Tensor)
                else torch.nn.utils.rnn.PackedSequence(
                    x.data[:, indices],
                    batch_sizes=x.batch_sizes,
                    sorted_indices=x.sorted_indices,
                    unsorted_indices=x.unsorted_indices,
                )
            )
            # collect meta: determine new/updated entries
            known = {key: id(value) for key, value in meta.items()}
            y_hat, meta = self.models[iview](x_, meta, y=y, dataset=dataset)
            ys.append(y_hat)
            meta.update(
                {
                    f"{key}_{iview}": value
                    for key, value in meta.items()
                    if key not in known or known[key] != id(value)
                },
            )

        if self.joint_attenuation:
            # run joint model on concatenated embeddings + append
            assert self.joint_attenuation_model is not None
            xs = torch.cat(
                [meta[f"meta_embedding_{i}"] for i in range(len(self.view_indices))],
                dim=-1,
            )
            attenuations = self.joint_attenuation_model(
                xs,
                meta=meta,
                y=y,
                dataset=dataset,
            )[0]
            ys = [
                torch.cat([y_hat, attenuations[:, i, None]], dim=1)
                for i, y_hat in enumerate(ys)
            ]
        elif self.attenuation or self.latent_gating:
            attenuations = torch.cat(
                [y_attenuation[:, -1:] for y_attenuation in ys],
                dim=1,
            )
        else:
            attenuations = -torch.cat(
                [
                    torch.max(y_attenuation, dim=1, keepdim=True)[0]
                    for y_attenuation in ys
                ],
                dim=1,
            )
        normed_attenuations = torch.softmax(-attenuations, dim=1)

        y_hat = torch.sum(
            torch.cat(
                [
                    (
                        y_attenuation[:, :-1, None]
                        if (self.attenuation or self.latent_gating)
                        else y_attenuation.unsqueeze(2)
                    )
                    * normed_attenuations[:, iview, None, None]
                    for iview, y_attenuation in enumerate(ys)
                ],
                dim=-1,
            ),
            dim=-1,
        )

        # for loss function
        meta["meta_all_y_scores_sigma_2"] = torch.cat(ys, dim=1)
        meta["meta_all_normed_attenuations"] = normed_attenuations

        # for neural_models : y || attenuation
        combined_attenuation = torch.sum(
            torch.cat(
                [
                    attenuations[:, iview, None] * normed_attenuations[:, iview, None]
                    for iview in range(len(self.models))
                ],
                dim=1,
            ),
            dim=1,
            keepdim=True,
        )
        meta["meta_sigma_2"] = combined_attenuation
        if self.attenuation:
            y_hat = torch.cat([y_hat, combined_attenuation], dim=1)

        return y_hat, meta

    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the loss.

        Note:
            loss = lambda * sum (attenuated individual loss)
                   + loss of weighted averages.
        """
        if not self.training or not self.competitive:
            # loss of weighted average
            return super().loss(scores, ground_truth, meta)

        # individual losses: for competitive loss
        size = meta["meta_all_y_scores_sigma_2"].shape[1] // len(self.models)
        loss_list = []
        for iview, model in enumerate(self.models):
            # without attenuation
            scores_view = meta["meta_all_y_scores_sigma_2"][
                :,
                iview * size : (iview + 1) * size,
            ]
            attenuation = model.loss.__self__.attenuation
            model.loss.__self__.attenuation = ""
            loss_list.append(
                model.loss(
                    scores_view[:, :-1],
                    ground_truth,
                    meta,
                    take_mean=False,
                ).mean(dim=1, keepdim=True),
            )
            model.loss.__self__.attenuation = attenuation

        # use competitive loss ("Adaptive Mixtures of Local Experts")
        losses = torch.cat(loss_list, dim=1)
        losses = -torch.log(
            torch.sum(
                meta["meta_all_normed_attenuations"] * torch.exp(-losses),
                dim=1,
                keepdim=True,
            ).clamp(min=1e-8),
        )
        return super().loss(
            scores,
            ground_truth,
            meta,
            loss=losses,
            take_mean=take_mean,
        )


@torch.jit.interface
class LossModuleInterface(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass


class Ensemble(LossModule):
    @beartype
    def __init__(
        self,
        *,
        size: int = 8,
        model: type[LossModule] = MLP,
        **kwargs: Any,
    ) -> None:
        model_kwargs = pop_with_prefix(kwargs, "model_")
        model_kwargs.setdefault("input_size", kwargs.pop("input_size"))
        model_kwargs.setdefault("output_size", kwargs.pop("output_size"))
        super().__init__(**kwargs)

        self.models = torch.nn.ModuleList(
            [model(**model_kwargs, **kwargs) for _ in range(size)],
        )

    def _index_forward_weigh(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor],
        dataset: str,
        imodel: int,
        index: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # run on subset
        model: LossModuleInterface = self.models[imodel]
        y_hat, meta = model.forward(
            x[index],
            index_dict(meta, index),
            y if y is None else y[index],
            dataset,
        )

        # weight to make aggregation easier
        weight = weight[index].unsqueeze(1)
        weight_cpu = weight.cpu()
        y_hat = y_hat * weight
        for key, value in meta.items():
            if value.device == weight.device:
                meta[key] = value * weight
            else:
                meta[key] = value * weight_cpu

        return y_hat, meta

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
        weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # run models in parallel
        if weights is not None:
            # given (sparse) weights
            # skip ensembles without all-zero weights
            keep = torch.nonzero(weights.sum(dim=0) > 0)[:, 0]
            weights = weights[:, keep]
            weights_binary = weights > 0

            # normalize weights
            weights = weights / weights.sum(dim=1, keepdim=True)

            # run on samples with positive weights
            index = torch.nonzero(weights_binary)
            meta_old = meta.copy()
            futures = [
                torch.jit.fork(
                    self._index_forward_weigh,
                    x,
                    meta_old,
                    y,
                    dataset,
                    imodel,
                    index[index[:, 1] == iweight, 0],
                    weights[:, iweight],
                )
                for iweight, imodel in enumerate(keep)
            ]
            index = index[torch.argsort(index[:, 1], stable=True), 0]

            # aggregate
            ys = []
            metas: list[dict[str, torch.Tensor]] = []
            for future in futures:
                y_hat, meta = torch.jit.wait(future)
                ys.append(y_hat)
                metas.append(meta)
            tmp = torch.zeros(
                x.shape[0],
                ys[0].shape[1],
                device=ys[0].device,
                dtype=ys[0].dtype,
            )
            ys = tmp.index_add(0, index, torch.cat(ys))
            index_cpu = index.cpu()
            meta = {}
            for key, value in metas[0].items():
                tmp = torch.zeros(
                    x.shape[0],
                    value.shape[1],
                    device=value.device,
                    dtype=value.dtype,
                )
                index_ = index if value.device == index.device else index_cpu
                meta[key] = tmp.index_add(
                    0,
                    index_,
                    torch.cat([meta[key] for meta in metas]),
                )
                if key in meta_old:
                    meta[key] = meta[key].to(dtype=meta_old[key].dtype)
            meta["meta_embedding"] = ys
            meta["meta_y_hat"] = ys
            return ys, meta

        # sample (dense) weights: run all models
        futures = [
            torch.jit.fork(model, x, meta.copy(), y=y, dataset=dataset)
            for model in self.models
        ]
        ys = torch.stack([torch.jit.wait(future)[0] for future in futures], dim=1)

        if self.training:
            # bootstrapping as multiplication
            weights = torch.zeros(ys.shape, device=ys.device, dtype=ys.dtype)
            ones = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
            for i in range(len(self.models)):
                weights[:, i].index_add_(
                    0,
                    torch.randint(
                        0,
                        x.shape[0],
                        (x.shape[0],),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                    ones,
                )

            # make sure every element has at least one prediction
            weights_sum = weights.sum(dim=1, keepdim=True)
            zeros = weights_sum == 0
            if zeros.any():
                index = torch.randint(
                    0,
                    len(self.models),
                    (int(torch.sum(zeros).item()),),
                    device=weights.device,
                    dtype=torch.int64,
                )
                weights[zeros.view(-1), index] = 1
            weights_sum = weights.sum(dim=1, keepdim=True)
            ys = (ys * weights.unsqueeze(-1)).sum(dim=1) / weights_sum
        else:
            ys = ys.mean(dim=1)
        meta["meta_embedding"] = ys
        meta["meta_y_hat"] = ys
        return ys, meta


class CNN(MLP):
    """Creates a generic n-dimensional CNN."""

    @beartype
    def __init__(
        self,
        *,
        conv_type: int = 1,
        in_channel: int = 1,
        n_kernels: tuple[int, ...] = (1,),
        kernel_shapes: tuple[tuple[int, ...], ...] = ((1,),),
        activation: dict[str, Any],
        final_activation: dict[str, Any],
        dropout: float = 0.0,
        cnn_kwargs: tuple[dict[str, Any], ...] = ({},),
        **kwargs: Any,
    ) -> None:
        """CNN.

        conv_type   Type of the convolution (1d, 2d, or 3d).
        in_channel  Number of channels along which the first kernel is
        replicated.
        n_kernels   Number of kernels for each layer.
        kernel_shapes List of the kernel shapes.
        activation  Activation function between layers.
        final_activation Activation function after the last layer.
        dropout     Rate of the dropout within the CNN network.
        cnn_kwargs  List of kwargs forwarded to pytorch.
        kwargs      Forwarded to LossModule.

        Forward:
            Input tensor (batch size x in_channel x unrolling)
        """
        super().__init__(
            activation=activation,
            dropout=dropout,
            final_activation=final_activation,
            **kwargs,
        )
        assert len(n_kernels) == len(kernel_shapes)
        # type
        conv_fun = torch.nn.Conv1d
        if conv_type == 2:
            conv_fun = torch.nn.Conv2d
        elif conv_type == 3:
            conv_fun = torch.nn.Conv3d

        # replicate default values
        if len(n_kernels) != len(cnn_kwargs) and len(cnn_kwargs) == 1:
            cnn_kwargs = cnn_kwargs * len(n_kernels)

        # CNNs
        modules = []
        for i, (kernels, shape, cnn_kwarg) in enumerate(
            zip(n_kernels, kernel_shapes, cnn_kwargs),
        ):
            in_size = in_channel
            if i != 0:
                in_size = n_kernels[i - 1]
            modules.append(conv_fun(in_size, kernels, shape, **cnn_kwarg))
        self.layers = torch.nn.ModuleList(modules)


class RNN(LossModule):
    """A simple wrapper for RNNs."""

    @beartype
    def __init__(
        self,
        *,
        rnn_type: str = "lstm",
        input_size: int = 1,
        output_size: int = 1,
        **kwargs: Any,
    ) -> None:
        """Instantiate a RNN.

        rnn_type    String to determine which type of RNN to use: 'rnn',
                    'lstm', or 'gru'.
        input_size  The input size of the RNN.
        output_size The hidden size of the RNN.
        kwargs      Other arguments for the RNN.
        """
        rnn_kwargs = pop_with_prefix(kwargs, "rnn_")
        rnn_kwargs["batch_first"] = True
        super().__init__(**kwargs)
        self.jit_me = False

        # define RNN
        rnn_type = rnn_type.upper()
        rnn = getattr(torch.nn.modules.rnn, rnn_type)
        self.rnn = rnn(input_size=input_size, hidden_size=output_size, **rnn_kwargs)

    def forward(
        self,
        x: Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence],
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[
        Union[torch.nn.utils.rnn.PackedSequence, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Forward pass.

        data    A packed sequence.

        Returns:
            A tuple containing two packed sequences.
        """
        return self.rnn(x)[0], meta


class NNPoolNN(LossModule):
    """A NN followed by some pooling and a final NN."""

    concat: Final[bool]
    bootstrap: Final[bool]

    @beartype
    def __init__(
        self,
        *,
        input_size: int = -1,
        output_size: int = -1,
        pooling_size: int = -1,
        pooling: dict[str, Any],
        concat: bool = False,
        bootstrap: bool = True,
        final_activation: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Process, pool, process.

        pooling     Defines the pooling function. Either a function or a
                    string (from torch.nn.modules.pooling). Pooling over the
                    second dimension.
        model_before/after  Class of the model run before/after the pooling.
        model_before/after_kwargs   Keyword arguments for the models.
        kwargs      Arguments forwarded to LossModule.
        """
        model_before_kwargs = pop_with_prefix(kwargs, "model_before_")
        model_after_kwargs = pop_with_prefix(kwargs, "model_after_")

        model_after_kwargs.setdefault("final_activation", final_activation)
        model_before_kwargs.setdefault("input_size", input_size)
        model_after_kwargs.setdefault("output_size", output_size)
        model_before = kwargs.pop("model_before")
        model_after = kwargs.pop("model_after")
        super().__init__(**kwargs)
        self.jit_me = False

        # determine output/input size
        pooling_size_output = pooling_size
        if pooling["name"] == "stats":
            pooling_size_output = pooling_size * 2
        model_before_kwargs.setdefault("output_size", pooling_size)
        model_after_kwargs.setdefault(
            "input_size",
            pooling_size_output + pooling_size if concat else pooling_size_output,
        )

        self.pooling = get_pool_module(**pooling)
        self.model_before = model_before(**model_before_kwargs, **kwargs)
        self.model_after = model_after(**model_after_kwargs, **kwargs)

        self.training_validation = (
            self.model_before.training_validation
            or self.model_after.training_validation
        )
        self.loss = self.model_after.loss

        if concat:
            self.norm = torch.nn.BatchNorm1d(int(pooling_size_output), affine=False)
        self.concat = concat
        self.bootstrap = bootstrap

    def forward(
        self,
        x: Union[torch.Tensor, torch.nn.utils.rnn.PackedSequence],
        meta: dict[str, torch.Tensor],
        dataset: str = "",
        y: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        x   A tensor (1 x in_channels x unrolling).

        Returns:
            A list of tensors.
        """
        x = self.model_before(x, meta, dataset=dataset, y=y)[0]

        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            # pool each sequence
            x = torch.cat([self.pooling(x, meta)[0] for x in packedsequence_to_list(x)])

        else:
            assert isinstance(x, torch.Tensor)
            # pool each ID
            xs = None
            ids = meta["meta_id"].view(-1)
            for identifier in torch.unique(ids):
                index = identifier == ids
                x_index = x[index]

                if self.bootstrap and self.training:
                    # bootstrap during training
                    index_perm = torch.randint(
                        x_index.shape[0],
                        (x_index.shape[0], x_index.shape[0]),
                        device=x_index.device,
                    )
                    x_index = self.pooling(x_index[index_perm], meta)[0]
                else:
                    x_index = self.pooling(x_index.unsqueeze(0), meta)[0].expand(
                        x_index.shape[0],
                        -1,
                        -1,
                    )
                if xs is None:
                    xs = torch.zeros(
                        x.shape[0],
                        x_index.shape[-1],
                        device=x.device,
                        dtype=x.dtype,
                    )
                if x_index.ndim == 3:
                    x_index = x_index.squeeze(1)
                xs[index] = x_index

            if self.concat:
                xs = torch.cat([x, self.norm(xs)], dim=-1)
            assert xs is not None
            x = xs

        meta["meta_embedding_pool"] = x
        return self.model_after(x, meta, dataset=dataset, y=y)
