"""Gaussian-mixture coordinates for the continuous-state transport algorithm.

This is the finite-dimensional representation of a population law used by
ContinuousTransport: the decoder Gamma_K, the mixture score
psi_K(x, z) = grad_z log gamma_K(x; z), its Jacobian D_z psi_K, and the
constrained EM fitting rule R_K. Coordinates follow the convention of
Appendix "Gaussian-mixture coordinates and representation",

    z = (beta, m_1, ..., m_K, s_1, ..., s_K) in R^{q_K},
    q_K = (K - 1) + K d + K d (d + 1) / 2,

with softmax weights whose K-th logit is pinned to zero, and covariances
Sigma_k = L(s_k) L(s_k)^T where L(s_k) is lower triangular with exponentiated
diagonal. Both reparametrizations are unconstrained, so Gamma_K(z) is a valid
mixture for every z in R^{q_K}: a perturbed coordinate never has to be projected
back into a feasible set. The bounds of the fitting set Z_K are enforced by the
fitting rule alone, where they only prevent a covariance from collapsing onto a
single observation.
"""

from dataclasses import dataclass
import math

import numpy
import torch


LOG_SCALE_LIMIT = 30.0


def coordinate_dim(n_components, dim):
    """Dimension q_K of the Gaussian-mixture coordinate space."""
    return (n_components - 1) + n_components * dim + n_components * dim * (dim + 1) // 2


@dataclass(frozen=True)
class MixtureConstraints:
    """Bounds defining the fitting set Z_K of admissible coordinates."""

    weight_floor: float = 1e-3
    sigma_min: float = 1e-3
    sigma_max: float = 1e3
    mean_radius: float | None = None


class GaussianMixture:
    """Decoder, score and fitting rule of the K-component Gaussian-mixture chart.

    All methods accept an arbitrary batch of coordinates: a coordinate tensor of
    shape (*batch, q_K) describes one mixture per leading index, which is what
    the main trajectory block needs when every trajectory carries its own
    perturbed population argument.
    """

    def __init__(self, n_components, dim, dtype, device, constraints=MixtureConstraints()):
        if n_components < 1:
            raise ValueError("A Gaussian-mixture chart needs at least one component.")
        self.n_components = n_components
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.constraints = constraints
        self.coordinate_dim = coordinate_dim(n_components, dim)
        self.scale_dim = dim * (dim + 1) // 2

        rows, columns = torch.tril_indices(dim, dim, device=device)
        self.tril_rows = rows
        self.tril_columns = columns
        self.tril_flat = rows * dim + columns
        self.is_diagonal = rows == columns

    def unpack(self, z):
        """Split a coordinate into weight logits, means, and covariance factors."""
        n_components, dim = self.n_components, self.dim
        offset = n_components - 1
        beta = z[..., :offset]
        means = z[..., offset : offset + n_components * dim]
        scales = z[..., offset + n_components * dim :]
        means = means.reshape(*z.shape[:-1], n_components, dim)
        scales = scales.reshape(*z.shape[:-1], n_components, self.scale_dim)
        return beta, means, scales

    def weights(self, beta):
        """Mixture weights of the softmax chart, with the K-th logit pinned to zero."""
        reference = torch.zeros(beta.shape[:-1] + (1,), dtype=beta.dtype, device=beta.device)
        return torch.softmax(torch.cat([beta, reference], dim=-1), dim=-1)

    def scale_tril(self, scales):
        """Lower-triangular Cholesky factors L(s_k) of the component covariances."""
        dim = self.dim
        diagonal = torch.exp(scales.clamp(min=-LOG_SCALE_LIMIT, max=LOG_SCALE_LIMIT))
        values = torch.where(self.is_diagonal, diagonal, scales)
        flat = torch.zeros(scales.shape[:-1] + (dim * dim,), dtype=scales.dtype, device=scales.device)
        index = self.tril_flat.expand(values.shape)
        return flat.scatter(-1, index, values).reshape(*scales.shape[:-1], dim, dim)

    def decode(self, z):
        """Gamma_K(z) as the triple of weights, means, and Cholesky factors."""
        beta, means, scales = self.unpack(z)
        return self.weights(beta), means, self.scale_tril(scales)

    def encode(self, weights, means, scale_tril):
        """Coordinate of the mixture with the given weights, means, and factors."""
        reference = weights[..., -1:].clamp_min(torch.finfo(weights.dtype).tiny)
        beta = torch.log(weights[..., :-1].clamp_min(torch.finfo(weights.dtype).tiny)) - torch.log(reference)
        values = scale_tril[..., self.tril_rows, self.tril_columns]
        values = torch.where(self.is_diagonal, torch.log(values.clamp_min(torch.finfo(values.dtype).tiny)), values)
        return torch.cat(
            [beta, means.reshape(*means.shape[:-2], -1), values.reshape(*values.shape[:-2], -1)], dim=-1
        )

    def component_log_densities(self, x, means, scale_tril):
        """Log Gaussian densities of every component, shaped (*batch, n, K)."""
        difference = x.unsqueeze(-2) - means.unsqueeze(-3)
        log_determinant = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(dim=-1)
        if self.dim == 1:
            mahalanobis = (difference.squeeze(-1) / scale_tril[..., 0, 0].unsqueeze(-2)).square()
        else:
            solved = torch.linalg.solve_triangular(
                scale_tril.unsqueeze(-4), difference.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahalanobis = solved.square().sum(dim=-1)
        normalizer = 0.5 * self.dim * math.log(2.0 * math.pi)
        return -normalizer - log_determinant.unsqueeze(-2) - 0.5 * mahalanobis

    def log_density(self, x, z):
        """log gamma_K(x; z) for points x of shape (*batch, n, d)."""
        weights, means, scale_tril = self.decode(z)
        component = self.component_log_densities(x, means, scale_tril)
        return torch.logsumexp(torch.log(weights).unsqueeze(-2) + component, dim=-1)

    def score(self, x, z):
        """psi_K(x_i, z) for every point, shaped (n, q_K)."""

        def single(point, coordinate):
            return self.log_density(point.unsqueeze(0), coordinate).squeeze(0)

        return torch.func.vmap(torch.func.grad(single, argnums=1), in_dims=(0, None))(x, z)

    def mean_score_jacobian(self, x, z, weights=None):
        """Weighted average of D_z psi_K(x_i, z), that is the matrix A of the paper.

        With uniform weights this is the empirical Jacobian of the particle block;
        with quadrature weights it is the exact one under a known mixture law.
        """

        def mean_log_density(coordinate):
            values = self.log_density(x, coordinate)
            if weights is None:
                return values.mean()
            return (values * weights).sum()

        return torch.func.jacrev(torch.func.jacrev(mean_log_density))(z)

    def mean_covariance(self, z):
        """Mean vector and covariance matrix of Gamma_K(z)."""
        weights, means, scale_tril = self.decode(z)
        mean = (weights.unsqueeze(-1) * means).sum(dim=-2)
        covariance = scale_tril @ scale_tril.transpose(-1, -2)
        centered = means - mean.unsqueeze(-2)
        spread = centered.unsqueeze(-1) * centered.unsqueeze(-2)
        return mean, (weights.unsqueeze(-1).unsqueeze(-1) * (covariance + spread)).sum(dim=-3)

    def hermite_rule(self, n_nodes):
        """Tensor-product Gauss-Hermite nodes and probability weights on R^d."""
        nodes, weights = numpy.polynomial.hermite.hermgauss(n_nodes)
        nodes = torch.as_tensor(nodes, dtype=self.dtype, device=self.device)
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device) / math.sqrt(math.pi)
        if self.dim == 1:
            return nodes.unsqueeze(-1), weights
        grids = torch.meshgrid(*([nodes] * self.dim), indexing="ij")
        weight_grids = torch.meshgrid(*([weights] * self.dim), indexing="ij")
        points = torch.stack([grid.reshape(-1) for grid in grids], dim=-1)
        return points, torch.stack([grid.reshape(-1) for grid in weight_grids], dim=-1).prod(dim=-1)

    def quadrature(self, z, n_nodes):
        """Nodes and weights integrating a test function against Gamma_K(z)."""
        weights, means, scale_tril = self.decode(z)
        nodes, node_weights = self.hermite_rule(n_nodes)
        points = means.unsqueeze(-2) + math.sqrt(2.0) * torch.einsum("...kij,pj->...kpi", scale_tril, nodes)
        combined = weights.unsqueeze(-1) * node_weights
        points = points.reshape(*points.shape[:-3], -1, self.dim)
        return points, combined.reshape(*combined.shape[:-2], -1)

    def constrain(self, weights, means, covariances):
        """Project mixture parameters onto the fitting set Z_K."""
        constraints = self.constraints
        weights = weights.clamp_min(constraints.weight_floor)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        if constraints.mean_radius is not None:
            if self.dim == 1:
                means = means.clamp(-constraints.mean_radius, constraints.mean_radius)
            else:
                norms = means.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(means.dtype).tiny)
                means = means * (constraints.mean_radius / norms).clamp(max=1.0)

        if self.dim == 1:
            return weights, means, covariances.clamp(constraints.sigma_min**2, constraints.sigma_max**2)

        covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariances)
        eigenvalues = eigenvalues.clamp(constraints.sigma_min**2, constraints.sigma_max**2)
        covariances = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        return weights, means, covariances

    def cholesky_factor(self, covariances):
        """Cholesky factors of constrained covariances, a square root when d = 1."""
        if self.dim == 1:
            return covariances.sqrt()
        return torch.linalg.cholesky(covariances)

    def initial_parameters(self, samples):
        """Deterministic initialization: component means at the sample quantiles."""
        n_components = self.n_components
        quantiles = (torch.arange(n_components, dtype=self.dtype, device=self.device) + 0.5) / n_components
        if self.dim == 1:
            means = torch.quantile(samples[:, 0], quantiles).unsqueeze(-1)
        else:
            # In more than one dimension, anchor on the samples sitting at those
            # quantiles of the first coordinate rather than on a per-coordinate
            # quantile, which need not be a plausible state.
            positions = (quantiles * (samples.shape[0] - 1)).round().long().clamp(0, samples.shape[0] - 1)
            means = samples[torch.argsort(samples[:, 0])][positions]

        centered = samples - samples.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(samples.shape[0], 1)
        covariances = covariance.expand(n_components, self.dim, self.dim).clone()
        weights = torch.full((n_components,), 1.0 / n_components, dtype=self.dtype, device=self.device)
        return self.constrain(weights, means, covariances)

    def fit(self, samples, warm_start=None, iterations=100, tolerance=1e-9):
        """The fitting rule R_K: constrained EM on the empirical law of the samples.

        The updates are the sample weights, means, and covariances of the EM
        responsibilities, so the fitted coordinate uses observed states only: no
        transition density and no derivative of the transition kernel. The
        previous fit is used as a warm start, and components are relabelled by
        increasing first mean coordinate so that the chart has a fixed labelling.
        """
        if self.n_components == 1:
            # A single component has a closed-form fixed point: every responsibility is one,
            # so EM returns the sample mean and covariance after a single step. Running the
            # responsibility loop for it costs about a hundred times its own arithmetic, and
            # the auxiliary block calls this once per shift per time step.
            mean = samples.mean(dim=0, keepdim=True)
            centered = samples - mean
            covariance = (centered.T @ centered / max(samples.shape[0], 1)).reshape(1, self.dim, self.dim)
            weights = torch.ones(1, dtype=samples.dtype, device=samples.device)
            weights, mean, covariance = self.constrain(weights, mean, covariance)
            return self.encode(weights, mean, self.cholesky_factor(covariance))

        if warm_start is None:
            weights, means, covariances = self.initial_parameters(samples)
        else:
            weights, means, scale_tril = self.decode(warm_start)
            covariances = scale_tril @ scale_tril.transpose(-1, -2)
            weights, means, covariances = self.constrain(weights, means, covariances)

        scale_tril = self.cholesky_factor(covariances)
        previous = None
        for _ in range(max(iterations, 1)):
            component = self.component_log_densities(samples, means, scale_tril)
            weighted = torch.log(weights) + component
            normalizer = torch.logsumexp(weighted, dim=-1)
            responsibilities = torch.exp(weighted - normalizer.unsqueeze(-1))

            counts = responsibilities.sum(dim=0).clamp_min(torch.finfo(samples.dtype).tiny)
            weights = counts / samples.shape[0]
            means = (responsibilities.T @ samples) / counts.unsqueeze(-1)
            difference = samples.unsqueeze(0) - means.unsqueeze(1)
            if self.dim == 1:
                squared = difference.squeeze(-1).square() * responsibilities.T
                covariances = (squared.sum(dim=-1) / counts).reshape(-1, 1, 1)
            else:
                covariances = torch.einsum("kni,knj,nk->kij", difference, difference, responsibilities)
                covariances = covariances / counts.reshape(-1, 1, 1)
            weights, means, covariances = self.constrain(weights, means, covariances)
            scale_tril = self.cholesky_factor(covariances)

            likelihood = normalizer.mean()
            if previous is not None and torch.abs(likelihood - previous) <= tolerance * (1.0 + torch.abs(previous)):
                break
            previous = likelihood

        order = torch.argsort(means[:, 0])
        return self.encode(weights[order], means[order], scale_tril[order])
