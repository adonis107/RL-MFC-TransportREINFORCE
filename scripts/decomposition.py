"""Decomposition of the residual error of a fixed-scale transport run.

For a fixed perturbation scale the estimator optimizes the perturbed objective
J^lambda rather than J.  The residual error of a run therefore splits into the
displacement of the perturbed optimum and the error left in optimizing it:

    target shift      |J^lambda(theta*_lambda) - J(theta*)|
    optimization gap  |J^lambda(theta_hat_lambda) - J^lambda(theta*_lambda)|
    true gap          |J(theta_hat_lambda) - J(theta*)|

Continuous-state benchmarks expose J^lambda in closed form.  Finite-state
benchmarks are evaluated by the exact population recursion, averaged over a
fixed common set of simplex draws, so every quantity below is deterministic
given the run.  theta*_lambda is always taken inside the control class that
also realizes theta*.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfc.environments.distribution import Distribution, DistributionConfig, DistributionPolicy
from mfc.environments.twostate import TwoState, TwoStateConfig
from mfc.environments.lq import LQ, LQConfig
from mfc.environments.portfolio import Portfolio, PortfolioConfig
from mfc.environments.randomizers import randomizer_for_run

N_DRAWS = 4096
RANDOMIZER_SEED = 20260923


class MixtureRandomizer:
    """Transport's randomizer: mu -> (1 - scale) mu + scale q, q = softmax(sigma N(0, I_{N-1}), 0)."""

    name = "mixture"

    def __init__(self, n_states, sigma, n_draws=N_DRAWS, dtype=torch.float64):
        generator = torch.Generator().manual_seed(RANDOMIZER_SEED)
        logits = torch.zeros(n_draws, n_states, dtype=dtype)
        logits[:, :-1] = sigma * torch.randn(n_draws, n_states - 1, dtype=dtype, generator=generator)
        self.draws = torch.softmax(logits, dim=-1)
        self.n_draws = n_draws

    def __call__(self, mu, t, scale):
        q = torch.roll(self.draws, (t * 7919) % self.n_draws, dims=0)
        return (1.0 - scale) * mu + scale * q


class LogitRandomizer:
    """MF-REINFORCE's randomizer: mu -> softmax(log mu + scale N(0, I_N))."""

    name = "logit"

    def __init__(self, n_states, n_draws=N_DRAWS, dtype=torch.float64):
        generator = torch.Generator().manual_seed(RANDOMIZER_SEED)
        self.noise = torch.randn(n_draws, n_states, dtype=dtype, generator=generator)
        self.n_draws = n_draws

    def __call__(self, mu, t, scale):
        z = torch.roll(self.noise, (t * 7919) % self.n_draws, dims=0)
        return torch.softmax(mu.clamp_min(1e-12).log() + scale * z, dim=-1)


def reward_matrix(env, laws):
    """(Q, S, A) reward of every state-action pair under each perturbed law."""
    n = laws.shape[0]
    rows = []
    for s in range(env.n_states):
        states = torch.full((n,), s, dtype=torch.long, device=env.device)
        columns = []
        for a in range(env.n_actions):
            actions = torch.full((n,), a, dtype=torch.long, device=env.device)
            columns.append(env.reward(states, laws, actions).expand(n))
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def terminal_matrix(env, laws):
    """(Q, S) terminal reward of every state under each perturbed law."""
    n = laws.shape[0]
    rows = []
    for s in range(env.n_states):
        states = torch.full((n,), s, dtype=torch.long, device=env.device)
        rows.append(env.terminal_reward(states, laws).expand(n))
    return torch.stack(rows, dim=-1)


def kernel_matrix(env, laws):
    """(Q, S, A, S) transition kernel under each perturbed law."""
    n = laws.shape[0]
    rows = []
    for s in range(env.n_states):
        states = torch.full((n,), s, dtype=torch.long, device=env.device)
        columns = []
        for a in range(env.n_actions):
            actions = torch.full((n,), a, dtype=torch.long, device=env.device)
            columns.append(env.transition(states, laws, actions).expand(n, env.n_states))
        rows.append(torch.stack(columns, dim=-2))
    return torch.stack(rows, dim=-3)


def action_probabilities(env, policy, t, laws):
    """(Q, S, A) action law of every state under each perturbed law."""
    n = laws.shape[0]
    rows = []
    for s in range(env.n_states):
        row = env.policy(policy, t, torch.tensor(s, device=env.device), laws)
        rows.append(row.expand(n, env.n_actions))
    return torch.stack(rows, dim=-2)


def discrete_value(env, policy, horizon, discount, initial_law, scale, randomizer):
    """Exact perturbed objective of a finite-state benchmark, under any randomizer.

    The population flow mu_t is the unperturbed mean-field flow of the policy;
    the controlled particle sees the randomized law at every step, with an
    independent draw per step, and its own marginal is propagated exactly.
    scale = 0 recovers the usual validation objective for either randomizer.
    """
    n_draws = randomizer.n_draws
    mu = initial_law
    marginal = initial_law.expand(n_draws, env.n_states)
    value = torch.zeros((), dtype=env.dtype, device=env.device)
    factor = 1.0
    for t in range(horizon):
        nu = randomizer(mu, t, scale)
        probabilities = action_probabilities(env, policy, t, nu)
        rewards = reward_matrix(env, nu)
        kernels = kernel_matrix(env, nu)
        step = (marginal.unsqueeze(-1) * probabilities * rewards).sum(dim=(-1, -2))
        value = value + factor * step.mean()
        marginal = torch.einsum("qs,qsa,qsax->qx", marginal, probabilities, kernels)
        base = action_probabilities(env, policy, t, mu.unsqueeze(0))
        base_kernel = kernel_matrix(env, mu.unsqueeze(0))
        mu = torch.einsum("s,qsa,qsax->x", mu, base, base_kernel)
        factor = factor * discount
    nu = randomizer(mu, horizon, scale)
    value = value + factor * (marginal * terminal_matrix(env, nu)).sum(dim=-1).mean()
    return value


def distribution_setup(metadata):
    config = DistributionConfig(device="cpu")
    env = Distribution(config)
    horizon = int(metadata["horizon"])

    def open_loop(logits):
        probabilities = torch.softmax(logits, dim=-1)
        return lambda t, mu: probabilities[min(int(t), horizon - 1)]

    return {
        "env": env,
        "horizon": horizon,
        "discount": 1.0,
        "initial_law": env.initial_distribution,
        "n_states": env.n_states,
        "class_shape": (horizon, env.n_states, env.n_actions),
        "parameterize": open_loop,
        "reference": torch.log(env.optimal_theta().clamp_min(1e-12)),
        "load": lambda folder: load_mlp(DistributionPolicy(config), folder),
    }


def twostate_setup(metadata):
    horizon = int(metadata["horizon"])
    env = TwoState(TwoStateConfig(T=horizon, device="cpu"))
    return {
        "env": env,
        "horizon": horizon,
        "discount": 1.0,
        "initial_law": env.initial_distribution,
        "n_states": env.n_states,
        "class_shape": (env.n_states,),
        "parameterize": lambda theta: theta,
        "reference": env.optimal_theta(),
        "load": lambda folder: torch.load(folder / "policy.pt", map_location="cpu", weights_only=False)["tensor"],
    }


def load_mlp(module, folder):
    blob = torch.load(Path(folder) / "policy.pt", map_location="cpu", weights_only=False)
    module.load_state_dict(blob["state_dict"])
    module.eval()
    return lambda t, mu: module(torch.tensor(float(t)), mu)


DISCRETE_SETUP = {
    "distribution": distribution_setup,
    "twostate": twostate_setup,
}


def discrete_evaluate(setup, raw, scale, randomizer):
    with torch.no_grad():
        return float(discrete_value(
            setup["env"], setup["parameterize"](raw), setup["horizon"],
            setup["discount"], setup["initial_law"], scale, randomizer,
        ))


def discrete_optimum(setup, scale, randomizer, steps=3000, lr=0.05):
    """theta*_lambda by gradient ascent inside the control class of theta*.

    The class is small but the objective is not concave in it, so the ascent is
    run from the unperturbed reference control as well as from the uninformative
    start, and the better of the two is kept.  At scale 0 this returns the
    benchmark's own optimum whenever the reference control already attains it.
    """
    starts = [torch.zeros(setup["class_shape"], dtype=setup["env"].dtype), setup["reference"].clone()]
    best = max(discrete_evaluate(setup, start, scale, randomizer) for start in starts)
    for start in starts:
        raw = start.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            value = discrete_value(
                setup["env"], setup["parameterize"](raw), setup["horizon"],
                setup["discount"], setup["initial_law"], scale, randomizer,
            )
            (-value).backward()
            optimizer.step()
        best = max(best, discrete_evaluate(setup, raw.detach(), scale, randomizer))
    return best


CONTINUOUS = {
    "lq": (lambda horizon: LQ(LQConfig(T=horizon, device="cpu")), -1.0),
    "portfolio": (lambda horizon: Portfolio(PortfolioConfig(T=horizon, device="cpu")), 1.0),
}


def continuous_value(env, theta, lambda_, sign, randomizer=None):
    return sign * env.objective(theta, lambda_=lambda_, randomizer=randomizer)


_CONTINUOUS_OPTIMUM_CACHE = {}


def continuous_optimum(env, lambda_, sign, randomizer=None, steps=20000, lr=0.01):
    """J^lambda at its own optimum, inside the control class that realizes theta*.

    Both continuous arms randomize the population mean the same way, so they ask
    for the same optimum at each scale; the ascent is run once per
    (benchmark, horizon, scale, randomizer) and reused.
    """
    key = (type(env).__name__, env.config.T, lambda_, sign, randomizer, steps, lr)
    if key in _CONTINUOUS_OPTIMUM_CACHE:
        return _CONTINUOUS_OPTIMUM_CACHE[key]
    theta = env.optimal_theta().clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        (-continuous_value(env, theta, lambda_, sign, randomizer)).backward()
        optimizer.step()
    with torch.no_grad():
        value = float(continuous_value(env, theta.detach(), lambda_, sign, randomizer))
    _CONTINUOUS_OPTIMUM_CACHE[key] = value
    return value


def load_theta(path):
    blob = torch.load(Path(path) / "policy.pt", map_location="cpu", weights_only=False)
    return blob["tensor"] if isinstance(blob, dict) and "tensor" in blob else blob


def continuous_rows(root, env, environment, sign, true_optimum, pattern):
    """Decomposition of one continuous-state arm against its own J^lambda.

    Each estimator randomizes the population its own way -- the mixture chart
    for Transport, the Gaussian manifold for Transport-Proba -- so the
    displacement and the optimization gap have to be taken against that
    estimator's perturbed objective, recovered from the run's metadata. The gap
    on J is the one quantity every arm shares, and it is evaluated at
    lambda = 0, where the randomizer does not enter.
    """
    directory = Path(root) / env
    found = {}
    for path in sorted(directory.glob(pattern)):
        stem = re.sub(r"_seed_\d+$", "", path.name)
        if "_K_" in stem and "_K_1_" not in stem:
            continue
        if not (path / "summary.json").exists():
            continue
        found.setdefault(stem, []).append(path)

    rows = []
    for stem, paths in sorted(found.items(), key=lambda kv: float(re.search(r"lambda_([0-9.]+)", kv[0]).group(1))):
        lambda_ = float(re.search(r"lambda_([0-9.]+)", stem).group(1))
        metadata = json.loads((paths[0] / "metadata.json").read_text())
        randomizer = randomizer_for_run(metadata)
        if randomizer is None:
            continue
        perturbed_optimum = continuous_optimum(environment, lambda_, sign, randomizer)
        perturbed, true = [], []
        for path in paths:
            theta = load_theta(path)
            with torch.no_grad():
                perturbed.append(abs(
                    float(continuous_value(environment, theta, lambda_, sign, randomizer)) - perturbed_optimum))
                true.append(abs(float(continuous_value(environment, theta, 0.0, sign)) - true_optimum))
        rows.append((lambda_, abs(perturbed_optimum - true_optimum),
                     float(np.mean(perturbed)), float(np.std(perturbed, ddof=1)),
                     float(np.mean(true)), float(np.std(true, ddof=1))))
    return rows


def runs(root, env, filter_text=None):
    directory = Path(root) / env
    found = {}
    for path in sorted(directory.glob("transport_*_seed_*")):
        stem = re.sub(r"_seed_\d+$", "", path.name)
        if filter_text is not None and filter_text not in stem:
            continue
        if "_K_" in stem and "_K_1_" not in stem:
            continue
        if not (path / "summary.json").exists():
            continue
        found.setdefault(stem, []).append(path)
    return found


def mfreinforce_decomposition(root, env, setup, true_optimum, horizon):
    """Decompose MF-REINFORCE against its own perturbed objective.

    MF-REINFORCE randomizes the population in logit space at its own scale
    epsilon, so its perturbed objective is not the transport J^lambda.  The
    displacement and the optimization gap are therefore taken with respect to
    that objective; the gap on J is the one quantity the two estimators share.
    """
    directory = Path(root) / env
    paths = sorted(p.parent for p in directory.glob(f"mfreinforce_*_T_{horizon}_exact_seed_*/summary.json"))
    if not paths:
        return None
    epsilon = float(json.loads((paths[0] / "metadata.json").read_text())["algorithm_config"]["perturbation_eta"])
    randomizer = LogitRandomizer(setup["n_states"], dtype=setup["env"].dtype)
    perturbed_optimum = discrete_optimum(setup, epsilon, randomizer)
    perturbed, true = [], []
    for path in paths:
        policy = setup["load"](path)
        with torch.no_grad():
            perturbed.append(abs(float(discrete_value(
                setup["env"], policy, setup["horizon"], setup["discount"],
                setup["initial_law"], epsilon, randomizer)) - perturbed_optimum))
            true.append(abs(float(discrete_value(
                setup["env"], policy, setup["horizon"], setup["discount"],
                setup["initial_law"], 0.0, randomizer)) - true_optimum))
    return (epsilon, abs(perturbed_optimum - true_optimum),
            float(np.mean(perturbed)), float(np.std(perturbed, ddof=1)),
            float(np.mean(true)), float(np.std(true, ddof=1)))


def reinforce_gap(root, env, optimum, horizon):
    """REINFORCE baseline at the same horizon as the transport runs being reported."""
    directory = Path(root) / env
    values = [json.loads(p.read_text())["last_validation_objective"]
              for p in sorted(directory.glob(f"reinforce_*_T_{horizon}_*_seed_*/summary.json"))]
    if not values:
        return None
    gaps = [abs(v - optimum) for v in values]
    return float(np.mean(gaps)), float(np.std(gaps, ddof=1))


def report(env, root, filter_text=None):
    found = runs(root, env, filter_text)
    if not found:
        print(f"{env}: no runs under {root}")
        return None

    rows, mfreinforce, gaussian = [], None, []
    if env in CONTINUOUS:
        factory, sign = CONTINUOUS[env]
        horizon = int(json.loads((next(iter(found.values()))[0] / "metadata.json").read_text())["horizon"])
        environment = factory(horizon)
        with torch.no_grad():
            true_optimum = float(continuous_value(environment, environment.optimal_theta(), 0.0, sign))
        rows = continuous_rows(root, env, environment, sign, true_optimum, "transport_*_seed_*")
        gaussian = continuous_rows(root, env, environment, sign, true_optimum, "gaussian_*_seed_*")
        baseline = reinforce_gap(root, env, true_optimum, horizon)
    else:
        metadata = json.loads((next(iter(found.values()))[0] / "metadata.json").read_text())
        setup = DISCRETE_SETUP[env](metadata)
        horizon = int(metadata["horizon"])
        sigma = metadata["algorithm_config"]["simplex_sigma"]
        randomizer = MixtureRandomizer(setup["n_states"], float(sigma), dtype=setup["env"].dtype)
        true_optimum = discrete_optimum(setup, 0.0, randomizer)
        for stem, paths in sorted(found.items(), key=lambda kv: float(re.search(r"lambda_([0-9.]+)", kv[0]).group(1))):
            lambda_ = float(re.search(r"lambda_([0-9.]+)", stem).group(1))
            perturbed_optimum = discrete_optimum(setup, lambda_, randomizer)
            perturbed, true = [], []
            for path in paths:
                policy = setup["load"](path)
                with torch.no_grad():
                    perturbed.append(abs(float(discrete_value(
                        setup["env"], policy, setup["horizon"], setup["discount"],
                        setup["initial_law"], lambda_, randomizer)) - perturbed_optimum))
                    true.append(abs(float(discrete_value(
                        setup["env"], policy, setup["horizon"], setup["discount"],
                        setup["initial_law"], 0.0, randomizer)) - true_optimum))
            rows.append((lambda_, abs(perturbed_optimum - true_optimum),
                         float(np.mean(perturbed)), float(np.std(perturbed, ddof=1)),
                         float(np.mean(true)), float(np.std(true, ddof=1))))
        mfreinforce = mfreinforce_decomposition(root, env, setup, true_optimum, horizon)
        baseline = reinforce_gap(root, env, true_optimum, horizon)

    print(f"\n== {env}  (root={root}, filter={filter_text}, J(theta*)={true_optimum:.6f})")
    print(f"{'lambda':>9} {'shift':>12} {'opt gap':>22} {'true gap':>22}"
          "   (rows prefixed P are the Gaussian-manifold arm)")
    for lambda_, shift, opt, opt_sd, true, true_sd in rows:
        print(f"{lambda_:>9.4g} {shift:>12.4f} {opt:>12.4f}+-{opt_sd:<8.4f} {true:>12.4f}+-{true_sd:<8.4f}")
    for lambda_, shift, opt, opt_sd, true, true_sd in gaussian:
        print(f"{'P ' + format(lambda_, '.4g'):>9} {shift:>12.4f} "
              f"{opt:>12.4f}+-{opt_sd:<8.4f} {true:>12.4f}+-{true_sd:<8.4f}")
    if mfreinforce is not None:
        eps, shift, opt, opt_sd, true, true_sd = mfreinforce
        print(f"{'MF eps=' + format(eps, '.4g'):>9} {shift:>12.4f} "
              f"{opt:>12.4f}+-{opt_sd:<8.4f} {true:>12.4f}+-{true_sd:<8.4f}")
    if baseline is not None:
        print(f"{'REINFORCE':>9} {'---':>12} {'---':>22} {baseline[0]:>12.4f}+-{baseline[1]:<8.4f}")
    return env, rows, gaussian, baseline


DISPLAY = {
    "lq": "Linear--quadratic",
    "portfolio": "Portfolio",
    "distribution": "Distribution",
    "twostate": "Two-state",
}


def tex_table(collected, path):
    """LaTeX form of the decomposition, both continuous arms side by side."""
    digits = 4
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Benchmark & Estimator & $\\lambda$ & "
        "$|J^\\lambda(\\theta_\\lambda^\\star)-J(\\theta^\\star)|$ & "
        "$|J^\\lambda(\\widehat\\theta_\\lambda)-J^\\lambda(\\theta_\\lambda^\\star)|$ & "
        "$|J(\\widehat\\theta_\\lambda)-J(\\theta^\\star)|$ \\\\",
        "\\midrule",
    ]
    for index, (env, rows, gaussian, baseline) in enumerate(collected):
        first = True
        for name, block in (("Transport", rows), ("Transport-Proba", gaussian)):
            for position, (lambda_, shift, opt, opt_sd, true, true_sd) in enumerate(block):
                lines.append(" & ".join([
                    DISPLAY.get(env, env) if first else "",
                    name if position == 0 else "",
                    f"${lambda_:g}$",
                    f"${shift:.{digits}f}$",
                    f"${opt:.{digits}f}\\pm{opt_sd:.{digits}f}$",
                    f"${true:.{digits}f}\\pm{true_sd:.{digits}f}$",
                ]) + " \\\\")
                first = False
            if block:
                lines.append("\\addlinespace[2pt]")
        if baseline is not None:
            lines.append(" & ".join([
                "", "REINFORCE", "---", "---", "---",
                f"${baseline[0]:.{digits}f}\\pm{baseline[1]:.{digits}f}$",
            ]) + " \\\\")
        if index != len(collected) - 1:
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Residual error of each run, split into the displacement of the perturbed "
        "optimum and the error left in optimizing it. Here $\\theta_\\lambda^\\star$ maximizes "
        "$J^\\lambda$ in the control class that realizes $\\theta^\\star$, and "
        "$\\widehat\\theta_\\lambda$ is the parameter the run returns. Dispersion is over five "
        "seeds.}",
        "\\label{tab:continuous-decomposition}",
        "\\end{table}",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmarks", nargs="*", default=["distribution", "portfolio", "lq", "twostate"])
    parser.add_argument("--root", default="results")
    parser.add_argument("--filter", default=None)
    parser.add_argument("--tex", default=None, help="also write the table to this path")
    args = parser.parse_args()
    collected = []
    for env in args.benchmarks:
        collected.append(report(env, args.root, args.filter))
    if args.tex:
        tex_table([entry for entry in collected if entry is not None], args.tex)


if __name__ == "__main__":
    main()
