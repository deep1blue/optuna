from __future__ import annotations

import binascii
from collections.abc import Callable
from unittest import mock

import pytest

import optuna


MIN_RESOURCE = 1
MAX_RESOURCE = 16
REDUCTION_FACTOR = 2
N_BRACKETS = 4
EARLY_STOPPING_RATE_LOW = 0
EARLY_STOPPING_RATE_HIGH = 3
N_REPORTS = 10
EXPECTED_N_TRIALS_PER_BRACKET = 10


def test_hyperband_pruner_intermediate_values() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )

    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(N_REPORTS):
            trial.report(i, step=i)

        return 1.0

    study.optimize(objective, n_trials=N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET)

    trials = study.trials
    assert len(trials) == N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET


def test_bracket_study() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
    bracket_study = pruner._create_bracket_study(study, 0)

    with pytest.raises(AttributeError):
        bracket_study.optimize(lambda *args: 1.0)

    with pytest.raises(AttributeError):
        bracket_study.set_user_attr("abc", 100)

    for attr in ("user_attrs", "system_attrs"):
        with pytest.raises(AttributeError):
            getattr(bracket_study, attr)

    with pytest.raises(AttributeError):
        bracket_study.trials_dataframe()

    bracket_study.get_trials()
    bracket_study.direction
    bracket_study._storage
    bracket_study._study_id
    bracket_study.pruner
    bracket_study.study_name
    # As `_BracketStudy` is defined inside `HyperbandPruner`,
    # we cannot do `assert isinstance(bracket_study, _BracketStudy)`.
    # This is why the below line is ignored by mypy checks.
    bracket_study._bracket_id  # type: ignore


def test_hyperband_max_resource_is_auto() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(N_REPORTS):
            trial.report(1.0, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return 1.0

    study.optimize(objective, n_trials=N_BRACKETS * EXPECTED_N_TRIALS_PER_BRACKET)

    assert N_REPORTS == pruner._max_resource


def test_hyperband_max_resource_value_error() -> None:
    with pytest.raises(ValueError):
        _ = optuna.pruners.HyperbandPruner(max_resource="not_appropriate")


@pytest.mark.parametrize(
    "sampler_init_func",
    [
        lambda: optuna.samplers.RandomSampler(),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=1)),
        (
            lambda: optuna.samplers.GridSampler(
                search_space={"value": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]}
            )
        ),
        (lambda: optuna.samplers.CmaEsSampler(n_startup_trials=1)),
    ],
)
def test_hyperband_filter_study(
    sampler_init_func: Callable[[], optuna.samplers.BaseSampler],
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("value", 0.0, 1.0)

    n_trials = 8
    n_brackets = 4
    expected_n_trials_per_bracket = n_trials // n_brackets
    with mock.patch(
        "optuna.pruners.HyperbandPruner._get_bracket_id",
        new=mock.Mock(side_effect=lambda study, trial: trial.number % n_brackets),
    ):
        for method_name in [
            "infer_relative_search_space",
            "sample_relative",
            "sample_independent",
        ]:
            sampler = sampler_init_func()
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=MIN_RESOURCE,
                max_resource=MAX_RESOURCE,
                reduction_factor=REDUCTION_FACTOR,
            )
            with mock.patch(
                f"optuna.samplers.{sampler.__class__.__name__}.{method_name}",
                wraps=getattr(sampler, method_name),
            ) as method_mock:
                study = optuna.study.create_study(sampler=sampler, pruner=pruner)
                study.optimize(objective, n_trials=n_trials)
                args = method_mock.call_args[0]
                study = args[0]
                trials = study.get_trials()
                assert len(trials) == expected_n_trials_per_bracket


@pytest.mark.parametrize(
    "pruner_init_func",
    [
        lambda: optuna.pruners.NopPruner(),
        lambda: optuna.pruners.MedianPruner(),
        lambda: optuna.pruners.ThresholdPruner(lower=0.5),
        lambda: optuna.pruners.SuccessiveHalvingPruner(),
    ],
)
def test_hyperband_no_filter_study(
    pruner_init_func: Callable[[], optuna.pruners.BasePruner],
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("value", 0.0, 1.0)

    n_trials = 10
    for method_name in [
        "infer_relative_search_space",
        "sample_relative",
        "sample_independent",
    ]:
        sampler = optuna.samplers.RandomSampler()
        pruner = pruner_init_func()
        with mock.patch(
            f"optuna.samplers.{sampler.__class__.__name__}.{method_name}",
            wraps=getattr(sampler, method_name),
        ) as method_mock:
            study = optuna.study.create_study(sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=n_trials)
            args = method_mock.call_args[0]
            study = args[0]
            trials = study.get_trials()
            assert len(trials) == n_trials


@pytest.mark.parametrize(
    "sampler_init_func",
    [
        lambda: optuna.samplers.RandomSampler(),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=1)),
        (
            lambda: optuna.samplers.GridSampler(
                search_space={"value": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
            )
        ),
        (lambda: optuna.samplers.CmaEsSampler(n_startup_trials=1)),
    ],
)
def test_hyperband_no_call_of_filter_study_in_should_prune(
    sampler_init_func: Callable[[], optuna.samplers.BaseSampler],
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        with mock.patch("optuna.pruners._filter_study") as method_mock:
            for i in range(N_REPORTS):
                trial.report(i, step=i)
                if trial.should_prune():
                    method_mock.assert_not_called()
                    raise optuna.TrialPruned()
                else:
                    method_mock.assert_not_called()

        return 1.0

    sampler = sampler_init_func()
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.study.create_study(sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=10)


def test_incompatibility_between_bootstrap_count_and_auto_max_resource() -> None:
    with pytest.raises(ValueError):
        optuna.pruners.HyperbandPruner(max_resource="auto", bootstrap_count=1)


def test_hyperband_pruner_and_grid_sampler() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )

    search_space = {"x": [-50, 0, 50], "y": [-99, 0, 99]}
    sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.study.create_study(sampler=sampler, pruner=pruner)

    def objective(trial: optuna.trial.Trial) -> float:
        for i in range(N_REPORTS):
            trial.report(i, step=i)

        x = trial.suggest_float("x", -100, 100)
        y = trial.suggest_int("y", -100, 100)
        return x**2 + y**2

    study.optimize(objective, n_trials=10)

    trials = study.trials
    assert len(trials) == 9


def test_hyperband_pruner_and_brute_force_sampler() -> None:
    def objective(trial: optuna.trial.Trial) -> int:
        x = trial.suggest_int("x", 0, 2)
        y = trial.suggest_int("y", 0, 2)
        for i in range(10):
            trial.report(step=i, value=i * (x + y) / 10)
            if trial.should_prune():
                raise optuna.TrialPruned
        return x + y

    sampler = optuna.samplers.BruteForceSampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(sampler=sampler, pruner=pruner)
    study.optimize(objective)

    trials = study.trials
    assert len(trials) == 9


def _reference_bracket_id(
    pruner: optuna.pruners.HyperbandPruner, study_name: str, number: int
) -> int:
    n = binascii.crc32(f"{study_name}_{number}".encode()) % pruner._total_trial_allocation_budget
    for bracket_id, budget in enumerate(pruner._trial_allocation_budgets):
        n -= budget
        if n < 0:
            return bracket_id
    assert False


@pytest.mark.parametrize(
    "min_resource,max_resource,reduction_factor",
    [(1, 16, 2), (1, 81, 3), (2, 100, 4), (1, 5, 3), (1, 1, 3), (3, 1000, 5)],
)
@pytest.mark.parametrize("study_name", ["study", "a_b_c", "\u30c6\u30b9\u30c8", ""])
def test_get_bracket_ids_matches_reference(
    min_resource: int, max_resource: int, reduction_factor: int, study_name: str
) -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=min_resource, max_resource=max_resource, reduction_factor=reduction_factor
    )
    study = optuna.create_study(study_name=study_name or None, pruner=pruner)
    pruner._try_initialization(study)
    assert len(pruner._pruners) > 0

    numbers = list(range(2000)) + [10**6, 2**31, 2**40]
    bracket_ids = pruner._get_bracket_ids(study.study_name, numbers)

    assert len(bracket_ids) == len(numbers)
    assert bracket_ids == [_reference_bracket_id(pruner, study.study_name, n) for n in numbers]
    assert set(bracket_ids) <= set(range(len(pruner._pruners)))


def test_get_bracket_ids_before_initialization() -> None:
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=16)
    assert pruner._get_bracket_ids("study", [0, 1, 2]) == [0, 0, 0]
    assert pruner._get_bracket_ids("study", []) == []
    assert pruner._get_bracket_ids("study", iter([5, 6])) == [0, 0]


def test_get_bracket_id_consistent_with_get_bracket_ids() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
    study.add_trials([optuna.trial.create_trial(value=0.0) for _ in range(50)])
    pruner._try_initialization(study)

    trials = study.get_trials(deepcopy=False)
    assert [pruner._get_bracket_id(study, t) for t in trials] == pruner._get_bracket_ids(
        study.study_name, [t.number for t in trials]
    )


def test_bracket_study_get_trials_partitions_trials() -> None:
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RESOURCE, max_resource=MAX_RESOURCE, reduction_factor=REDUCTION_FACTOR
    )
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)
    states = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    ]
    for i in range(120):
        study.add_trial(
            optuna.trial.create_trial(
                state=states[i % 3], value=0.0 if states[i % 3] == states[0] else None
            )
        )
    pruner._try_initialization(study)

    collected: list[int] = []
    for bracket_id in range(len(pruner._pruners)):
        bracket_study = pruner._create_bracket_study(study, bracket_id)
        for deepcopy in (True, False):
            numbers = [t.number for t in bracket_study.get_trials(deepcopy=deepcopy)]
            assert numbers == [
                t.number
                for t in study.get_trials(deepcopy=False)
                if _reference_bracket_id(pruner, study.study_name, t.number) == bracket_id
            ]
        collected += [t.number for t in bracket_study.get_trials()]

        complete_numbers = [
            t.number for t in bracket_study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        ]
        assert all(
            study.trials[n].state == optuna.trial.TrialState.COMPLETE for n in complete_numbers
        )

    assert sorted(collected) == list(range(120))
