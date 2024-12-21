from typing import Any

import pytest
import pytest_check as check  # type: ignore
import torch

from src.multiagent_ppo import (
    MultiAgentMLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    TensorDictModule,
    TransformedEnv,
    VmasEnv,
    render_policy,
    train_vmas,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="module")
def test_env(device: torch.device) -> VmasEnv:
    env = VmasEnv(
        scenario="navigation",
        num_envs=2,
        continuous_actions=True,
        max_steps=10,
        device=device,
        n_agents=2,
    )
    return env


@pytest.fixture(scope="module")
def test_policy(test_env: VmasEnv, device: torch.device) -> ProbabilisticActor:
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=test_env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * test_env.action_spec.shape[-1],
            n_agents=test_env.n_agents,
            centralised=False,
            share_params=True,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=test_env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[test_env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": torch.tensor([-1.0, -1.0]),
            "high": torch.tensor([1.0, 1.0]),
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return policy


@pytest.mark.timeout(300)
def test_environment_creation(test_env: VmasEnv) -> None:
    """Test environment creation and basic properties"""
    check.is_instance(test_env, VmasEnv)
    check.equal(test_env.n_agents, 2)
    check.is_true(test_env.continuous_actions)
    check.equal(test_env.max_steps, 10)


@pytest.mark.fast
def test_policy_creation(test_policy: ProbabilisticActor) -> None:
    """Test policy network creation and properties"""
    check.is_instance(
        test_policy,
        ProbabilisticActor,
        "Policy should be an instance of ProbabilisticActor",
    )

    # Check for required attributes and methods
    required_attributes = {
        "module": "Policy network module",
        "get_dist": "Distribution getter method",
        "random_sample": "Random sampling method",
        "log_prob": "Log probability method",
    }

    for attr, description in required_attributes.items():
        check.is_true(
            hasattr(test_policy, attr),
            f"Missing {description} ({attr}). "
            f"Available attributes: {[a for a in dir(test_policy) if not a.startswith('_')]}",
        )


@pytest.mark.timeout(300)
def test_policy_forward_pass(
    test_env: VmasEnv, test_policy: ProbabilisticActor
) -> None:
    """Test policy forward pass"""
    tensordict = test_env.reset()
    with torch.no_grad():
        output_tensordict = test_policy(tensordict)

    check.is_true(test_env.action_key in output_tensordict)
    check.equal(
        output_tensordict[test_env.action_key].shape[-1], test_env.action_spec.shape[-1]
    )


@pytest.mark.timeout(300)
def test_environment_step(test_env: VmasEnv, test_policy: ProbabilisticActor) -> None:
    """Test environment stepping"""
    tensordict = test_env.reset()
    with torch.no_grad():
        tensordict = test_policy(tensordict)
    next_tensordict = test_env.step(tensordict)

    # Print full tensordict structure for debugging
    print("\nTensorDict structure:")
    for key in next_tensordict.keys(True):  # type: ignore
        print(f"Key: {key}")
        if key in next_tensordict:
            print(f"Shape: {next_tensordict[key].shape}")

    # Check for reward in the correct location
    check.is_true(
        ("next", "agents", "reward") in next_tensordict,
        f"Reward key not found at ('next', 'agents', 'reward'). "
        f"Available keys: {list(next_tensordict.keys(True))}",  # type: ignore
    )


@pytest.mark.slow
def test_training_loop() -> None:
    """Test the training loop with minimal settings"""
    # Override global constants for faster testing
    import src.multiagent_ppo as mp

    mp.frames_per_batch = 200
    mp.n_iters = 2
    mp.num_epochs = 2
    mp.minibatch_size = 100

    env, policy = train_vmas()
    check.is_instance(env, TransformedEnv)
    check.is_instance(policy, ProbabilisticActor)


@pytest.mark.slow
def test_render_policy(
    test_env: VmasEnv, test_policy: ProbabilisticActor, mocker: Any
) -> None:
    """Test render policy function with mocked rendering"""
    # Mock the render method to avoid actual rendering
    mocker.patch.object(test_env, "render")
    mocker.patch.object(test_env, "close")

    # Create a TransformedEnv wrapper around test_env
    transformed_env = TransformedEnv(test_env)

    # Ensure correct shape for observation
    tensordict = transformed_env.reset()
    print(f"\nObservation shape: {tensordict[('agents', 'observation')].shape}")

    try:
        render_policy(transformed_env, test_policy, num_episodes=1)
    except ValueError as e:
        pytest.fail(f"Shape mismatch in render_policy: {e!s}")

    check.is_true(test_env.render.called)
    check.is_true(test_env.close.called)


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_device_compatibility(device_name: str) -> None:
    """Test compatibility with different devices"""
    device = torch.device(device_name)
    env = VmasEnv(
        scenario="navigation",
        num_envs=1,
        continuous_actions=True,
        max_steps=10,
        device=device,
        n_agents=2,
    )
    check.equal(str(env.device), device_name)


@pytest.mark.fast
def test_error_handling() -> None:
    """Test error handling for invalid inputs"""
    with pytest.raises(AssertionError, match=".*scenario not found.*"):
        VmasEnv(
            scenario="non_existent_scenario",
            num_envs=1,
            continuous_actions=True,
            max_steps=10,
            device=torch.device("cpu"),
            n_agents=2,
        )


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=src.multiagent_ppo",
        "--cov-report=term-missing",
    ])
