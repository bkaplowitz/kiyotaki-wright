from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch as th
from tensordict import TensorDict, TensorDictBase  # type: ignore[import-untyped]
from torch import Tensor
from torchrl.data import (  # type: ignore[import-untyped]
    Categorical,
    Composite,
    UnboundedContinuous,
)
from torchrl.envs import EnvBase  # type: ignore[import-untyped]


@dataclass
class KWParams:
    """Parameters for Kiyotaki-Wright model following the original paper"""

    n_agents: int = 300  # number of agents per type
    n_types: int = 3  # number of types of agents/goods

    # Each type i produces good i and consumes good i+1 (mod 3)
    # Storage costs: Type 1 has lowest cost storing good 1, etc.
    storage_costs: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([
            [0.0, 2.0, 3.0],  # Type 0's storage costs
            [3.0, 0.0, 2.0],  # Type 1's storage costs
            [2.0, 3.0, 0.0],  # Type 2's storage costs
        ])
    )

    # Consumption values when agent gets their consumption good
    consumption_values: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([3.0, 3.0, 3.0])
    )

    # Production costs when agent produces their good
    production_costs: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )

    # Trading costs
    trading_cost: float = 0.0

    # Discount factor
    discount_factor: float = 0.95


class KWEnvironment(EnvBase):
    """
    Kiyotaki-Wright environment implementing the monetary search model
    following the original paper specifications, using TorchRL's EnvBase
    """

    params: KWParams
    _device: th.device | None
    _batch_size: th.Size
    inventories: Tensor
    agent_types: Tensor
    storage_costs_tensor: Tensor
    consumption_values_tensor: Tensor

    def __init__(self, params: KWParams):
        total_agents = params.n_types * params.n_agents
        self.params = params
        self._device = None
        self._batch_size = th.Size([total_agents])
        super().__init__()

        total_agents = params.n_types * params.n_agents

        # Define the observation and action spaces using updated specs
        self.observation_spec = Composite({
            "agent_type": Categorical(
                n=params.n_types,
            ),
            "inventory": Categorical(
                n=params.n_types,
            ),
        }).expand(total_agents)  # Expand to handle all agents

        self.action_spec = Categorical(
            n=2,  # accept or reject trade
        ).expand(total_agents)

        self.reward_spec = UnboundedContinuous(
            shape=th.Size([]),
        ).expand(total_agents)

        self.done_spec = Composite({
            "done": Categorical(
                n=2,
            ),
        }).expand(total_agents)

        # Initialize state
        self.reset()

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Reset the environment to initial state"""
        # Initialize agents with their production good
        self.inventories = th.zeros(
            (self.params.n_types * self.params.n_agents,),
            dtype=th.int64,
            device=self._device,
        )

        for i in range(self.params.n_types):
            start_idx = i * self.params.n_agents
            end_idx = (i + 1) * self.params.n_agents
            self.inventories[start_idx:end_idx] = i

        self.agent_types = th.repeat_interleave(
            th.arange(self.params.n_types, device=self._device), self.params.n_agents
        )

        # Convert storage costs to tensor
        if not hasattr(self, "storage_costs_tensor"):
            self.storage_costs_tensor = th.tensor(
                self.params.storage_costs, device=self._device
            )
            self.consumption_values_tensor = th.tensor(
                self.params.consumption_values, device=self._device
            )

        # Create observation tensordict
        td = TensorDict(  # type: ignore[union-attr]
            {
                "agent_type": self.agent_types,
                "inventory": self.inventories,
            },
            batch_size=th.Size([self.params.n_types * self.params.n_agents]),
        )

        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step in the environment"""
        actions = tensordict.get("action")
        rewards = th.zeros_like(self.inventories, dtype=th.float32)

        # Randomly match agents
        perm = th.randperm(len(self.inventories), device=self._device)
        matched_inventories = self.inventories
        matched_types = self.agent_types[perm]
        matched_actions = actions[perm]

        # Process trades in pairs
        for i in range(0, len(self.inventories), 2):
            if i + 1 >= len(self.inventories):
                break

            # Get agent types and inventories
            type1, type2 = matched_types[i], matched_types[i + 1]
            inv1, inv2 = matched_inventories[i], matched_inventories[i + 1]
            action1, action2 = matched_actions[i], matched_actions[i + 1]

            # Check if both agents want to trade
            if action1 and action2:
                # Execute trade
                matched_inventories[i] = inv2
                matched_inventories[i + 1] = inv1

                # Calculate rewards
                # Type i consumes good (i+1) mod 3
                if inv2 == (type1 + 1) % self.params.n_types:
                    rewards[perm[i]] += self.consumption_values_tensor[type1]
                if inv1 == (type2 + 1) % self.params.n_types:
                    rewards[perm[i + 1]] += self.consumption_values_tensor[type2]

                # Apply trading costs
                rewards[perm[i]] -= self.params.trading_cost
                rewards[perm[i + 1]] -= self.params.trading_cost

            # Apply storage costs
            rewards[perm[i]] -= self.storage_costs_tensor[type1, inv1]
            rewards[perm[i + 1]] -= self.storage_costs_tensor[type2, inv2]

        # Update inventories
        self.inventories = matched_inventories[th.argsort(perm)]

        # Return tensordict with all required fields
        return TensorDict(
            {
                # Current state
                "agent_type": self.agent_types,
                "inventory": self.inventories,
                "reward": rewards,
                "terminated": th.zeros(
                    (self.params.n_types * self.params.n_agents,),
                    dtype=th.bool,
                    device=self._device,
                ),
                "done": th.zeros(
                    (self.params.n_types * self.params.n_agents,),
                    dtype=th.bool,
                    device=self._device,
                ),
                # Next state using nested keys
                ("next", "agent_type"): self.agent_types,
                ("next", "inventory"): self.inventories,
            },
            batch_size=th.Size([self.params.n_types * self.params.n_agents]),
            device=self._device,
        )

    def _set_batch_size(self, batch_size: int) -> None:
        self._batch_size = th.Size([batch_size])

    @property
    def batch_size(self) -> th.Size | None:
        """Return the current batch size of the environment."""
        return self._batch_size

    def to(self, device: th.device) -> "KWEnvironment":
        """Move the environment to the specified device.

        Args:
            device: The torch device to move the environment to

        Returns:
            The environment instance
        """
        self._device = device
        if hasattr(self, "storage_costs_tensor"):
            self.storage_costs_tensor = self.storage_costs_tensor.to(device)
            self.consumption_values_tensor = self.consumption_values_tensor.to(device)
        if hasattr(self, "inventories"):
            self.inventories = self.inventories.to(device)
            self.agent_types = self.agent_types.to(device)
        return self

    def get_inventory_distribution(self) -> Tensor:
        """Get the distribution of goods across all agents"""
        return th.bincount(
            self.inventories, minlength=self.params.n_types
        ).float() / len(self.inventories)

    def _set_seed(self, seed: int | None = None) -> None:
        """Sets the seed for the environment's random number generator."""
        if seed is not None:
            th.manual_seed(seed)
            if th.cuda.is_available():
                th.cuda.manual_seed_all(seed)
            np.random.seed(seed)
