from dataclasses import dataclass, field

import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs import EnvBase


@dataclass
class KWParams:
    """Parameters for Kiyotaki-Wright model following the original paper"""

    n_agents: int = 300  # number of agents per type
    n_types: int = 3  # number of types of agents/goods

    # Each type i produces good i and consumes good i+1 (mod 3)
    # Storage costs: Type 1 has lowest cost storing good 1, etc.
    storage_costs: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.0, 2.0, 3.0],  # Type 0's storage costs
            [3.0, 0.0, 2.0],  # Type 1's storage costs
            [2.0, 3.0, 0.0],  # Type 2's storage costs
        ])
    )

    # Consumption values when agent gets their consumption good
    consumption_values: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 3.0])
    )

    # Production costs when agent produces their good
    production_costs: np.ndarray = field(
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

    def __init__(self, params: KWParams):
        super().__init__()
        self.params = params
        self._device = None
        self._batch_size = None

        # Define the observation and action spaces
        self.observation_spec = CompositeSpec(
            agent_type=DiscreteTensorSpec(n=params.n_types),
            inventory=DiscreteTensorSpec(n=params.n_types),
            shape=(params.n_types * params.n_agents,),
        )

        self.action_spec = DiscreteTensorSpec(
            n=2,  # accept or reject trade
            shape=(params.n_types * params.n_agents,),
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(params.n_types * params.n_agents,),
        )

        self.done_spec = CompositeSpec(
            done=DiscreteTensorSpec(n=2),
            shape=(1,),
        )

        # Initialize state
        self.reset()

    def _reset(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        """Reset the environment to initial state"""
        # Initialize agents with their production good
        self.inventories = torch.zeros(
            (self.params.n_types * self.params.n_agents,),
            dtype=torch.int64,
            device=self._device,
        )

        for i in range(self.params.n_types):
            start_idx = i * self.params.n_agents
            end_idx = (i + 1) * self.params.n_agents
            self.inventories[start_idx:end_idx] = i

        self.agent_types = torch.repeat_interleave(
            torch.arange(self.params.n_types, device=self._device), self.params.n_agents
        )

        # Convert storage costs to tensor
        if not hasattr(self, "storage_costs_tensor"):
            self.storage_costs_tensor = torch.tensor(
                self.params.storage_costs, device=self._device
            )
            self.consumption_values_tensor = torch.tensor(
                self.params.consumption_values, device=self._device
            )

        # Create observation tensordict
        td = tensordict.TensorDict(
            {
                "agent_type": self.agent_types,
                "inventory": self.inventories,
            },
            batch_size=[self.params.n_types * self.params.n_agents],
        )

        return td

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Execute one step in the environment"""
        actions = tensordict["action"]
        rewards = torch.zeros_like(self.inventories, dtype=torch.float32)

        # Randomly match agents
        perm = torch.randperm(len(self.inventories), device=self._device)
        matched_inventories = self.inventories[perm]
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
        self.inventories = matched_inventories[torch.argsort(perm)]

        # Create return tensordict
        td = tensordict.TensorDict(
            {
                "agent_type": self.agent_types,
                "inventory": self.inventories,
                "reward": rewards,
                "done": torch.zeros(1, dtype=torch.bool, device=self._device),
                "next": {
                    "agent_type": self.agent_types,
                    "inventory": self.inventories,
                },
            },
            batch_size=[self.params.n_types * self.params.n_agents],
        )

        return td

    def _set_batch_size(self, batch_size) -> None:
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def to(self, device):
        self._device = device
        if hasattr(self, "storage_costs_tensor"):
            self.storage_costs_tensor = self.storage_costs_tensor.to(device)
            self.consumption_values_tensor = self.consumption_values_tensor.to(device)
        if hasattr(self, "inventories"):
            self.inventories = self.inventories.to(device)
            self.agent_types = self.agent_types.to(device)
        return self

    def get_inventory_distribution(self) -> torch.Tensor:
        """Get the distribution of goods across all agents"""
        return torch.bincount(
            self.inventories, minlength=self.params.n_types
        ).float() / len(self.inventories)

    def _set_seed(self, seed: int | None = None) -> None:
        """Sets the seed for the environment's random number generator."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
