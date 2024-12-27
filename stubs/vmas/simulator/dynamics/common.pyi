#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import abc
from abc import ABC

from torch import Tensor

from vmas.simulator.core import Agent

class Dynamics(ABC):
    def __init__(
        self,
    ) -> None:
        self._agent: Agent | None = None
    def reset(self, index: Tensor | int | None = None) -> None: ...
    def zero_grad(self) -> None: ...
    @property
    def agent(self) -> Agent: ...
    @agent.setter
    def agent(self, value: Agent) -> None: ...
    def check_and_process_action(self) -> None: ...
    @property
    @abc.abstractmethod
    def needed_action_size(self) -> int: ...
    @abc.abstractmethod
    def process_action(self) -> None: ...
