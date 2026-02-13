from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR

# import logging


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup scheduler."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int) -> None:
        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Get the learning rate at the current step."""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """Get the LR factor at the current step."""
        lr_factor = 1.0
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


class STLRScheduler(LambdaLR):
    """Slanted triangular learning rates (STLR) Scheduler. Only Linear. Adapted for LambdaLR."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        ratio: float = 0.03125,
        max_lr: float = 0.00005,
        full_cycle_length: int = 7850,
        cycle_midpoint_fraction: float = 0.1,
    ):
        self.min_lr = ratio * max_lr
        self.max_lr = max_lr
        self.ratio = ratio
        self.full_cycle_length = full_cycle_length
        self.cycle_midpoint_fraction = cycle_midpoint_fraction
        super().__init__(optimizer, self.one_cycle_lr_lambda, last_epoch)

    def one_cycle_lr_lambda(self, step: int) -> float:
        """Helper function for STLRScheduler. Has to return a learning rate factor."""
        step_in_cycle = step % self.full_cycle_length
        cycle_midpoint = int(self.full_cycle_length * self.cycle_midpoint_fraction)

        if step_in_cycle < cycle_midpoint:
            lr_factor = step_in_cycle / cycle_midpoint  # Linear increase from 0 to 1
        else:
            lr_factor = (self.full_cycle_length - step_in_cycle) / (
                self.full_cycle_length - cycle_midpoint
            )  # Linear decrease from 1 to 0
        lr_factor = self.ratio + (1 - self.ratio) * lr_factor

        return lr_factor


class STLRSchedulerEpochBased(LambdaLR):
    """Slanted triangular learning rates (STLR) Scheduler. Epoch Based Only Linear. Adapted for LambdaLR."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lrs_list: list,
        last_epoch: int = -1,
        num_epochs: int = 10,
        ratio: float = 0.03125,
        full_cycle_length: int = 7850,
        cycle_midpoint_fraction: float = 0.1,
    ):
        self.ratio = ratio
        self.max_lrs_list = max_lrs_list
        self.num_epochs = num_epochs
        self.full_cycle_length = full_cycle_length
        self.cycle_midpoint_fraction = cycle_midpoint_fraction
        self.global_step = 0
        super().__init__(optimizer, self.one_cycle_lr_lambda, last_epoch)

    def one_cycle_lr_lambda(self, step: int) -> float:
        """Helper function for STLRScheduler. Has to return a learning rate factor."""
        # logging.info(f"Local step: {local_step}")
        # logging.info(f"Global step: {self.global_step}")
        # step = self.global_step
        epoch = (step // self.full_cycle_length) % self.num_epochs
        try:
            new_lr = self.max_lrs_list[epoch]
        except IndexError:
            new_lr = self.max_lrs_list[-1]
        initial_lr = self.max_lrs_list[0]
        lr_multiplier = new_lr / initial_lr  # Epoch based multiplier

        step_in_cycle = step % self.full_cycle_length
        cycle_midpoint = int(self.full_cycle_length * self.cycle_midpoint_fraction)

        if step_in_cycle < cycle_midpoint:
            lr_factor = step_in_cycle / cycle_midpoint  # Linear increase from 0 to 1
        else:
            lr_factor = (self.full_cycle_length - step_in_cycle) / (
                self.full_cycle_length - cycle_midpoint
            )  # Linear decrease from 1 to 0
        lr_factor = (self.ratio + (1 - self.ratio) * lr_factor) * lr_multiplier
        # self.global_step += 1
        return lr_factor
