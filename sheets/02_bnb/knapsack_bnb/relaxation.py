"""
Relaxation Module

In branch-and-bound, a relaxation of the original 0/1 knapsack yields an upper bound
on the best feasible solution within a branch. If this bound does not exceed your
current best feasible solution, you can prune that branch and skip exploring it.

This file provides three example strategies:
  1. VeryNaiveRelaxationSolver:
     - Ignores capacity entirely, sets every unfixed item to 1.
     - Fastest, loosest bound.
  2. NaiveRelaxationSolver:
     - Checks that already-fixed items of 1 fit capacity.
     - Sets all unfixed items to 1, ignoring capacity beyond fixed part.
     - Slightly tighter bound than VeryNaive.
  3. MyRelaxationSolver:
     - Stub for your own algorithm (e.g., fractional knapsack, propagation).

You should subclass RelaxationSolver and implement solve(instance, decisions)
so that:
  a) fixed decisions remain unchanged;
  b) objective >= best 0/1 solution consistent with those decisions.
"""

import abc

from .branching_decisions import BranchingDecisions
from .instance import Instance
from .relaxed_solution import RelaxedSolution


class RelaxationSolver(abc.ABC):
    """
    Abstract base for relaxation strategies.

    Implement solve to compute an upper bound on the best 0/1 solution
    consistent with given decisions.
    """

    @abc.abstractmethod
    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        """
        Return a RelaxedSolution satisfying:
          - fixed items in decisions remain at 0 or 1;
          - upper_bound >= best feasible 0/1 solution under those decisions.
        """
        ...


class VeryNaiveRelaxationSolver(RelaxationSolver):
    """
    A relaxation solver for the knapsack problem that naively sets every unfixed
    item to 1 without considering the capacity constraint. This approach provides
    a very loose upper bound for the problem.

    Explanation:
    The solver assumes that all unfixed items can be fully included in the knapsack
    (i.e., their selection is set to 1.0) regardless of the capacity constraint.
    This results in an overestimation of the objective value, making it an upper
    bound. The rationale is that the true optimal solution cannot exceed this
    value since it must respect the capacity constraint, which this naive approach
    ignores.
    """

    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        # build selection: 1.0 for fixed 1 or unfixed, 0 for fixed 0
        selection = [0.0 if x == 0 else 1.0 for x in decisions]
        # compute objective value
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        return RelaxedSolution(instance, selection, upper)


class NaiveRelaxationSolver(RelaxationSolver):
    """
    Ensure fixed 1's fit capacity; set every unfixed item to 1.
    """

    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        # compute capacity after fixed 1 items
        used = sum(item.weight for item, x in zip(instance.items, decisions) if x == 1)
        if used > instance.capacity:
            return RelaxedSolution.create_infeasible(instance)

        selection = [0.0 if x == 0 else 1.0 for x in decisions]
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        return RelaxedSolution(instance, selection, upper)


class MyRelaxationSolver(RelaxationSolver):
    """
    Your relaxation solver stub.

    Implement any relaxation (e.g., fractional knapsack, propagation) to tighten bounds.
    """

    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        # placeholder: behave like NaiveRelaxationSolver
        used = sum(item.weight for item, x in zip(instance.items, decisions) if x == 1)
        if used > instance.capacity:
            return RelaxedSolution.create_infeasible(instance)
        selection = [0.0 if x == 0 else 1.0 for x in decisions]
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        if used > instance.capacity:
            return RelaxedSolution.create_infeasible(instance)
        
        remaining_capacity = instance.capacity - used
        not_fixed = []
        selection = [1.0 if x == 1.0 else 0.0 for x in decisions]
        for i in range(len(decisions)):
            if decisions[i] is None:
                not_fixed.append(i)
        not_fixed_can_fit = [] # items that can fit in the remaining capacity
        for i in not_fixed:
            if instance.items[i].weight <= remaining_capacity:
                not_fixed_can_fit.append(i)
        can_fit = [(instance.items[i], i) for i in not_fixed_can_fit]  # items that can fit in the remaining capacity
        can_fit.sort(key=lambda item: item[0].weight/item[0].value) # sort it
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        while True:
            if len(can_fit) == 0:
                break
            remaining_capacity = instance.capacity - used
            current = can_fit.pop(0)
            if current[0].weight > remaining_capacity:
                upper += remaining_capacity/current[0].weight * current[0].value
                selection[current[1]] = remaining_capacity/current[0].weight
                break
            upper += current[0].value
            used += current[0].weight
            selection[current[1]] = 1.0



        return RelaxedSolution(instance, selection, upper)