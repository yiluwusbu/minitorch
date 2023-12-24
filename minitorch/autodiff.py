from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    f_x = f(*vals)
    vals_h = list(vals)
    vals_h[arg] += epsilon
    f_xh = f(*vals_h)
    return (f_xh - f_x) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # run bfs to collect the out degrees of all nodes
    out_degrees = dict()
    work_list = [variable]
    while len(work_list) > 0:
        var = work_list.pop(0)
        for p in var.parents:
            if p.is_constant():
                continue
            if p.unique_id not in out_degrees:
                out_degrees[p.unique_id] = 1
                work_list.append(p)
            else:
                out_degrees[p.unique_id] += 1

    # topological sort
    work_list = [variable]
    sorted_list = []
    while len(work_list) > 0:
        var = work_list.pop(0)
        if not var.is_constant():
            sorted_list.append(var)
        for p in var.parents:
            if p.is_constant():
                continue
            out_degrees[p.unique_id] -= 1
            if out_degrees[p.unique_id] == 0:
                work_list.append(p)

    return sorted_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    var_derivs = dict()
    var_derivs[variable.unique_id] = deriv
    topo_order_list = topological_sort(variable)
    for var in topo_order_list:
        d_var = var_derivs[var.unique_id]
        # check whether it is a leaf node
        if var.is_leaf():
            var.accumulate_derivative(d_var)
        else:
            derivs = var.chain_rule(d_var)
            for input, d in derivs:
                id = input.unique_id
                if id not in var_derivs:
                    var_derivs[id] = 0
                var_derivs[id] += d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
