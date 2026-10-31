"""Scalar root finding with Newton's method and numerical derivatives.

The module intentionally keeps a small surface: :func:`newton` for the common
case where only the function is known, and :func:`newton_derivative` when the
analytic derivative is available and worth the extra accuracy.
"""

from __future__ import annotations

from typing import Callable, Optional


class RootFindingError(RuntimeError):
    """Raised when Newton's method cannot make further progress."""


def _check_tolerance(tol: float) -> None:
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")


def _check_x0(x0: float) -> None:
    if x0 != x0:
        raise ValueError("initial guess must not be NaN")


def newton_derivative(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Find a root of ``f`` starting from ``x0`` using an analytic derivative.

    The stopping rule is the relative step size ``|x_{n+1} - x_n| <= tol *
    max(1.0, |x_{n+1}|)``.  Using a relative rule makes the same ``tol``
    meaningful whether the root is near 1 or near 1e9; using an absolute rule
    would force callers to rescale ``tol`` per problem.

    Raises :class:`RootFindingError` if the derivative is zero at a step (the
    method cannot proceed) or if :func:`newton_derivative` does not converge
    within ``max_iter`` iterations.  A zero derivative is reported separately
    from non-convergence because the caller may want to retry from a different
    ``x0`` rather than raise the iteration budget.
    """
    _check_tolerance(tol)
    _check_x0(x0)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    x = float(x0)
    for _ in range(max_iter):
        fx = f(x)
        if fx == 0.0:
            return x
        dfx = df(x)
        if dfx == 0.0:
            raise RootFindingError(
                f"derivative is zero at x={x!r}; cannot continue"
            )
        step = fx / dfx
        x_next = x - step
        if abs(step) <= tol * max(1.0, abs(x_next)):
            return x_next
        x = x_next

    raise RootFindingError(
        f"newton_derivative did not converge within {max_iter} iterations"
    )


def newton(
    f: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
    h: Optional[float] = None,
) -> float:
    """Find a root of ``f`` starting from ``x0`` using a numerical derivative.

    The derivative is estimated with a symmetric difference:

        f'(x) ~= (f(x + h) - f(x - h)) / (2 * h)

    A symmetric stencil is used instead of a forward difference because it is
    second-order accurate, so the same ``h`` gives a much better derivative
    estimate and therefore a much better Newton step.  The default ``h`` is
    ``sqrt(eps) * max(1.0, |x|)``, which keeps the finite-difference error and
    the floating-point cancellation error roughly balanced across scales.

    Pass ``h`` explicitly when ``f`` has a known scale or when the default
    step lands on a non-smooth point of ``f``.  ``h`` must be positive.

    Raises :class:`RootFindingError` if the estimated derivative is zero or if
    the method does not converge within ``max_iter`` iterations.
    """
    _check_tolerance(tol)
    _check_x0(x0)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if h is not None and h <= 0.0:
        raise ValueError("h must be positive")

    def df(x: float) -> float:
        step = h if h is not None else (2.220446049250313e-16 ** 0.5) * max(1.0, abs(x))
        return (f(x + step) - f(x - step)) / (2.0 * step)

    return newton_derivative(f, df, x0, tol=tol, max_iter=max_iter)
