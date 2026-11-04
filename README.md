# newton_s_method_root_finding_utils

A small, dependency-free scalar root finder that uses Newton's method with a numerical derivative when no analytic derivative is available.

## Usage

```python
from newton_s_method_root_finding_utils import newton, newton_derivative

# Numerical derivative (symmetric difference).
root = newton(lambda x: x * x - 2.0, 1.0)
print(root)  # 1.4142135623730951

# Analytic derivative, when you have one.
root = newton_derivative(lambda x: x * x - 2.0, lambda x: 2.0 * x, 1.0)
print(root)  # 1.4142135623730951
```

Both functions accept keyword-only `tol` (default `1e-12`) and `max_iter` (default `100`). `newton` additionally accepts `h`, the finite-difference step. On failure they raise `RootFindingError`; on invalid arguments they raise `ValueError`.

## Why this exists

Newton's method is the fastest general scalar root finder when you have a derivative, but many callers only have the function. Wrapping the method in a helper that silently differentiates numerically is the usual solution, and it hides a real trade-off: the numerical derivative limits the achievable accuracy, so a very small `tol` does not buy you a very accurate root. This library exposes both entry points so the choice is explicit. `newton` uses a symmetric difference because it is second-order accurate; `newton_derivative` is there for callers who can pay for the analytic derivative and want the extra digits.

## The awkward edge

The stopping rule is relative: `|x_{n+1} - x_n| <= tol * max(1.0, |x_{n+1}|)`. That keeps `tol` meaningful when the root is large, but it means that for roots near zero the effective absolute tolerance is just `tol`. If you need absolute control near zero, pass a larger `tol`; if you need more digits, note that `newton` cannot deliver them past the numerical-derivative floor.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

