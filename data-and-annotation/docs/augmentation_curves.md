# Quadratic Augmentation Curves

Let $t\in[0,1]$ be the normalized time within a clip, and let
$a_{\min}$ and $a_{\max}$ be the minimum and maximum augmentation strengths.

| Curve | Increasing clip | Decreasing clip |
|---|---|---|
| `quad_first_flat` | $\displaystyle a(t)=a_{\min}+(a_{\max}-a_{\min})t^2$ | $\displaystyle a(t)=a_{\max}-(a_{\max}-a_{\min})t^2$ |
| `quad_last_flat` | $\displaystyle a(t)=a_{\min}+(a_{\max}-a_{\min})\left[1-(1-t)^2\right]$ | $\displaystyle a(t)=a_{\max}-(a_{\max}-a_{\min})\left[1-(1-t)^2\right]$ |

`quad_first_flat` has zero slope at $t=0$, while `quad_last_flat` has zero
slope at $t=1$. Their progress functions are $t^2$ and
$1-(1-t)^2=2t-t^2$, respectively.
