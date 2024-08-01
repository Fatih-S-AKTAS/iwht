# Iterative Weighted Hard Thresholding (IWHT) Algorithms for Sparsity Constrained Optimization

This repository contains few algorithms that can be used to solve the following optimization model

```math
\begin{aligned}
& \min_{x} \ f(x) = \|Ax-b\|_2^2 \\
& \text{s.t.} \ \|x\|_0 \leq s
\end{aligned}
```

Algorithms implemented use Lipschitz continuity of the gradient with respect to $\ell_2$ norm ($||x||_2 = \sqrt{x^Tx}$) and weighted $\ell_2$ norm 
($||x||_D = \sqrt{x^TDx}$ for some diagonal matrix $D$) which is computed numerically by the algorithms in get_dsm.py

