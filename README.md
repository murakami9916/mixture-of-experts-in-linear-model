# Mixture of Experts in Linear Model

## 出力モデル
$$
    \bm{y} = \sum_{m=1}^{M}{g_m(\bm{x})e_m(\bm{x})}
$$

## 確率モデル

$$
\begin{aligned}
    p(\bm{y} \mid \bm{x}, \theta) &= \sum_{m=1}^{M}{p(\bm{y}, m \mid \bm{x}, \theta)},\\
    &= \sum_{m=1}^{M}{p(m \mid \bm{x}, \theta_g)p(\bm{y} \mid m, \bm{x}, \theta_e)},\\
    &= \sum_{m=1}^{M}{p(m \mid \bm{x}, \theta_g)p(\bm{y} \mid m, \bm{x}, \theta_e)},
\end{aligned}
$$

## フォワードモデル

$$
\begin{aligned}
    e_m(x) &= w_m\bm{x} + b_m,\\
    g_m(x) &= \frac{\mathcal{N}(x \mid \mu_m, \sigma_m)}{\sum_{i=1}^{M}{\mathcal{N}(x \mid \mu_i, \sigma_i)}},
\end{aligned}
$$

## 尤度設計

$$
    p(y_n \mid x_n, m, \theta_{e,m}) = \sqrt{\frac{\beta}{2\pi}} \exp{ \left[ -\frac{\beta}{2} \{ y_n - e_m(x_n) \}^2 \right]}
$$

$$
\begin{aligned}
    p(Y \mid X, \theta) =& \prod_{n=1}^{N}{ \sum_{m=1}^{M}{p(y_n, m \mid x_n, \theta_m)} },\\
    &= \prod_{n=1}^{N}{ \sum_{m=1}^{M}{p(m \mid x_n, \theta_{g,m})p(y_n \mid m, x_n, \theta_{e,m})} }
\end{aligned}
$$
