---
layout:     post
title:      "Information Directed Sampling Revisited"
subtitle:   "A random note"
date:       2021-05-14
author:     "Ziyu Ye"
header-img: "img/post-bg-svdund.jpg"
catalog:    true
mathjax:    true
tags:
    - Bandits
    - IDS
---

*This is an additional note to my [previous note](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/21-05-10-notes-IDS.md). Also see the new [paper list on IDS](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/Paper-List-on-Information-Directed-Sampling.md), and the [note on Andreas Krause's IMSI talk](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/21-05-10-notes-IMSI-workshop.md).*

---

# 1. High-level Idea
### Decoupling *Exploitation* and *Exploration*
$$\pi_{t}^{I D S}=\arg \min _{\pi \in \mathcal{D}(\mathcal{A})}\left\{\Psi_{t}(\pi):=\frac{\Delta_{t}(\pi)^{2}}{I_{t}(\pi)}\right\}$$

As a decision-making policy, Information-Directed Sampling (IDS) is featured by its decoupling of exploitation and exploration in optimization:
- **Exploitation** is governed by **immediate regret** $\Delta_{t}(\pi)$.
- **Exploration** is governed by **mutual information** (i.e., information gain) $I_{t}(\pi)$.

### Exploration: Decoupling `Epistemic` and `Aleatoric` Uncertainty
Specifically in **exploration**, IDS is more efficient than other approaches (e.g., UCB, Thompson Sampling). The reason is that, IDS only concerns about the **informative uncertainty**:
- `Epistemic uncertainty`: $P\left(\mathcal{E} \in \cdot \mid H_{t}\right)$, i.e., uncertainty of the environment given your observed history.
- `Aleatoric uncertainty`: $P\left(O_{t+1}=\cdot \mid \mathcal{E}, H_{t}, A_{t}\right)$, i.e., uncertainty of the next observation conditioned on the environment.
- Notice that we consider only `epistemic uncertainty` as the informative part of uncertainty, as it allows us to learn about the environment.

### Mutual Information: Two Perspectives
As summarized in the [paper list](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/Paper-List-on-Information-Directed-Sampling.md), the two lines of work describes the exploration phase of IDS by interpreting mutual information differently:
- [Van Roy's line](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/21-05-10-notes-IDS.md): $I_{t}(\mathbf{x})$ represents the **reduction of uncertainty** on the **posterior distribution of best arm** $P\left(a^{*}=a \mid \mathcal{F}_{t-1}\right)$.
- [Krause's line](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/21-05-10-notes-IMSI-workshop.md): $I_{t}(\mathbf{x})$ represents the **ratio of epistemic and aleatoric uncertainty**. The key idea here is that not uncertainty $\neq$ informativeness, i.e., only the environment-related uncertainty is informative.

Either perspective, the key spirit is the same: we should get only information *relevant* to the environment.

# 2. Problem Statement
### Setting
- **Action set**: $\mathcal{A}$.
    - An agent chooses an arm $a_{t}$ at time $t \in[1, T]$.
- **Reward distribution**: $p_{a}$.
    - Reward $r_{a}$ are drawn from the reward distribution $p_{a}$ of the arm $a$, and the estimated reward distribution at each time $t$ is denoted as $\hat{p}_{a,t}$.
- **Optimal arm**: $a^{*}=\arg \max _{a \in \mathcal{A}} \mathbb{E}_{r_{a} \sim p_{a}}\left[r_{a}\right]$.
- **Belief (posterior) on best arm**: $\alpha_{t}(a)=P\left(a^{*}=a \mid \mathcal{F}_{t-1}\right)$.
    - $\mathcal{F}_{t-1}$ is the past history including chosen arms and their observed rewards.
- **Policy**: $\pi$.
    - This would be constructed based on the posterior $\alpha_{t}$.

### Regret
$$\mathbb{E}[\operatorname{Regret}(T)]=\underset{r_{a}^{*} \sim p_{a^{*}}}{\mathbb{E}} \sum_{t=1}^{T} r_{a^{*}}-\underset{a \sim \pi \atop r_{a, t} \sim p_{a}}{\mathbb{E}} \sum_{t=1}^{T} r_{a, t}$$

### `Immediate Regret` $\Delta_{t}(a)$
$$\Delta_{t}(a)=\underset{a^{*} \sim \alpha_{t} \atop r_{a^{*}, t} \sim \hat{p}_{a^{*}, t}}{\mathbb{E}}\left[r_{a^{*}, t} \mid \mathcal{F}_{t-1}\right]-\underset{r_{a, t} \sim \hat{p}_{a, t}}{\mathbb{E}}\left[r_{a, t} \mid \mathcal{F}_{t-1}\right]$$

### `Information Gain` $I_{t}(x)$
$$\begin{aligned} I_{t}(a) &=I\left(a_{t}^{*}, r_{a, t}\right) \\ &=\underset{r_{i} \sim \hat{p}_{a,t}}\mathbb{E}\left[H\left(a_{t}^{*}\right)-H\left(a_{t+1}^{*}\right) \mid \mathcal{F}_{t-1}, a_{t}=a, r_{a, t}=r_{i}\right] \end{aligned}$$

### Optimization
$$\pi_{t}^{I D S}=\arg \min _{\pi \in \mathcal{D}(\mathcal{A})}\left\{\Psi_{t}(\pi):=\frac{\Delta_{t}(\pi)^{2}}{I_{t}(\pi)}\right\}$$

<!-- #### Regret Bound
$$
\mathbb{E}\left(\operatorname{Regret}\left(T, \pi^{I D S}\right)\right) \leq \sqrt{\frac{1}{2}|\mathcal{A}| H\left(\alpha_{1}\right) T}
$$ -->

# 3. Key Theoretical Conclusions
Fix a deterministic $\lambda \in \mathbb{R}$ and a policy $\pi=\left(\pi_{1}, \pi_{2}, \ldots\right)$ such that $\Psi_{t}\left(\pi_{t}\right) \leq \lambda$ almost surely for each $t \in\{1, . ., T\} .$ Then:
$$
\begin{aligned}
\mathbb{E}(\operatorname{Regret}(T, \pi)) &=\mathbb{E} \sum_{t=1}^{T} \Delta_{t}(\pi) \\
& \leq \sqrt{\lambda} \mathbb{E} \sum_{t=1}^{T} \sqrt{g_{t}(\pi)} \\
& \leq \sqrt{\lambda T} \sqrt{\mathbb{E} \sum_{t=1}^{T} g_{t}(\pi)} \ \ \text { (Caushy-Schwardsz inequality) } \\
& \leq \sqrt{\lambda H\left(\alpha_{1}\right) T}.
\end{aligned}
$$

# 4. Connections to Combinatorial Bandits
Suppose $\mathcal{A} \subset\{a \subset\{0,1, \ldots, d\}:|a| \leq m\}$, and that there are random variables $\left(X_{t, i}: t \in \mathbb{N}, i \in\{1, \ldots, d\}\right)$ such that
$$
Y_{t, a}=\left(X_{t, i}: i \in a\right) \quad \text { and } \quad R_{t, a}=\frac{1}{m} \sum_{i \in a} X_{t, i}.
$$
Assume that the random variables $\left\{X_{t, i}: i \in\{1, \ldots, d\}\right\}$ are independent conditioned on $\mathcal{F}_{t}$ and $X_{t, i} \in\left[\frac{-1}{2}, \frac{1}{2}\right]$ almost surely for each $(t, i) .$ Then for all $t \in \mathbb{N}, \Psi_{t}\left(\pi_{t}^{\mathrm{IDS}}\right) \leq \frac{d}{2 m^{2}}$ almost surely. Thus:
$$
\mathbb{E}[\operatorname{Regret}(T, \pi)] \leq \sqrt{\frac{d}{2 m^{2}} H\left(\alpha_{1}\right) T}.
$$
We could further prove that the lower bound of this problem is of order $\sqrt{\frac{d}{m} T}$, so the bound is order optimal to a $\sqrt{\log \left(\frac{d}{m}\right)}$ factor.


# 5. Can We Do Better: Adaptive IDS?
(To be updated.) An extension could be exploration inside each inner loop.
