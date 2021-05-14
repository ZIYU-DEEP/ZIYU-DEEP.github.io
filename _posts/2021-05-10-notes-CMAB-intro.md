# Combinatorial MAB
*Summarized from Wei Chen's ICML'13 paper and talk.*
*🎃 This notes is messy as I haven't pruned it yet.*

## Motivation
### Introduction
- How to *learn probability* while *doing optimization*?
- **Combinatorial online learning** creates *iterative feedback loop* between *optimization* and *learning*
    - Action to optimize is combinatorial
- Main difficulties:
    - *Combinatorial* in nature
    - *Non-linear* (reward function?) optimization objective, based on underlying random events
        - Also, the reward may depend not only on the means of its component rewards (i.e. not a simple sum), for example, there might exist substitutional/complementary effect.
    - Offline optimization may already be hard, *need approximation*
    - Online learning: learn while doing *repeated optimization*

### MAB Problem
- Regret= $n \mu^{*}-\mathbb{E}\left[\sum_{t=1}^{n} R_{t}\left(i_{t}^{A}\right)\right]$
- Objective: minimizing regret in $n$ rounds

### Example (Why Combinatorial?)
For many combinatorial optimization problems, when the input is **uncertain**, they may be turned into an **online** learning problem.
- **Linear CMAB**
    - *GPS routing* / $s-t$ *shortest path*: a combination of road segments
    - *Matching*: e.g., wireless channel allocation
    - *Spanning tree*: e.g., wireless routing planning
- **Nonlinear CMAB**
    - *Ads placement* (probabilistic max cover): finding $k$ pages to put ads to maximize total number of users clicking through rate
        - Bipartite graph of pages and users who are interested in certain pages, while each edge has a click-through probability
        - When click-through probabilities are known, this can be solved by approximation
        - When click-through probabilities are not know, the question is – how to learn click-through prob. while doing optimization?
        - Each edge is a base arm, with Bernoulli distribution
        - Reward is the number of users a super covered. Example of nonlinearity: 2 webpages covering the same user is counted as reward 1 not 2.
        - Offline problem is NP hard, a greedy algorithm achieves $(1-1 / e, 1)$-approximation.
    - *News recommendation*: combination of different type of news a user may be interested in

### Contribution of the paper
- CMAB Framework which handles non-linear reward
- Compare to related work
    - Linear bandits: more general
    - Submodular optimization: **no submodularity requirement**

### Summary
- Need combinatorial online learning in practice
- Naïve MAB is not feasible

## CMAB Framework and Solution
### CMAB Framework
- At each round, select a subset of arms (i.e., super arm) $S_{t}^{A} \subseteq[m]$ according to algorithm $A$.
    - Outcomes of all played base arms are observed.
    - Outcome of an arm $i \in[m]$ has an unknown distribution with unknown mean $\mu _{i}$
- Reward of the super arm $r_{\mu}(S) = R_{t}(S_{t}^{A})$ is a functions of the outcomes of all played arms.
    - Expected reward $\mathbb{E}\left[R_{t}(S)\right]$ only depends on $S$ and $\boldsymbol{\mu}=\left(\mu_{1}, \mu_{2}, \ldots, \mu_{m}\right)$.
    - Optimal reward: $\operatorname{opt}_{\boldsymbol{\mu}}=\max _{S} r_{\mu}(S)$

### Assumptions on $r_{\mu}(S)$
- **Monotonicity**: If $\boldsymbol{\mu} \leq \boldsymbol{\mu}^{\prime}$ (pairwise), $r_{\mu}(S) \leq r_{\mu^{\prime}}(S)$.
- **Bounded Smoothness**: $\exists f(\cdot)$, such that $\left|r_{\mu}(S)-r_{\mu^{\prime}}(S)\right| \leq f(\Delta)$, where $\Delta=\max _{i \in S}\left|\mu_{i}-\mu_{i}^{\prime}\right|$.

### Offline Computation Oracle
> **Key idea**: We do not analyze the performance with the true optimal, rather, we compare against the $\alpha \cdot \beta$ fraction of the optimal reward. This is because only a $\beta$ fraction of the oracle computation is successful, and when successful, the reward is only $\alpha$-approximate of the optimal value.

- $(\alpha, \beta)$-**approximation oracle**
    - Input: $\boldsymbol{\mu}=\left(\mu_{1}, \mu_{2}, \ldots, \mu_{m}\right)$
    - Output: a super arm $S$ such that $\operatorname{Pr}\left[r_{\mu}(S) \geq \alpha \cdot \operatorname{opt}_{\mu}\right] \geq \beta$.

- $(\alpha, \beta)$-**approximation regret**
    - Compare against the $\alpha \beta$ fraction of the optimal:
    $$
    \text { Regret }=n \cdot \alpha \beta \cdot \operatorname{opt}_{\mu}-\mathbb{E}\left[\sum_{i=1}^{n} r_{\mu}\left(S_{t}^{A}\right)\right]
    $$

- **Challenges**: unknown
    - combinatorial structure
    - reward function
    - arm outcome distribution
    - how oracle computes the solution

- **CUCB Algorithm in $(\alpha, \beta)$-oracle**
    - `Offline computation oracle`: given $\overline{\boldsymbol{\mu}}=\left(\bar{\mu}_{1}, \bar{\mu}_{2}, \ldots, \bar{\mu}_{m}\right)$, output a super-arm $S$ using an $(\alpha, \beta)$-**approximation oracle** and play $S$.
    - `Estimation`: get $\hat{\mu}_{i}$ which is the sample mean outcome on arm $i$.
    - `Adjustment`: in UCB manner, compute $\bar{\mu}_{i}=\hat{\mu}_{i}+\sqrt{\frac{3 \ln n}{2 T_{i}}}$.
    - Repeat.

## Theorems
- **Theorem 1**: The $(\alpha, \beta)$-approximation oracle regret of the CUCB algorithm in $n$ rounds using an $(\alpha, \beta)$-approximation oracle is at most:
$$
\sum_{i \in[m], \Delta_{\min }^{i}>0}\left(\frac{6 \ln n \cdot \Delta_{\min }^{i}}{\left(f^{-1}\left(\Delta_{\min }^{i}\right)\right)^{2}}+\int_{\Delta_{\min }^{i}}^{\Delta_{\max }^{i}} \frac{6 \ln n}{\left(f^{-1}(x)\right)^{2}} \mathrm{~d} x\right)+\left(\frac{\pi^{2}}{3}+1\right) \cdot m \cdot \Delta_{\max }
$$
($\Delta_{\min }^{i}$ denotes the minimum gap between $\alpha \cdot \operatorname{opt}_{\boldsymbol{\mu}}$ and reward of a bad super arm containing arm $i$.)

- **Theorem 2**: If the bounded smoothness function $f(x)=\gamma \cdot x^{\omega}$ for some $\gamma \geq 0$ and $\omega \in (0, 1]$, the regret then becomes:
    $$
    \frac{2 \gamma}{2-\omega} \cdot(6 m \ln n)^{\omega / 2} \cdot n^{1-\omega / 2}+\left(\frac{\pi^{2}}{3}+1\right) \cdot m \cdot \Delta_{\max }
    $$
    - When $\omega = 1$, the distribution-independent bound is $O(\sqrt{m n \ln n})$.

## Thompson Sampling for CMAB
- With an exact computation oracle, TS can achieve $O\left(\sum_{i=1}^{m} \log T / \Delta_{i}\right)$ regret.
- With an approximation oracle, TS CANNOT guarantee sublinear regret.
    - 🎃 Note in our case, every step in the "offline session" is adaptive. Cumulative bias may exist. *Belief state*.

## Summary
- $(\alpha, \beta)$-**approximation oracle** and **regret** make it easier to analyze the algorithm performance
- Modular approach: separation between *online learning* and *offline optimization*.
