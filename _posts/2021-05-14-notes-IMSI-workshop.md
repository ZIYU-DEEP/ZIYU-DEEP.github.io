# Notes on Information-Directed Exploration in Bandits and RL
*Summarized the bandit's part from Andreas Krause's [talk](https://www.imsi.institute/videos/%E2%80%8Binformation-directed-exploration-in-bandits-and-reinforcement-learning/) on April 2021.*

## 1. Motivation on Sequential Decision Making
- **Research questions**:
    - *How much* information is needed for a task?
    - How do we *efficiently* collect it?

- **Key Challenges**:
    - We may have to gather information with only *suboptimal actions*.
        - In this context, it refers to the classical exploitation-exploration trade-offs.
        - More broadly, we may face constraints on computational efficiency/tractability that can leads to *approximate solutions* that are suboptimal.
    - Achievable performance depends on *problem difficulty*.
        - Currently there are only specialized algorithms for different problems.

<!-- - **Key Takeaways**:
    - An efficient algorithm on *partial monitoring* problems.
        - It performs well simultaneously in all *linear* settings.
        - Also it works for *nonparametric* and *contextual* settings.
    - Approach is based on *Information-Directed Sampling*. -->

## 2. Multi-Armed Bandits Problem as an Example
### 2.1. Basic Setting
- **Research questions**: how should we allocate $T$ tokens to $k$ arms to maximize our returns?
- **Algorithmic principle**: optimism in the face of uncertainty.
- **Structured bandits**:
    - Action: $x_{t}$.
    - Reward: $r_{t}=f(x_{t})$.
    - Feedback: $y_{t} = f(x_{t}) + \epsilon _{t}$, where $\epsilon _{t}$ is $\sigma _{n}$-sub-Gaussian noise.
    - Regret: $R_{T}=\sum_{t=1}^{T}\left(f\left(x^{*}\right)-f\left(x_{t}\right)\right)$.

### 2.2. Estimating Reward: Failure of Optimism!
In estimating the reward function $f(x)$, a common approach is to construct **confidence bounds**.
- Uncertainty is used to guide exploration & exploitation.
- A common construction: $f(x) \in \mu_{t}(x) \pm \beta_{t} \sigma_{t}(x)$.
    - *Bayesian optimization*  is a convenient option, since smoothness is characterized via covariance function that $k\left(x, x^{\prime}\right)=\operatorname{Cov}\left(f(x), f\left(x^{\prime}\right)\right)$.
    - **Optimistic exploration**: focus exploration on *plausible maximizers*.
        - i.e., upper confidence bound $\geq$ best lower bound.

**Upper confidence sampling** (UCB) is one common technique:
- **Key idea**: pick input that maximizes UCB.
    - $x_{t}=\arg \max _{x \in D} \mu_{t-1}(x)+\beta_{t} \sigma_{t-1}(x)$.
    - Sublinear regret can often be achieved with appropriate $\beta$.
- **Issues**: optimistic exploration favors *uncertainty*.
    - **Uncertainty** $\neq$ **Informativeness** (on environment  )!
    - In the case of heteroscedastic noise ($y_{t}=f\left(x_{t}\right)+\epsilon_{t}\left(x_{t}\right)$) where noise depends on the input, this could be a problem.
    - 🎃 UCB fails to distinguish **epistemic uncertainty** from **aleatoric uncertainty**.
        - `Epistemic uncertainty`: $P\left(\mathcal{E} \in \cdot \mid H_{t}\right)$, i.e., uncertainty of the environment given your observed history.
        - `Aleatoric uncertainty`: $P\left(O_{t+1}=\cdot \mid \mathcal{E}, H_{t}, A_{t}\right)$, i.e., uncertainty of the next observation conditioned on the environment. In this case, this will be like heteroscedastic noise of an observation.
    - In English, UCB has no preference on less noisy options over more noisy options. This is often not undesirable as people usually want to impose preference on variance of an option.

### 2.3. Resolve Failure of Optimism by Mutual Information
#### Big Picture
The crucial thing is that we can **inject preference** on `epistemic uncertainty` v.s. `aleatoric uncertainty` by **mutual information**.
- **Key technique**: mutual information can be considered as (i.e., approximated by) the the **ratio** of `epistemic uncertainty` and `aleatoric uncertainty`.
- 👾 My random thought: can information bottleneck somehow play a role here?

#### Information Gain
A traditional approach is active learning and experiment design is to choose the options giving you larger information gain. Suppose $S$ is the set of observations/experiments, we have:
- $F(S):=I\left(f ; y_{S}\right)=\underbrace{H(f)}_{\begin{array}{c}\text { uncertainty of } f \\ \text { before evaluation }\end{array}} - \underbrace{H\left(f \mid y_{S}\right)}_{\begin{array}{c}\text { uncertainty of } f \\ \text { after evaluation at } S\end{array}}$.

#### Submodularity of Mutual Information
A nice thing is that **mutual information** $F(S)$ is **monotone submodular**. $\forall x \in X$ and $\forall A \subseteq B \subseteq X$, we have:
- $F(A \cup\{x\})-F(A) \geq F(B \cup\{x\})-F(B)$
- In this case, greedy algorithm is near-optimal, i.e., we could find a set with near-optimal information gain by a greedy algorithm.

#### Mutual Information as $\sigma_{t}(x) / \sigma_{n}(x)$
In the above heteroscedastic noise example where the noise is Gaussian, we have that:
- $\mathrm{I}_{\mathrm{t}}(\mathrm{x}):=F\left(S_{t} \cup\{x\}\right)-F\left(S_{t}\right)=\frac{1}{2} \log \left(1+\frac{\sigma_{t}(x)}{\sigma_{n}(x)}\right)$.
    - $\sigma_{t}(x)$ is the `epistemic uncertainty`, i.e., uncertainty of the distribution of reward function $f(x_{t})$. This uncertainty is regarded as **informative**, as it does help you to learn about the environment.
    - $\sigma_{n}(x)$ is the `aleatoric uncertainty`, i.e., uncertainty of the heteroscedastic noise $\epsilon_{t}$. This uncertainty is regarded as **non-informative**, as it does *not* help you to learn about the environment.

#### Exploring by Maximizing Mutual Information
A natural way to explore and pick the next experiment would then be to maximize the mutual information, given the preference on reduction of `epistemic uncertainty` $\sigma_{t}(x)$:
- $x_{t}=\arg \max _{x} I_{t}(x)=\arg \max _{x} \frac{\sigma_{t}(x)^{2}}{\sigma_{n}(x)^{2}}$

#### Balancing Exploration and Exploitation
As in Van Roy's 2014 paper, we take immediate regret into account to trade off between exploitation and exploration:
- $\mathbf{x}_{t+1}=\arg \min _{\mathbf{x}} \frac{\Delta_{t}(\mathbf{x})^{2}}{I_{t}(\mathbf{x})}$

Specifically, we know that:
- $I_{t}(\mathbf{x})=I\left(f ; y(\mathbf{x}) \mid D_{t}\right) \propto \log \left(1+\frac{\sigma_{t}^{2}(\mathbf{x})}{\sigma_{n}^{2}(\mathbf{x})}\right)$
    - Note that other information measures can also be used here, e.g., $I\left(f\left(x_{U C B}\right) ; y(x) \mid D_{t}\right)$.

## 3. Summary
The decomposition of `epistemic uncertainty` $\sigma_{t}(x)$ and `aleatoric uncertainty` $\sigma_{n}(x)$ provides a different perspective on Van Roy's original IDS paper.
- From Van Roy's perspective: $I_{t}(\mathbf{x})$ represents the **reduction of uncertainty** on the **posterior distribution of best arm** $P\left(a^{*}=a \mid \mathcal{F}_{t-1}\right)$.
- From Krause's perspective: $I_{t}(\mathbf{x})$ represents the **ratio of epistemic and aleatoric uncertainty**. The key idea here is that not uncertainty $\neq$ informativeness, i.e., only the environment-related uncertainty is informative.

Either perspective, the goal is to get only **information** *relevant* to the **environment**.
