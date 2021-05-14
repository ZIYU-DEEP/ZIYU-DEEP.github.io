# Implications from CMAB, Variational TS and IDS
*Last updated on May 10, 2021.*

## 0. Keys to Our Research Question
We consider the online learning problem for decision making, where the underlying probabilistic parameters are unknown. Specifically, I list the following four aspects which I think as the key features of our problem.
1. **Combinatorial nature**
    - Rather than singleton, actions are taken in subsets – *superarm*.
    - Nested/separated *offline optimization* in *online learning*
        - 🎃 In decision tree, it could be adaptive in each offline session.
    - Rewards can be non-linear(modular) – there may exist *substitutional* and *complementary* effects inside the superarm.
2. **Approximation oracle**
    - Approximation is usually used in the offline optimization step, for the sake of *computational tractability* or *data efficiency*
        - 🎃 Relevance to **bias**
3. **Latent probabilistic structure**
    - There may exist disconnection for *uncertainty on the reward* $r(d|\mathbf{X}_{\mathcal{A}})$ and *uncertainty on the environment* $\mathbb{P}[x_{i}=1 | y_{j}]$
        - 🎃 low priority
4. **Contextual nature**
    - Taking one super arm can reveal information for other super arms.
        - 🎃 not i.i.d. (potential relevance to rl)

The 1st and 2nd (combinatorial nature & approximation oracle) are relevant to **Combinatorial MAB**.\
The 3rd is relevant to **Variational TS** (i.e., influence diagram bandits).\
The 4th is relevant to **Information-Directed Sampling**.

## 1. Combinatorial MAB [[notes](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/21-05-10-notes-CMAB-intro.md)]
- **Similarity**
    - Combinatorial nature
    - Approximation oracle
- **Disimilarity**
    - Latent probabilistic structure is not considered
        - In TS for MAB, prior is directly assumed on the reward.
        - Typically there is no latent/intermediate variable.
- **Useful techniques for us**
    - $(\alpha, \beta)$-approximation oracle (as a general description for offline optimization)
    - $(\alpha, \beta)$-approximation regret analysis

## 2. Variational Thompson Sampling [[talk](https://papertalk.org/papertalks/5503)]
- **Similarity**
    - Latent probabilistic structure
- **Disimilarity**
    - We may need to find a more general probabilistic structure, or to extend based on theirs.
    - We may be able to do more efficiently – with the help of IDS.
- **Useful techniques for us**
    - Influence diagram
    - Variational techniques (p.s., this can further connects with deep bandits)


## 3. Information-Directed Sampling [[notes](https://github.com/ZIYU-DEEP/efficient-online-decision-learning/blob/main/notes/21-05-10-notes-IDS.md)]
- **Similarity**
    - Actions are informationally entangled
        - An action (a sequences of arms) can imply useful information for another action (a sequences of arms).
        - e.g., If we want to learn/explore more, we need to diversify inside a super arm.
- **Disimilarity**
    - Our latent probabilistic structure is different from existing IDS papers.
- **Useful techniques for us**
    - Information ratio and its pricipled sampling methods
    - Rate-distortion techniques for deriving better bounds

## 4. Summary
I feel the key features of our problem can be found reflection in CMAB, variational TS and IDS, however, it still possess unique challenge in the intersection.

In technical contributions, it seems that the regret bound on the approximation side has been much understood now, we may be able to explore more on the variational and informational structure side, i.e., the 3rd and 4th keys. We may further extend this to deep bandits with bayesian network, also as to match the flavor of ICLR.
