## Overview

In this project, you will train a JEPA world model on a set of pre-collected trajectories from a toy environment involving an agent in two rooms.


### JEPA

Joint embedding prediction architecture (JEPA) is an energy based architecture for self supervised learning first proposed by [LeCun (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf). Essentially, it works by asking the model to predict its own representations of future observations.

More formally, in the context of this problem, given an *agent trajectory* $\tau$, *i.e.* an observation-action sequence $\tau = (o_0, u_0, o_1, u_1, \ldots, o_{N-1}, u_{N-1}, o_N)$ , we specify a recurrent JEPA architecture as:

$$
\begin{align}
\text{Encoder}:   &\tilde{s}\_0 = s\_0 = \text{Enc}\_\theta(o_0) \\
\text{Predictor}: &\tilde{s}\_n = \text{Pred}\_\phi(\tilde{s}\_{n-1}, u\_{n-1})
\end{align}
$$

Where $\tilde{s}_n$ is the predicted state at time index $n$, and $s_n$ is the encoder output at time index $n$.

The architecture may also be teacher-forcing (non-recurrent):

$$
\begin{align}
\text{Encoder}:   &s\_n = \text{Enc}\_\theta(o_n) \\
\text{Predictor}: &\tilde{s}\_n = \text{Pred}\_\phi(s\_{n-1}, u\_{n-1})
\end{align}
$$

The JEPA training objective would be to minimize the energy for the observation-action sequence $\tau$, given to us by the sum of the distance between predicted states $\tilde{s}\_n$ and the target states $s'\_n$, where:

$$
\begin{align}
\text{Target Encoder}: &s'\_n = \text{Enc}\_{\psi}(o_n) \\
\text{System energy}:  &F(\tau) = \sum\_{n=1}^{N}D(\tilde{s}\_n, s'\_n)
\end{align}
$$

Where the Target Encoder $\text{Enc}\_\psi$ may be identical to Encoder $\text{Enc}\_\theta$ ([VicReg](https://arxiv.org/pdf/2105.04906), [Barlow Twins](https://arxiv.org/pdf/2103.03230)), or not ([BYOL](https://arxiv.org/pdf/2006.07733))

$D(\tilde{s}\_n, s'\_n)$ is some "distance" function. However, minimizing the energy naively is problematic because it can lead to representation collapse (why?). There are techniques (such as ones mentioned above) to prevent this collapse by adding regularisers, contrastive samples, or specific architectural choices. Feel free to experiment.

Here's a diagram illustrating a recurrent JEPA for 4 timesteps:

![Alt Text](assets/hjepa.png)


### Environment and data set

The dataset consists of random trajectories collected from a toy environment consisting of an agent (dot) in two rooms separated by a wall. There's a door in a wall.  The agent cannot travel through the border wall or middle wall (except through the door). Different trajectories may have different wall and door positions. Thus your JEPA model needs to be able to perceive and distinguish environment layouts. Two training trajectories with different layouts are depicted below.

<img src="assets/two_rooms.png" alt="Alt Text" width="500"/>


### Task

Your task is to implement and train a JEPA architecture on a dataset of 2.5M frames of exploratory trajectories (see images above). Then, your model will be evaluated based on how well the predicted representations will capture the true $(x, y)$ coordinate of the agent we'll call $(y\_1,y\_2)$. 

Here are the constraints:
* It has to be a JEPA architecture - namely you have to train it by minimizing the distance between predictions and targets in the *representation space*, while preventing collapse.
* You can try various methods of preventing collapse, **except** image reconstruction. That is - you cannot reconstruct target images as a part of your objective, such as in the case of [MAE](https://arxiv.org/pdf/2111.06377).
* You have to rely only on the provided data in folder `/scratch/DL25SP/train`. However you are allowed to apply image augmentation.

**Failing to meet the above constraints will result in deducted points or even zero points**

