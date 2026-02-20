# Policy Gradient Homework Report

Author: Hao Ren
Date: 18-Feb-2026

## Part 1a: Understanding and Implementing a Basic Policy Gradient Algorithm

I ran the `1_simple_pg_gymnasium.py` and here's the result (the last 5 epochs):

```
epoch:  45       loss: 134.770   return: 220.391         ep_len: 220.391
epoch:  46       loss: 145.005   return: 240.571         ep_len: 240.571
epoch:  47       loss: 180.381   return: 271.211         ep_len: 271.211
epoch:  48       loss: 193.272   return: 283.333         ep_len: 283.333
epoch:  49       loss: 211.870   return: 335.250         ep_len: 335.250
```

### Discussion

**Is the agent learning?**

I believe the agent is learning. The return is increasing over time, which indicates the agent is improving its policy to achieve higher rewards. 

**What happens to it's performance over time?**

Its performance, measured by the return, is generally increasing over time. Therefore the agent is improving its performance over time.

**Is it monotonically improving?**

To evaluate this, I ran the script 6 times and plotted the return over 50 epochs for each run. The plot is shown below:

| | |
|---|---|
| ![Run 0](results/vanilla_pg/return_run0.png) | ![Run 1](results/vanilla_pg/return_run1.png) |
| ![Run 2](results/vanilla_pg/return_run2.png) | ![Run 3](results/vanilla_pg/return_run3.png) |
| ![Run 4](results/vanilla_pg/return_run4.png) | ![Run 5](results/vanilla_pg/return_run5.png) |

As shown above, the return is not monotonically improving. In some runs (run 0, 1, 3), the return increases over the epochs, while in others (run 2, 4, 5), it fluctuates and even decreases at certain epochs. I belive this is a common issue for vanilla policy gradient, because its high variance can lead to unstable learning.

## Part 1b: Understanding and Implementing a Basic Policy Gradient Algorithm

I implemented the render function in `1_simple_pg_gymnasium.py` to visualize the agent's behavior. Below is one of the rendered policies in an epoch:

![Rendered Policy](results/vanilla_pg/frames/epoch_026.gif)

## Discussion

**What do you notice qualitatively about how its policy changes over time?**

In the first few episodes (0~16): The agent failed to make corrective actions in time, resulting in the pole falling down quickly in a direction.

In the middle episodes (17~33): The agent started to learn to make corrective actions in time, but sometimes the actions are too aggressive, causing the cart to move to the edge of the track.

In the final episodes (34~49): The agent learned to make more balanced actions to keep the pole upright for a longer time, and also keep the cart within the track. In the final episode, the agent successfully kept the pole upright for a long time, reaching the maximum return of 500.

![Final Episode](results/vanilla_pg/frames/epoch_049.gif)

## Part 2: Reducing Variance with Reward-to-Go

I added the reward-to-go implementation in `reward_to_go.py` and ran both the vanilla policy gradient and reward-to-go implementations 5 times each. The learning curves are shown below:

![Vanilla PG vs Reward-to-Go](results/comparison.png)

Generally speaking the reward-to-go implementation achieves higher returns than the vanilla one on average return, indicating that it is more stable and possibly more effective in learning a policy. However, the reward-to-go implementation still has some flucturations and drops in the final epochs.

## Part 3: Continuous Actions

I modified the code to use `MLPActorCritic` from `core.py` to make it work for both discrete and continuous action spaces. I tested it on CartPole (discrete) and "InvertedPendulum-v5" (continuous). I actually also tried it on Hopper-v5 and Ant-v5, but the learning was very unstable and the return was very low, so I won't include those results here. All the hyperparameters are the same as the default ones (epochs=50, lr=1e-2, etc.).

For CartPole, it is just like the previous implementation.

[CartPole](results/part_3/CartPole-v1/CartPole-v1_run4.png)

For InvertedPendulum-v5:

[InvertedPendulum-v5](results/part_3/InvertedPendulum-v5.png)

And here's the visualization result:

[InvertedPendulum-v5-visualization](results/part_3/InvertedPendulum-v5/checkpoints/gifs/epoch_049.gif)

As shown above, the agent is able to keep the pendulum upright for a long time, until reaching a maximum return of 1000. This indicates that our implementation of policy gradient correctly works for continuous action spaces and can learn a good policy for the InvertedPendulum-v5 environment.

## Extra Credit A: Baselines and Generalized Advantage Estimation



