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

