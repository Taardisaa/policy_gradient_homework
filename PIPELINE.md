## Details of Actor-Critic Training

┌─────────────────────────────────────────────────────┐
│                    ROLLOUT PHASE                    │
│                                                     │
│  For each step t:                                   │
│    Critic: V(s_t)  ──────────────────┐              │
│    Actor:  a_t ~ π(·|s_t)            │              │
│    Env:    r_t, s_{t+1}              │              │
│                                      ▼              │
│    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  ← TD error  │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                   UPDATE PHASE                      │
│                                                     │
│  GAE: A_t = Σ (γλ)^l · δ_{t+l}                     │
│             (critic's TD errors, smoothed)          │
│                                                     │
│  Actor update:                                      │
│    loss = -E[log π(a_t|s_t) · A_t]                 │
│    → push policy toward actions with A_t > 0       │
│                                                     │
│  Critic update (×80):                               │
│    loss = MSE(V(s_t),  RTG_t)                       │
│    → make V more accurate for next epoch            │
└─────────────────────────────────────────────────────┘

