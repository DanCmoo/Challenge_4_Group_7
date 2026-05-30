# Machine Learning — Challenge 4

# Learning from Demonstration with Adversarial Training

# (GAIL on Atari) — Three-Way Comparison with Challenges 1

# & 3

## Prof. Carlos Andrés Sierra, M.Sc.

```
Full-time Adjunct Professor
Computer Engineering Program
School of Engineering
Universidad Distrital Francisco José de Caldas
```
**Overview**
This document describes Challenge 4 for the Machine Learning course. It is the third
and final extension of Challenge 1, completing a three-algorithm comparison series:

```
Challenge 1 (DQN) −→ Challenge 3 (PPO) −→ Challenge 4 (GAIL: Learning from
Demonstration + Adversarial Training)
```
Each group continues working on the **same ALE environment** assigned in Chal-
lenge 1. The new learning paradigm introduced here is **Generative Adversarial Im-
itation Learning (GAIL)** (Ho & Ermon, 2016): a framework that combines ideas from
Generative Adversarial Networks (GANs) and Reinforcement Learning to allow an agent
to learn from _demonstrations_ without requiring an explicit reward function from the envi-
ronment.
The central research question of Challenge 4 is:

```
“Does an agent that learns by imitating demonstrations — through adversarial
training rather than direct reward maximisation — converge faster, transfer bet-
ter, or reach higher final performance than pure RL agents (DQN, PPO) on the
assigned Atari game? Under which conditions does the source of demonstrations
matter?”
```
**Demonstration source.** A central practical decision students must make is where
the demonstrations come from. Three legitimate sources are authorised in this challenge:

Carlos Andrés Sierra, Computer Engineer, M.Sc. in Computer Engineering, Full-time Adjunct Professor
at Universidad Distrital Francisco José de Caldas.
Any comment or concern about this document can be sent to: _cavirguezs@udistrital.edu.co_.


1. **Self-generated** : Roll out the best DQN checkpoint from Challenge 1 (or PPO from
   Challenge 3) and record ( _st,at_ ) tuples. The expert is imperfect, which is realistic and
   scientifically interesting.
2. **Policy-gradient warm-start** : Train PPO for a short budget (e.g., 500 000 steps) and
   use those trajectories as demonstrations; then train GAIL with the adversarial signal
   replacing the environment reward.
3. **Pre-collected Atari demonstrations** (optional, if available): Use publicly avail-
   able human demonstration datasets (e.g., Atari Grand Challenge dataset) for a small
   number of games.

Regardless of the source, the quality and quantity of demonstrations must be clearly
documented and treated as a variable in the experimental analysis.

```
Challenge objective
Groups will:
```
1. Implement a **Behavioral Cloning (BC)** baseline that directly minimises the cross-
   entropy between demonstrations and policy outputs. This establishes a supervised-
   learning lower bound.
2. Implement a **GAIL agent** : a discriminator network trained adversarially against the
   policy, providing a learned reward signal used with PPO as the inner RL algorithm.
3. Collect and document a demonstration dataset from the group’s own challenge agents
   (or from an authorised external source).
4. Conduct experiments over the demonstration quantity and quality, and compare GAIL
   against BC, DQN (Challenge 1), and PPO (Challenge 3) using the same evaluation pro-
   tocol.
5. Produce a scientific report in IEEE format extending the Challenge 1 and Challenge 3
   papers with GAIL results and a three-way algorithmic comparison.

**Environments — same assignment as Challenges 1 and 3**
Each group works on the **same ALE game** assigned in Challenge 1. Preprocessing
must be **identical** to Challenges 1 and 3 (grayscale, 84 × 84 , frame-stack 4, frame-skip 4,
pixels in [0 _,_ 1]) so results are directly comparable across all three challenges.

1. Group 1 — **ALE/MontezumaRevenge-v**
2. Group 2 — **ALE/Pitfall-v**
3. Group 3 — **ALE/PrivateEye-v**
4. Group 4 — **ALE/Gravitar-v**
5. Group 5 — **ALE/Solaris-v**


6. Group 6 — **ALE/Venture-v**
7. Group 7 — **ALE/MsPacman-v**
8. Group 8 — **ALE/Phoenix-v**

```
Theoretical background
```
**Behavioral Cloning (BC)**

BC treats imitation as a supervised learning problem. Given a demonstration dataset
D ={( _si,ai_ )} _Ni_ =1, BC minimises the negative log-likelihood:

```
LBC( θ ) =−
```
### 1

### N

```
∑ N
```
```
i =
```
```
log πθ ( ai | si ) (1)
```
BC is fast to train but suffers from _distributional shift_ : the policy’s own errors accu-
mulate at test time because the learnt distribution does not cover states visited only by the
agent, not the demonstrator.

**Generative Adversarial Imitation Learning (GAIL)**

GAIL frames imitation as a two-player game between a **policy** _πθ_ (the generator)
and a **discriminator** _Dφ_. The discriminator is trained to distinguish expert state-action
pairs from policy-generated ones:

```
max
Dφ
E( s,a )∼D[log Dφ ( s,a )] +E( s,a )∼ πθ [log(1− Dφ ( s,a ))] (2)
```
The policy is then trained with any RL algorithm — here PPO — using the _adversarial
reward_ :

```
r adv( s,a ) =− log(1− Dφ ( s,a )) or equivalently r adv( s,a ) = log Dφ ( s,a ) (3)
```
This signal encourages the policy to produce trajectories that the discriminator can-
not separate from the expert demonstrations, driving the learnt policy towards the expert’s
occupancy measure without requiring the environment’s true reward. The full GAIL objec-
tive is:

```
min π
θ
```
```
max
Dφ
E( s,a )∼ πθ [log Dφ ( s,a )] +E( s,a )∼D[log(1− Dφ ( s,a ))]− λH ( πθ ) (4)
```
```
where H ( πθ ) is the policy entropy regulariser (same as PPO’s entropy bonus).
Training alternates between:
```
- **Discriminator step** : update _Dφ_ with a mini-batch of expert and agent transitions
  using binary cross-entropy.
- **Policy step** : run a PPO update using _r_ advas the reward signal (the environment
  reward is _not_ used).


```
Required implementation elements
```
- **Demonstration collector** : a script that loads a trained checkpoint (DQN from Chal-
  lenge 1 or PPO from Challenge 3) and records a configurable number of ( _st,at_ ) pairs
  into a file (e.g., .npz or pickle).
- **Behavioral Cloning baseline** : a training loop that minimises LBC, followed by
  direct evaluation (no further RL).
- **Discriminator network** : a CNN that takes a stacked-frame observation (and op-
  tionally the action) as input and outputs a scalar in (0 _,_ 1).
- **GAIL training loop** : alternating discriminator updates (binary cross-entropy) and
  PPO updates (using adversarial reward).
- **Reward replacement** : the environment’s true reward must be completely replaced
  by _r_ advduring GAIL training. The true reward is used only for evaluation.
- **Demonstration ablation** : at least two demonstration dataset sizes (e.g., 5 000 and
  50 000 state-action pairs) must be tested to study the effect of demonstration qual-
  ity/quantity.

```
Suggested hyperparameter search space for GAIL
```
- Discriminator learning rate: 1 × 10 −^4 ; 3 × 10 −^4 ; 5 × 10 −^4.
- Discriminator update frequency: 1 update per PPO rollout; 5 updates per PPO
  rollout.
- Discriminator architecture: shared CNN backbone (frozen or jointly trained); separate
  CNN.
- Demonstration dataset size: 5 000; 20 000; 50 000 ( _s,a_ ) pairs.
- Demonstration quality: from best DQN checkpoint vs. from mid-training DQN check-
  point (deliberately imperfect).
- PPO hyperparameters: inherit the best configuration found in Challenge 3 (no need to
  re-sweep these; treat them as fixed unless investigation motivates otherwise).
- Entropy coefficient _λ_ : 0.001; 0.01; 0.02.
- Gradient penalty coefficient (optional, for training stability): 0.0; 10.0.

**Three-way comparison methodology**
The following protocol must be applied so that DQN, PPO, and GAIL results are directly
comparable. All three algorithms must be evaluated under the same conditions.

1. **Budget parity** : fix a total environment step budget (e.g., 5 000 000). For GAIL,
   environment steps count only the steps the _agent_ takes; discriminator gradient updates
   do not count.


2. **Identical preprocessing and evaluation** : same wrappers, same 10-episode deter-
   ministic evaluation at fixed intervals.
3. **Metrics to report for all three algorithms** :
    - Learning curve: episode return vs. environment steps.
    - Sample efficiency: steps to reach the target score threshold defined in Challenge 3.
    - Final performance: mean ± std over 3 seeds at end of training.
    - Training stability: AUC normalised by total steps.
    - For GAIL additionally: discriminator accuracy over training time (does it col-
      lapse? does it stay informative?).
4. **BC baseline** : evaluated without any RL steps; treated as a zero-step proxy for demon-
   stration quality.
5. **Analysis questions** students must address:
    - Does GAIL outperform pure RL (DQN, PPO) on games with sparse rewards? If yes,
      why? If no, why not?
    - How sensitive is GAIL to demonstration quality and quantity?
    - Does the adversarial reward signal remain informative throughout training, or
      does the discriminator collapse?
    - In what regime is BC alone competitive with full GAIL or PPO?

```
Shared implementation: demonstration collector, BC, and GAIL core
The preprocessing wrapper and AtariActorCritic backbone from Challenge 3 apply
unchanged. The following additional components are required.
```
Listing 1: Demonstration collection from a saved checkpoint
1 import numpy as np
2 import torch
3 import gymnasium as gym
4 # Reuse make_env and AtariActorCritic from Challenge 3
5
6 def collect_demonstrations(env_id: str , checkpoint_path: str ,
7 n_steps: int = 20_000 , seed: int = 0,
8 device: str = "cpu") -> dict:
9 """
10 Roll out a saved policy and record (obs , action) pairs.
11
12 Returns a dict with keys ’observations ’ and ’actions ’,
13 each a numpy array of shape (n_steps , ...).
14 """
15 env = make_env(env_id , seed=seed)
16 n_actions = env.action_space.n
17
18 model = AtariActorCritic(n_actions).to(device)
19 model.load_state_dict(torch.load(checkpoint_path , map_location=device
))


20 model.eval()
21
22 obs_buf , act_buf = [], []
23 obs , _ = env.reset()
24 for _ in range(n_steps):
25 obs_t = torch.tensor(obs , dtype=torch.float32).unsqueeze (0).to(
device)
26 with torch.no_grad ():
27 logits , _ = model(obs_t)
28 action = logits.argmax(dim=-1).item() # greedy / deterministic
29
30 obs_buf.append(obs)
31 act_buf.append(action)
32
33 obs , _, terminated , truncated , _ = env.step(action)
34 if terminated or truncated:
35 obs , _ = env.reset()
36
37 env.close()
38 demos = {
39 "observations": np.array(obs_buf , dtype=np.float32),
40 "actions": np.array(act_buf , dtype=np.int64),
41 }
42 np.savez_compressed("demos.npz", ** demos)
43 print(f"Saved {n_steps} demo steps to demos.npz")
44 return demos

Listing 2: Behavioral Cloning (BC) baseline
1 import torch
2 import torch.nn as nn
3 import torch.optim as optim
4 from torch.utils.data import DataLoader , TensorDataset
5
6 def train_bc(env_id: str , demos_path: str = "demos.npz",
7 n_epochs: int = 20, batch_size: int = 256,
8 lr: float = 1e-4, device: str = "cpu"):
9 """
10 Supervised imitation: minimise cross -entropy between
11 demonstrations and policy logits.
12 """
13 data = np.load(demos_path)
14 obs_t = torch.tensor(data["observations"], dtype=torch.float32)
15 act_t = torch.tensor(data["actions"], dtype=torch.long)
16 dataset = TensorDataset(obs_t , act_t)
17 loader = DataLoader(dataset , batch_size=batch_size , shuffle=True)
18
19 env = make_env(env_id)
20 n_actions = env.action_space.n
21 env.close()
22
23 model = AtariActorCritic(n_actions).to(device)
24 optimizer = optim.Adam(model.parameters (), lr=lr)
25 criterion = nn.CrossEntropyLoss ()
26
27 for epoch in range(n_epochs):


28 total_loss = 0.
29 for obs_b , act_b in loader:
30 obs_b , act_b = obs_b.to(device), act_b.to(device)
31 logits , _ = model(obs_b)
32 loss = criterion(logits , act_b)
33 optimizer.zero_grad ()
34 loss.backward ()
35 optimizer.step()
36 total_loss += loss.item()
37 avg = total_loss / len(loader)
38 print(f"BC epoch {epoch +1}/{ n_epochs} loss={avg:.4f}")
39
40 torch.save(model.state_dict (), "bc_policy.pt")
41 return model

Listing 3: Discriminator network for GAIL
1 import torch
2 import torch.nn as nn
3
4 class GAILDiscriminator(nn.Module):
5 """
6 Takes a stacked -frame observation (and optionally a one -hot encoded
7 action) and outputs P(expert | s, a) in (0, 1).
8
9 Using observation only (obs -only variant) is simpler and often
10 sufficient for image -based environments.
11 """
12 def __init__(self , n_actions: int , use_action: bool = False):
13 super().__init__ ()
14 self.use_action = use_action
15 # Shared CNN - same architecture as the policy backbone
16 self.cnn = nn.Sequential(
17 nn.Conv2d(4, 32, kernel_size =8, stride =4), nn.ReLU(),
18 nn.Conv2d (32, 64, kernel_size =4, stride =2), nn.ReLU(),
19 nn.Conv2d (64, 64, kernel_size =3, stride =1), nn.ReLU(),
20 nn.Flatten (),
21 )
22 cnn_out = 64 * 7 * 7 # 3136
23 fc_in = cnn_out + n_actions if use_action else cnn_out
24
25 self.fc = nn.Sequential(
26 nn.Linear(fc_in , 512), nn.Tanh(),
27 nn.Linear (512, 1), nn.Sigmoid (),
28 )
29
30 def forward(self , obs , actions_onehot=None):
31 feats = self.cnn(obs)
32 if self.use_action and actions_onehot is not None:
33 feats = torch.cat([feats , actions_onehot], dim=-1)
34 return self.fc(feats).squeeze (-1)

```
Listing 4: GAIL training loop (PPO as inner RL algorithm)
1 import torch
```

2 import torch.nn.functional as F
3 import torch.optim as optim
4 from torch.distributions import Categorical
5
6 def train_gail(env_id , demos_path="demos.npz",
7 total_steps =5_000_000 , horizon =1024 ,
8 n_ppo_epochs =4, batch_size =128,
9 lr_policy =2.5e-4, lr_disc =3e-4,
10 disc_updates_per_rollout =5,
11 gamma =0.99 , gae_lambda =0.95 ,
12 clip_eps =0.2, ent_coef =0.01 , vf_coef =0.5,
13 max_grad_norm =0.5, seed=42,
14 device="cuda" if torch.cuda.is_available () else "cpu"):
15
16 # --- load demonstrations ---
17 data = np.load(demos_path)
18 demo_obs = torch.tensor(data["observations"], dtype=torch.float32)
19 demo_act = torch.tensor(data["actions"], dtype=torch.long)
20 n_demos = len(demo_obs)
21
22 env = make_env(env_id , seed=seed)
23 n_actions = env.action_space.n
24
25 policy = AtariActorCritic(n_actions).to(device)
26 disc = GAILDiscriminator(n_actions , use_action=False).to(device)
27
28 opt_policy = optim.Adam(policy.parameters (), lr=lr_policy)
29 opt_disc = optim.Adam(disc.parameters (), lr=lr_disc)
30 bce = torch.nn.BCELoss ()
31
32 obs , _ = env.reset()
33 ep_return = 0.
34 all_returns = []
35
36 for global_step in range(0, total_steps , horizon):
37
38 # ---- rollout collection ----
39 obs_buf , act_buf , logp_buf = [], [], []
40 rew_buf , done_buf , val_buf = [], [], []
41
42 for _ in range(horizon):
43 obs_t = torch.tensor(obs , dtype=torch.float32).unsqueeze (0).
to(device)
44 with torch.no_grad ():
45 logits , value = policy(obs_t)
46 dist = Categorical(logits=logits)
47 action = dist.sample ()
48
49 obs_buf.append(obs_t.squeeze (0))
50 act_buf.append(action)
51 logp_buf.append(dist.log_prob(action))
52 val_buf.append(value.squeeze ())
53
54 obs , env_reward , terminated , truncated , _ = env.step(action.
item())


55 done = terminated or truncated
56 done_buf.append(done)
57 ep_return += env_reward
58
59 if done:
60 all_returns.append(ep_return)
61 ep_return = 0.
62 obs , _ = env.reset()
63
64 # ---- adversarial reward (replace env reward) ----
65 obs_stack = torch.stack(obs_buf).to(device)
66 with torch.no_grad ():
67 d_scores = disc(obs_stack) # P(expert | s)
68 # reward: log D(s,a) -- agent wants to look like the expert
69 adv_rewards = torch.log(d_scores + 1e-8).cpu()
70 rew_buf = adv_rewards.tolist ()
71
72 # ---- GAE advantages ----
73 with torch.no_grad ():
74 obs_t = torch.tensor(obs , dtype=torch.float32).unsqueeze
(0).to(device)
75 _, nv = policy(obs_t)
76 advantages , returns = compute_gae(
77 rew_buf , val_buf , done_buf , nv.item(), gamma , gae_lambda
78 )
79
80 # ---- discriminator update ----
81 act_one_hot = F.one_hot(
82 torch.stack(act_buf), n_actions).float().to(device)
83
84 for _ in range(disc_updates_per_rollout):
85 # sample expert mini -batch
86 idx_e = torch.randint(0, n_demos , (batch_size ,))
87 e_obs = demo_obs[idx_e].to(device)
88 # agent mini -batch
89 idx_a = torch.randint(0, horizon , (batch_size ,))
90 a_obs = obs_stack[idx_a]
91
92 d_expert = disc(e_obs)
93 d_agent = disc(a_obs)
94
95 loss_disc = bce(d_expert , torch.ones_like(d_expert)) + \
96 bce(d_agent , torch.zeros_like(d_agent))
97 opt_disc.zero_grad ()
98 loss_disc.backward ()
99 opt_disc.step()
100
101 # ---- PPO update ----
102 act_t = torch.stack(act_buf).to(device)
103 logp_t = torch.stack(logp_buf).detach ().to(device)
104 adv_t = (advantages - advantages.mean()) / (advantages.std() + 1
e-8)
105 ret_t = returns.to(device)
106
107 idx = torch.randperm(horizon)


108 for _ in range(n_ppo_epochs):
109 for start in range(0, horizon , batch_size):
110 mb = idx[start:start + batch_size]
111 lg , vn = policy(obs_stack[mb])
112 dn = Categorical(logits=lg)
113 lp_new = dn.log_prob(act_t[mb])
114 ent = dn.entropy ().mean()
115 ratio = (lp_new - logp_t[mb]).exp()
116
117 s1 = ratio * adv_t[mb]
118 s2 = ratio.clamp(1-clip_eps , 1+ clip_eps) * adv_t[mb]
119 l_pi = -torch.min(s1 , s2).mean()
120 l_vf = ((vn - ret_t[mb]) ** 2).mean()
121 loss = l_pi + vf_coef * l_vf - ent_coef * ent
122
123 opt_policy.zero_grad ()
124 loss.backward ()
125 torch.nn.utils.clip_grad_norm_(policy.parameters (),
max_grad_norm)
126 opt_policy.step()
127
128 if len(all_returns) % 10 == 0 and all_returns:
129 mean_ret = np.mean(all_returns [ -100:])
130 d_acc = (( d_expert > 0.5).float().mean() +
131 (d_agent < 0.5).float().mean()) / 2
132 print(f"step={ global_step} ret={ mean_ret :.1f} "
133 f"disc_loss ={ loss_disc.item():.3f} "
134 f"disc_acc ={ d_acc.item():.2f}")
135
136 env.close()
137 return policy , disc , all_returns

```
Note on compute_gae: reuse the same helper defined (or referenced) in Challenge 3.
The only change is that rew_buf now contains adversarial rewards instead of environment
rewards.
```
```
Per-game starter guides
All groups share the common code above. The starters below specify the recom-
mended initial configuration, the source of demonstrations, and the game-specific hypothesis
students should investigate.
```
```
Group 1 — ALE/MontezumaRevenge-v
```
```
Hypothesis. DQN and PPO both score near 0 due to extreme reward sparsity. GAIL
bypasses the reward entirely — if the demonstration policy visits Room 1, the adversarial
reward will implicitly guide the agent there even without a game score signal. Students
should test whether GAIL is the first of the three algorithms to consistently enter Room 1.
```
```
Listing 5: Group 1 — Montezuma’s Revenge GAIL starter
1 # Step 1: collect 50,000 steps from the best \texttt{DQN} checkpoint
2 collect_demonstrations(
3 env_id = "ALE/MontezumaRevenge -v5",
```

4 checkpoint_path = "challenge1/group1/best_dqn.pt",
5 n_steps = 50_000 ,
6 )
7
8 # Step 2: BC baseline (how far does supervised cloning get alone?)
9 train_bc(
10 env_id = "ALE/MontezumaRevenge -v5",
11 demos_path = "demos.npz",
12 n_epochs = 30,
13 )
14
15 # Step 3: \texttt{GAIL} - high entropy coefficient for residual
exploration
16 policy , disc , returns = train_gail(
17 env_id = "ALE/MontezumaRevenge -v5",
18 demos_path = "demos.npz",
19 total_steps = 5_000_000 ,
20 horizon = 2048,
21 disc_updates_per_rollout = 5,
22 ent_coef = 0.02, # entropy crucial: demos are sparse
23 seed = 42,
24 )
25 # Ablation: try demos from mid -training \texttt{DQN} (worse quality) vs
best.
26 # Key metric: first step at which agent enters Room 1.

```
Group 2 — ALE/Pitfall-v
```
```
Hypothesis. Even an imperfect DQN demonstrator that survives a few seconds pro-
vides a meaningful occupancy measure. GAIL should learn to avoid the worst traps faster
than PPO because the adversarial reward penalises states never visited by the demonstrator.
```
Listing 6: Group 2 — Pitfall! GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/Pitfall -v5",
3 checkpoint_path = "challenge1/group2/best_dqn.pt",
4 n_steps = 20_000 , # Pitfall demos are short - collect less
5 )
6
7 train_bc(env_id="ALE/Pitfall -v5", n_epochs =20)
8
9 policy , disc , returns = train_gail(
10 env_id = "ALE/Pitfall -v5",
11 total_steps = 5_000_000 ,
12 horizon = 1024,
13 disc_updates_per_rollout = 3,
14 ent_coef = 0.01,
15 gamma = 0.995 ,
16 seed = 42,
17 )
18 # Ablation: 5,000 demos vs 20,000 demos.
19 # Measure: minimum ’negative reward episodes ’ per 100 training episodes.


```
Group 3 — ALE/PrivateEye-v
```
```
Hypothesis. Private Eye is so difficult that even the DQN demonstrator likely scores
```
0. This group should study the _failure case_ of GAIL: if demonstrations are uninformative
   (zero-score trajectories), can the adversarial signal still provide useful guidance? Compare
   BC, GAIL, and PPO all scoring 0 and reason about why.

Listing 7: Group 3 — Private Eye GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/PrivateEye -v5",
3 checkpoint_path = "challenge1/group3/best_dqn.pt",
4 n_steps = 50_000 ,
5 )
6
7 train_bc(env_id="ALE/PrivateEye -v5", n_epochs =20)
8
9 policy , disc , returns = train_gail(
10 env_id = "ALE/PrivateEye -v5",
11 total_steps = 5_000_000 ,
12 horizon = 2048,
13 disc_updates_per_rollout = 5,
14 ent_coef = 0.02,
15 seed = 42,
16 )
17 # KEY experiment: plot discriminator accuracy over time.
18 # If D collapses to 0.5, the signal is uninformative -- document this.
19 # Compare discriminator accuracy across all three games as a meta -
analysis.

```
Group 4 — ALE/Gravitar-v
Hypothesis. Gravitar requires maintaining specific thrust sequences. GAIL may
learn the style of control (e.g., constant small thrusts) from the discriminator before the
agent discovers any reward. Test whether GAIL’s BC warm-start helps stabilise early
training compared to PPO from scratch.
```
Listing 8: Group 4 — Gravitar GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/Gravitar -v5",
3 checkpoint_path = "challenge1/group4/best_dqn.pt",
4 n_steps = 30_000 ,
5 )
6
7 # Initialise GAIL policy with BC weights for a warm -start
8 bc_model = train_bc(env_id="ALE/Gravitar -v5", n_epochs =25)
9
10 policy , disc , returns = train_gail(
11 env_id = "ALE/Gravitar -v5",
12 total_steps = 5_000_000 ,
13 horizon = 1024,
14 disc_updates_per_rollout = 3,
15 ent_coef = 0.01,


16 seed = 42,
17 )
18 # Ablation: GAIL with BC warm -start vs GAIL from random init.
19 # Metric: action histogram -- does the agent learn thrust patterns?

```
Group 5 — ALE/Solaris-v
Hypothesis. Multi-stage games may benefit most from imitation because demon-
strations implicitly encode when to switch strategy. GAIL’s occupancy-measure matching
should learn stage-transition behaviour better than epsilon-greedy exploration. Students
should count how many distinct in-game stages each algorithm reaches.
```
Listing 9: Group 5 — Solaris GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/Solaris -v5",
3 checkpoint_path = "challenge1/group5/best_dqn.pt",
4 n_steps = 50_000 ,
5 )
6
7 train_bc(env_id="ALE/Solaris -v5", n_epochs =25)
8
9 policy , disc , returns = train_gail(
10 env_id = "ALE/Solaris -v5",
11 total_steps = 5_000_000 ,
12 horizon = 2048,
13 disc_updates_per_rollout = 5,
14 gamma = 0.995 ,
15 gae_lambda = 0.97,
16 ent_coef = 0.01,
17 seed = 42,
18 )
19 # Track: maximum in -game stage reached per episode.
20 # Compare with DQN and PPO on this metric.

```
Group 6 — ALE/Venture-v
```
```
Hypothesis. Venture’s heavy penalty for taking damage may cause the GAIL agent
to over-imitate the demonstrator’s conservative behaviour and never explore new dungeons.
Students should test whether increasing the entropy bonus or reducing demonstrations
quality forces GAIL to deviate from the demonstrator and find more reward.
```
```
Listing 10: Group 6 — Venture GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/Venture -v5",
3 checkpoint_path = "challenge1/group6/best_dqn.pt",
4 n_steps = 20_000 ,
5 )
6
7 train_bc(env_id="ALE/Venture -v5", n_epochs =20)
8
9 policy , disc , returns = train_gail(
```

10 env_id = "ALE/Venture -v5",
11 total_steps = 5_000_000 ,
12 horizon = 1024,
13 disc_updates_per_rollout = 3,
14 clip_eps = 0.1,
15 ent_coef = 0.02,
16 seed = 42,
17 )
18 # Ablation: high -quality demos (best DQN) vs low -quality (early DQN
checkpoint).
19 # Question: does lower -quality demo lead to more dungeon exploration?

```
Group 7 — ALE/MsPacman-v
Hypothesis. Ms. Pac-Man has relatively dense rewards. Here GAIL may converge
more slowly than PPO or DQN because the adversarial reward is a proxy rather than the
true game signal. Students should study whether BC alone — which can imitate good
movement patterns — is competitive with full PPO or GAIL on this denser-reward game.
```
Listing 11: Group 7 — Ms. Pac-Man GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/MsPacman -v5",
3 checkpoint_path = "challenge1/group7/best_dqn.pt",
4 n_steps = 50_000 ,
5 )
6
7 # BC is expected to perform well here -- evaluate it rigorously
8 bc_model = train_bc(env_id="ALE/MsPacman -v5", n_epochs =30, lr=1e-4)
9
10 policy , disc , returns = train_gail(
11 env_id = "ALE/MsPacman -v5",
12 total_steps = 5_000_000 ,
13 horizon = 1024,
14 disc_updates_per_rollout = 3,
15 ent_coef = 0.01,
16 seed = 42,
17 )
18 # PRIMARY COMPARISON: BC score vs GAIL vs PPO vs DQN on this game.
19 # This is the richest four -way comparison in the challenge series.

```
Group 8 — ALE/Phoenix-v
Hypothesis. Phoenix requires fast reactions. A discriminator that evaluates sin-
gle frames (no action context) may struggle to capture reactive timing. Students should
compare a state-only discriminator (use_action=False) with a state-action discriminator
(use_action=True) to test whether including action context helps the adversarial reward
signal.
```
```
Listing 12: Group 8 — Phoenix GAIL starter
1 collect_demonstrations(
2 env_id = "ALE/Phoenix -v5",
```

3 checkpoint_path = "challenge1/group8/best_dqn.pt",
4 n_steps = 30_000 ,
5 )
6
7 train_bc(env_id="ALE/Phoenix -v5", n_epochs =20)
8
9 # Ablation A: state -only discriminator
10 policy_a , disc_a , ret_a = train_gail(
11 env_id = "ALE/Phoenix -v5",
12 horizon = 512,
13 n_ppo_epochs= 10,
14 seed = 42,
15 )
16
17 # Ablation B: modify GAILDiscriminator(use_action=True) and retrain
18 # compare learning curves of A vs B to assess action -context benefit

```
Deliverables
```
- **Repository folder:** Add challenge4/group<k>/ to the same GitHub repository.
  Include all GAIL source code, the demonstration collection script, the BC training
  script, a README.md with exact run instructions, and logging artifacts for all three
  algorithms (DQN, PPO, GAIL).
- **Extended IEEE paper:** Extend the Challenge 1 (and 3) paper to include GAIL
  results and the full three-way comparison. The paper is limited to 10 pages (excluding
  references). Submit as challenge4_group<k>_paper.pdf.
- **Demonstration dataset metadata:** Include a short plain-text or JSON file
  (demos_info.txt) describing the source checkpoint used, the number of steps col-
  lected, and the mean/std return of the demonstrating policy.
- **Checklist:** A CHECKLIST.md inside the repository folder with:
  **-** Exact commands to collect demonstrations, train BC, and train GAIL.
  **-** Seeds used for all repeated experiments.
  **-** Pointers to logs and figures for DQN, PPO, and GAIL.
  **-** A 200-word comparative summary: under what conditions did GAIL add value
  over pure RL on this specific environment?

```
Evaluation criteria
```
- **Implementation correctness (25%):** BC baseline is correct; discriminator and
  GAIL loop are properly implemented; adversarial reward replaces environment reward
  during training.
- **Experimental rigour (25%):** demonstration ablation is performed (at least two
  dataset sizes), variance is reported over 3 seeds, and all three algorithms use identical
  evaluation conditions.


- **Comparison quality and analysis (35%):** the three-way comparison (DQN, PPO,
  GAIL) is fair, the four analysis questions are addressed with empirical evidence, and
  the discriminator dynamics are discussed.
- **Presentation and writing (15%):** quality of the extended IEEE paper (updated
  figures, tables, related work on imitation learning, and updated conclusions covering
  all three challenges).

**Notes on scope and computational budget**
GAIL requires evaluating the discriminator on both expert and agent batches at
every rollout, which adds modest overhead. Discriminator gradient updates are fast but
accumulate over training. If compute is limited:

- Reduce total_steps to 2 000 000 and document this constraint.
- Use a frozen CNN for the discriminator (only train the fully-connected head) to reduce
  memory and compute.
- Prioritise the 3-seed DQN vs PPO vs GAIL comparison over wide hyperparameter sweeps.

**References and further reading**
Core references for this challenge: Ho & Ermon (2016, GAIL), Fu et al. (2018, AIRL),
Ross et al. (2011, DAgger), Goodfellow et al. (2014, GANs), Schulman et al. (2017, PPO),
and Mnih et al. (2015, DQN). Students should additionally consult recent survey articles on
imitation learning and offline RL. All citations must be in IEEE style.

_Challenge 4 closes the three-algorithm arc of this course. DQN explored the value-based
paradigm; PPO introduced on-policy actor-critic methods; GAIL asks whether an agent can
learn to behave well without ever being told what “well” means via a numerical reward.
Answering this question rigorously — not just empirically but conceptually — is the goal of
this final challenge._

