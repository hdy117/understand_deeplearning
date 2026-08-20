# 第 19 章：强化学习（Reinforcement Learning）

> 书：《Understanding Deep Learning》Ch.19  
> 前面章节：监督 / 生成学习通常在**给定**的数据上拟合 objective。  
> 本章：agent 的 action 会改变未来 state、reward，以及自己接下来能看见的数据；怎样学习长期决策？  
> 贯穿例子：企鹅在冰面上避开洞、寻找鱼。

---

## 0. 数据不再是事先摆好的

监督学习是：

```text
拿到固定数据 (x, y) → 预测 y → 立刻得到 loss
```

强化学习是一个闭环：

```text
Environment 给 state
        ↓
Agent 选 action
        ↓
Environment 改变：产生 reward 和 next state
        ↓
Agent 再选
        ↓
持续循环
```

棋类把难点一次说完。整盘可能只有终局的 \(+1/-1/0\)，中间三十步都是 0——reward 稀疏。决定胜负的那步棋可能发生在三十步以前——这是 temporal credit assignment。对手不会在同一局面永远走同一手——环境随机。为了发现更好的开局，又必须去下那些目前看起来并不最优的棋——exploration vs exploitation。

因此母问题不是“拟合一个固定数据集”，而是：

> **怎样选择当前 action，使未来一整段 trajectory 的 expected cumulative reward 最大？**

更尖锐的那句是：

> **Policy 不但决定现在怎么做，还决定下一批评训数据从哪里来。**

后面每个算法都是在这条闭环上补一个缺口：

```text
Action 会改变未来数据
        → 用 MDP 把 state / action / transition / reward 说清楚
        → 一步 reward 太短视，所以定义 return
        → return 的期望形成 value
        → return 递归展开得到 Bellman
        → 知道完整 model → Dynamic Programming
        → 不知道 model、只能等完整 episode → Monte Carlo
        → 不想等终点 → TD：一步 sample + bootstrap
        → 要控制：SARSA 跟实际行为走，Q-learning 跟 greedy 走
        → Q table 存不下 → fitted Q / DQN
        → 函数近似 + bootstrap + off-policy 叠在一起可能炸
        → 直接参数化 policy → Policy Gradient
        → REINFORCE 方差太高 → baseline / Actor–Critic
        → 不能继续与环境互动 → Offline RL，coverage 成为信息边界
```

这不是方法清单。每一行都是上一行做不到的那件事。

---

## 1. Agent–Environment Loop：一次交互到底留下什么

时刻 \(t\)：

```text
state s_t
   ↓ policy π(a|s)
action a_t
   ↓ environment dynamics
reward r_{t+1}, next state s_{t+1}
```

一条 trajectory：

\[
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots).
\]

企鹅例子把每个符号钉死：

```text
State：  企鹅当前所在的冰格
Action： 上、下、左、右
Reward： 吃到鱼为正；掉进洞为负；走路可能有小代价
Transition：冰面滑，动作不一定精确执行
```

Policy \(\pi(a\mid s)\) 是“在这个 state 选各 action 的分布”。Deterministic 时每个 state 固定一个 action；stochastic 时给出概率。后者对探索和连续动作几乎是必需的。

一次真实交互只给一条 sampled transition：

\[
(s_t,a_t,r_{t+1},s_{t+1},done).
\]

它不会把真实 \(q_\pi(s_t,a_t)\) 写在标签里。Value learning 的全部别扭都在这里：

```text
手里只有一条随机经历
        ↓
却要估计从这个 state / action 出发的所有可能未来的期望
```

`done=True` 表示 episode 已终止，后面没有 next-state future return。若终端仍把 \(V(s')\) 加进去，等于给已经结束的故事续写并不存在的未来。

| 符号 | 含义 | 谁产生 |
|------|------|--------|
| \(s_t\) | 当前 state | environment；或由 history 构造 |
| \(a_t\) | 当前 action | behavior policy 抽样 |
| \(r_{t+1}\) | 这一步之后的 reward | environment |
| \(s_{t+1}\) | next state | transition |
| \(\pi(a\mid s)\) | 各 action 的概率 | agent 参数 |
| \(G_t\) | 从 \(t\) 起的完整 future return | 整段后续 trajectory |
| \(v_\pi,q_\pi\) | expected return | 对未来的随机性取期望 |

下一节把“环境”收成一个数学对象。没有这个对象，后面的 Bellman、DP、off-policy 都没有地方站。

---

## 2. 从 Markov Process 到 MDP：state 必须够用

Markov 假设不是“历史不重要”，而是：当前 state 已经把预测未来所需的历史装进去了。

\[
p(s_{t+1}\mid s_t,s_{t-1},\ldots)
=
p(s_{t+1}\mid s_t).
\]

企鹅在格子 6，若格子编号就是全部信息，下一步只依赖“现在在 6”，不依赖它是从 2 滑过来还是从 10 滑过来。

加上 reward，变成 Markov Reward Process：\(p(r_{t+1},s_{t+1}\mid s_t)\)。再让 agent 能选 action，才是 Markov Decision Process：

\[
p(s_{t+1},r_{t+1}\mid s_t,a_t).
\]

MDP 的零件：

\[
(\mathcal S,\mathcal A,P,R,\rho_0,\gamma)
\]

——状态、动作、转移、奖励、初始分布、折扣。还要说清 episode 从哪开始、何时结束。Stationary policy \(\pi(a\mid s)\) 不显式依赖时间；有限视野任务若剩余步数没放进 state，最优策略可能必须看 \(t\)。这些细节不是学究，后面 `done` 截断 bootstrap 就靠它们。

现实里 agent 常常看不见 true state，只能看见 observation \(o_t\sim p(o_t\mid s_t)\)。这是 POMDP。一张游戏截图可能有位置、没有速度；单帧不够预测下一帧。堆叠 frames 或用 recurrent state，是在把 history 手工塞回“近似 Markov 的 state”。Atari 上的 DQN 后面会回到这件事。

State 定义错了，后面所有 value 都在估一个答非所问的期望。先把对象说对，再谈“长期”。

---

## 3. Reward 是一步，Return 才是目标

Reward 是瞬时反馈 \(r_{t+1}\)。企鹅这一步没吃到鱼、也没掉洞，reward 可能是走路的小负数。若目标是最大化眼前这一步，它会站着不动或者乱走。

真正最大化的是从现在起的 discounted cumulative future reward：

\[
\boxed{
G_t
=
\sum_{k=0}^{\infty}\gamma^k r_{t+k+1},
\qquad
0\le\gamma\le 1.
}
\]

\(\gamma\) 小则短视；接近 1 则看长期。Continuing task 里 rewards 有界且 \(\gamma<1\)，无穷和才保证有限。有限 episode 里 \(\gamma=1\) 常可用，因为故事会结束；无限视野只写 \(\gamma=1\) 并不自动收敛。

未来 rewards 为 \([1,0,2]\)、\(\gamma=0.9\) 时：

\[
G_t=1+0.9\cdot 0+0.9^2\cdot 2=2.62.
\]

一个动作眼前扣分，若能绕开冰洞、最终吃到鱼，长期仍可能更好。Credit assignment 的数学座位就是 \(G_t\)：终局的鱼，会贴现回当初那一步转向。

Return 仍是一条轨迹上的随机数。同一个 state 出发，冰面一滑、鱼一刷新，\(G_t\) 就不一样。我们真正比较“这个格子好不好”时，要比的是它的期望。

---

## 4. Value：把随机的 return 收成一张表

State value：从 \(s\) 出发、之后一直按 \(\pi\) 走，平均能拿到多少 return。

\[
\boxed{
v_\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].
}
\]

Action value：在 \(s\) 先做 \(a\)，之后再按 \(\pi\) 走，平均 return 是多少。

\[
\boxed{
q_\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
}
\]

```text
vπ(s)：这个格子整体有多好（已经把 π 会选的动作平均进去了）
qπ(s,a)：在这个格子做这个动作有多好
```

若有最优 action values，决策退化成查表：

\[
\pi^*(s)=\arg\max_a q^*(s,a).
\]

这就是后来 Q-learning 的诱惑：把 \(q\) 学对，policy 几乎免费。诱惑的另一面是——\(q\) 是期望，手里却永远只有一条样本轨迹。下一节把期望写成自己和自己的关系，才有地方用一步经验去更新一张表。

---

## 5. Bellman：长期价值不是额外公理，是 return 的拆开

Return 自己就能拆：

\[
G_t=r_{t+1}+\gamma G_{t+1}.
\]

两边取期望，state value 变成：

\[
\boxed{
v_\pi(s)
=
\mathbb E_{a\sim\pi,\,s'\sim P}
\bigl[r(s,a)+\gamma v_\pi(s')\bigr].
}
\]

Action value：

\[
\boxed{
q_\pi(s,a)
=
\mathbb E
\bigl[
r(s,a)+\gamma\mathbb E_{a'\sim\pi}q_\pi(s',a')
\bigr].
}
\]

最优时，下一步不再跟 \(\pi\) 平均，而是取最好的：

\[
\boxed{
q^*(s,a)
=
\mathbb E
\bigl[
r(s,a)+\gamma\max_{a'}q^*(s',a')
\bigr].
}
\]

读成一句人话：

> **当前动作的价值 = 立即 reward + 打折后的下一格潜力。**

Bellman equation 是对所有可能 next states、rewards、后续动作的 **expectation 关系**。实际交互通常只看见一条 \((s,a,r,s',a')\)，于是用 sampled one-step target

\[
y_{\mathrm{TD}}=r+\gamma\widehat q(s',a'),
\qquad
\delta_{\mathrm{TD}}=y_{\mathrm{TD}}-\widehat q(s,a),
\]

再让估计朝 target 挪一小步：

\[
\widehat q(s,a)\leftarrow\widehat q(s,a)+\alpha\delta_{\mathrm{TD}}.
\]

\(y_{\mathrm{TD}}\) 是 Bellman 期望的 noisy estimate，不是真实 value 标签。Target 里又用了另一个 learned estimate \(\widehat q(s',a')\)，这叫 **bootstrapping**：用已有估计更新当前估计。好处是不必等 episode 结束；代价是 target 有偏，而且会随着模型更新自己移动。

三个东西不要黏成一个词：

```text
Bellman equation：全体可能未来的自洽关系
Bellman target：  一条样本上的 r + γ·(下一个估计)
TD error：        target 减当前估计，用来挪一步
```

还没谈完“挪哪一步”之前，必须先面对一个更先验的问题：如果永远选当前看起来最好的动作，你可能根本收不到该挪的那些格子上的数据。

---

## 6. Exploration：不是加噪声，是在决定能看见什么证据

总选 \(\arg\max_a q(s,a)\)，企鹅可能永远发现不了冰面另一侧的鱼。Exploitation 使用当前最优知识；exploration 尝试尚不确定的 actions，换信息。

\(\epsilon\)-greedy 是最粗糙的折中：

```text
概率 1-ε：选当前 argmax
概率 ε：  随机选一个 action
```

\(\epsilon\) 太小，困在错策略；太大，持续乱走。和监督学习的本质差别在这里：分类器的数据集不会因为你今天预测错了就少一类样本；RL 里不探索，未选过的 \((s,a)\) 就没有数据。后面 offline RL 会把这句话推到极限——历史数据里没做过的动作，价值不可识别。

有了“必须探索”和“value 是期望”，可以谈表格方法。表格假设 state/action 少到能为每一格存一个数字。它们的差别不在公式漂不漂亮，而在信息从哪来。

---

## 7. 表格方法：model、整段回报、还是一步 bootstrap

每个 pair 一格：\(Q[s,a]\)。

若知道完整 \(P(s'\mid s,a)\) 和 \(r(s,a)\)，不必和环境试错，可以在脑子里做 Dynamic Programming：反复 policy evaluation（按 Bellman 更新 value）和 policy improvement（改选 value 更高的动作）。这是 model-based。企鹅若有一张精确冰图，DP 就能算出每格该往哪走。现实里这张图通常没有。

不知道 model，就去跑完整 episode，用实际拿到的 \(G_t\) 做平均。这是 Monte Carlo：

\[
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)+\alpha\bigl[G_t-Q(s_t,a_t)\bigr].
\]

优点：不 bootstrap，target 来自真实完整 return。缺点：必须等终点，variance 高。\(G_t\) 仍只是期望的一个 noisy sample，不是无噪声标签——冰面滑一次，同一格的回报可以差很多。

不想等终点，就走 Temporal Difference：观察一步，用 \(r+\gamma\cdot\)（下一个估计）更新。TD(0) 的 state-value 形式：

\[
V(s_t)
\leftarrow
V(s_t)
+\alpha\bigl[r_{t+1}+\gamma(1-d_t)V(s_{t+1})-V(s_t)\bigr].
\]

不必等结束，但 target 含当前估计，带 bias，也可能不稳。

| 方法 | Target | 等终点？ | Bootstrap？ |
|------|--------|----------|-------------|
| MC | \(G_t\) | 是 | 否 |
| TD(0) | \(r+\gamma V(s')\) | 否 | 是 |
| SARSA | \(r+\gamma Q(s',a'_{\mathrm{behavior}})\) | 否 | 是 |
| Q-learning | \(r+\gamma\max_{a'}Q(s',a')\) | 否 | 是 |

评价一个 policy 和学习一个更好的 policy 还不是同一件事。下一节的 SARSA / Q-learning 差在：bootstrap 时，你假装未来会怎么走。

---

## 8. SARSA 跟行为走，Q-learning 跟 greedy 走

SARSA 的 target 使用政策**实际**选的下一步 \(a_{t+1}\)：

\[
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+\alpha\bigl[r_{t+1}+\gamma(1-d_t)Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\bigr].
\]

它学的是当前 behavior policy 的价值。探索带来的风险会写进 Q：企鹅若 \(\epsilon\)-greedy 会在洞边乱走，SARSA 会把这格看得更危险。

Q-learning 把下一步换成 greedy 的 max：

\[
\boxed{
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+\alpha\bigl[r_{t+1}+\gamma(1-d_t)\max_a Q(s_{t+1},a)-Q(s_t,a_t)\bigr].
}
\]

Behavior 可以继续探索，target 却假设未来会选最好的，因此在合适条件下趋向 optimal policy。条件包括：tabular finite MDP、每个 pair 被充分访问、合适的递减学习率、折扣或适当的 episode。换上非线性函数近似之后，这些保证不再一般成立。

同一条经历，两个 target 可以差一截。设 \(r=1\)，到达 \(s_{t+1}\) 后

```text
Q(s_{t+1}, safe)    = 4
Q(s_{t+1}, explore) = 1
```

behavior 为了探索实际选了 `explore`，\(\gamma=0.9\)：

\[
y_{\mathrm{SARSA}}=1+0.9\cdot 1=1.9,
\qquad
y_{Q}=1+0.9\cdot 4=4.6.
\]

```text
同一条 (s,a,r,s')
     ├── SARSA 看行为实际选了什么 → 1.9
     └── Q-learning 看 greedy 会选什么 → 4.6
```

On-policy / off-policy 的核心就是这句话，不只是公式里有没有 `max`。

再走一步更新的算术。\(Q(s_t,a_t)=2,\,r=1,\,\gamma=0.9,\,\max Q=4,\,\alpha=0.1\)：

\[
y=4.6,\quad \delta=2.6,\quad Q_{\mathrm{new}}=2+0.1\cdot 2.6=2.26.
\]

估计只向“一步 reward + 下一格估计”挪了一小段。表格世界里，这段小步在访问足够时会走稳。格子多到存不下时，同一套 target 会碰到完全不同的麻烦。

---

## 9. 表变成网络：moving target 和错误的终点

大型 state 无法建表，改用 \(q_\theta(s,a)\)。离散动作常一次输出所有 Q：

```text
state s → net → [Q(s,a₁), …, Q(s,a_K)]
```

Loss 看起来像监督回归：

\[
\mathcal L(\theta)=\bigl(y-q_\theta(s,a)\bigr)^2,
\qquad
y=r+\gamma\max_{a'}q_\theta(s',a').
\]

但 \(y\) 不是固定标签。当前一步把 \(y\) 当 stop-gradient；下一轮 \(\theta\) 一变，\(y\) 跟着变。数据又来自连续轨迹、强相关。这是 vanilla fitted Q 不稳的根。

终止必须截断 bootstrap：

\[
\boxed{
y=r+\gamma(1-\mathrm{done})\max_{a'}q_\theta(s',a').
}
\]

`done=1` 时 target 只剩 \(r\)。还要区分环境真正结束（掉进洞、吃到鱼）和因为 time limit 被 wrapper 截断。通常只对真正 `terminated` 截断；time-limit truncation 是否继续 bootstrap，取决于剩余时间有没有进 state。现代 API 把两者分开返回，不要把旧式 `done` 一律当成同一种终点。

表格 Q-learning 的收敛故事到此结束。Deep 网络要另请两个工程机制，外加承认一个叫 deadly triad 的叠加风险。

---

## 10. DQN：两帖药，以及为什么仍可能炸

Experience replay 把 \((s,a,r,s',done)\) 写入 buffer，训练时随机抽 mini-batch。连续样本的相关性被打散，经验被复用，不同时期的行为混在一起。Buffer 仍会随 behavior 漂移，不是真正的 i.i.d. 数据集；数据来自过去的 policies，所以 DQN 本质上在用 off-policy 数据。

Target network 用较慢更新的 \(\theta^-\) 算 TD target：

\[
y=r+\gamma(1-\mathrm{done})\max_{a'}q_{\theta^-}(s',a').
\]

Online 网络频繁更新，target 延迟复制或软更新，避免 target 每一步都跟 prediction 一起跑。

单帧图像通常看不到速度，构不成充分 Markov state。Atari 的经典做法是堆叠相邻 frames，让网络自己推断运动——第 2 节的 POMDP 在工程里的落点。

把三件事叠在一起，文献称为 deadly triad：

```text
Function approximation：用网络泛化 value
Bootstrapping：         target 使用另一个 learned estimate
Off-policy learning：   收数据的 policy 与被评价的 policy 不同
```

单独都不致命；同时存在时，误差会互相放大：

```text
OOD / 被噪声抬高的 Q
 → 被 max 选成 target
 → 网络泛化到更多没见过的 (s,a)
 → 新 target 又建立在已经偏掉的网络上
 → 正反馈
```

Replay、target network、Double DQN、gradient clipping 都是稳定化工程，不是“神经网络 Q-learning 已经继承表格收敛保证”。

Max 操作还有一个更具体的偏差。多个 action 的估计都有噪声时，\(\mathbb E[\max_a\hat Q_a]\) 会高于真实 maximum——谁碰巧被高估，谁就被选中。根因是同一套估计既负责**选**哪个动作，又负责**评**它值多少。

Tabular Double Q-learning 用两个估计器，一个选、一个评，交替更新。Deep 里 Double DQN 借用 online / target 做同一件事：

\[
a^*=\arg\max_a q_\theta(s',a),
\qquad
y=r+\gamma(1-\mathrm{done})q_{\theta^-}(s',a^*).
\]

Online 选动作，target 评价动作。Target network 稳定 target，Double 拆开 selection / evaluation，两帖药治的不是同一种病。

Q-learning 这条路适合离散动作：先估价值再取 max。连续动作的 max 不好做，随机策略也不好从一张 Q 表里长出来。于是换一条路：不要先学 \(q\)，直接学 \(\pi\)。

---

## 11. Policy Gradient：环境不必可微

直接参数化 \(\pi_\theta(a\mid s)\)，目标是轨迹回报的期望：

\[
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G(\tau)].
\]

Policy gradient theorem 的 Monte Carlo 形式：

\[
\boxed{
\nabla_\theta J
=
\mathbb E
\left[
\sum_t G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
}
\]

某动作后来得到高 return，就增加它在相似 state 下的 log probability。Likelihood-ratio / log-trick 把“轨迹分布依赖 \(\theta\)”转成对 log-policy 的梯度，**environment dynamics 不必可微**。冰面怎么滑、鱼怎么刷新，都可以当黑盒。

原始 REINFORCE 用完整 episode 的 \(G_t\) 当权重。若回报总为正，sampled action 通常仍被强化，只是力度不同——“较小但仍为正”并不会自动降低概率。真正让好坏分家的是减去一个不依赖当前动作的 baseline：

\[
\boxed{
\nabla_\theta J
=
\mathbb E
\left[
\sum_t (G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
}
\]

一行证明期望不变：

\[
\mathbb E_{a\sim\pi}\bigl[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\bigr]
=
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)
=0,
\]

因为概率和恒为 1。Baseline 改变单次抽样的梯度，不改变它的期望，方差却可以下降。最自然的 \(b(s)\) 就是 \(v(s)\)。差值

\[
A(s,a)=q(s,a)-v(s)
\]

叫 advantage：这个动作比该 state 的平均水平好多少。\(A>0\) 提高 \(\log\pi(a\mid s)\)；\(A<0\) 降低。代码里 optimizer 默认 descent，所以常写成

\[
\mathcal L_{\mathrm{actor}}=-A(s,a)\log\pi_\theta(a\mid s).
\]

负号来自框架，不是算法想降低 return。

REINFORCE 等完整回报，bias 低、variance 高。用一个 critic 把 \(G_t\) 换成 bootstrapped advantage，就得到 Actor–Critic。

---

## 12. Actor–Critic：一个人选，一个人评

```text
Actor πθ(a|s)：决定做什么
Critic vφ(s) 或 qφ(s,a)：评价局势 / 动作
```

一步 TD advantage：

\[
\delta_t
=
r_{t+1}+\gamma(1-\mathrm{done}_t)v_\phi(s_{t+1})-v_\phi(s_t).
\]

Actor 用 \(\delta_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\) 更新；critic 让 \(v_\phi(s_t)\) 靠近同一个 bootstrapped target。优化 \(v\) 时右侧通常 stop-gradient；actor 把 \(\delta_t\) 当外部权重，不让 policy gradient 随便穿过 critic。否则更新方向会混进 theorem 里没有的路径。

| 方法 | Target | 何时更新 | 主要性格 |
|------|--------|----------|----------|
| REINFORCE | 完整 MC return | episode 后 | 低 bias、高 variance |
| Actor–Critic | bootstrapped value | 可逐步 | 方差较低，引入 bias |

两条分类轴不要揉成一个词。Model-based 显式使用或学习 \(P,R\)，可以 planning；model-free 直接学 value / policy。On-policy 学的就是正在收数据的那套政策；off-policy 可以用别人收的数据学另一套 target policy。

| 方法 | Model | Policy 关系 |
|------|-------|-------------|
| DP | based | 依算法 |
| SARSA | free | on-policy |
| Q-learning / DQN | free | off-policy |
| REINFORCE | free | on-policy |

Online 算法至少还能再试一次。有些环境不许试：自动驾驶、医疗、金融。数据已经写死，coverage 变成硬边界。

---

## 13. Offline RL：没做过的动作，价值不可识别

只有历史

\[
D=\{\tau_i\},
\qquad
\tau_i=(s_0,a_0,r_1,\ldots),
\]

训练期不能与环境互动。Value 方法把轨迹拆成 tuples；sequence model 则要保住顺序。

核心危险是 distribution shift：dataset 只覆盖 behavior 常做的动作，学出来的政策却可能选数据集之外的动作；Q 在那些区域没有证据，却可能高估。所以 offline 方法往往对 OOD action 保守，或把 learned policy 拴在 behavior 附近。

一个不可识别反例。Behavior 永远选 `safe`，reward 都是 \(+1\)，从未选 `risky`。两个环境与这份数据完全一致：

```text
MDP A：risky = +10
MDP B：risky = -10
```

Observed dataset 相同，最优动作相反。因此：

> **没有 action coverage 或额外结构假设，任何 offline 算法都不可能只凭这份数据判断 unseen action 的真实价值。**

Online off-policy 至少还能回去补数据；offline 不能主动制造反事实。这比“普通 off-policy”更硬。

Decision Transformer 把轨迹排成 `(return-to-go, state, action)`，用 causal Transformer 预测下一步动作。训练用数据集里真实的 \(R_t=\sum_{k=t}^{T}r_k\)；inference 给定 desired return \(R_0\)，环境返回 \(r_1\) 后令 \(R_1=R_0-r_1\) 再继续。这是 return-conditioned sequence modeling，不是凭空最优控制。Desired return 超出 dataset support，模型不必知道如何实现；监督 loss 稳定，也消除不了闭环 OOD execution。

Offline RL 也不等于 imitation：imitation 主要复制动作，未必用 reward；offline RL 用 reward，目标可以超过 behavior，风险也更高。

---

## 14. 贯穿始终的几组对打

Immediate vs delayed：\(\gamma\) 大则看长，value 更难估。  
Exploration vs exploitation：信息要用 reward 或安全去换。  
Bias vs variance：MC 用真实采样回报，方差高；TD bootstrap 方差较低，估计不准时 target 有偏。满足条件的表格 TD 仍可收敛，不是永久偏差。  
Value-based vs policy-based：Q-learning 适合离散动作；policy gradient 直接对付随机 / 连续动作，方差高；actor–critic 绑在一起。  
Online vs offline：能探索则能补 coverage；只能看历史，则没做过的动作没有 identifiability。

这些对打没有统一最优解。和第 14 章一样：先问自己在哪一条轴上，再决定该信哪一个算法。

---

## 15. 自测

1. 为什么 RL 不能看成固定数据集上的监督学习？Policy 改变了什么？  
2. 稀疏 reward、temporal credit assignment、环境随机、探索–利用，各对应企鹅 / 下棋的哪件事？  
3. Markov 假设是“没有历史”，还是“历史已经进了 state”？  
4. POMDP 与 MDP 差在哪？单帧 Atari 为什么不够？  
5. Reward 与 return 为什么不能混？\(\gamma=1\) 在无限视野里有什么问题？  
6. \(v_\pi(s)\) 与 \(q_\pi(s,a)\) 各平均掉了什么？  
7. Bellman equation、TD target、TD error 三者分别是什么？Bootstrap 带来什么好处和代价？  
8. 为什么不探索就没有关于未选 actions 的数据？  
9. DP、MC、TD 的信息来源有什么不同？  
10. 同一条 transition 上，SARSA target 1.9、Q-learning 4.6，差在假装未来会怎么走？  
11. Terminal transition 为什么必须乘 \((1-\mathrm{done})\)？`terminated` 和 `truncated` 为什么不能混？  
12. Replay buffer 和 target network 各治哪种病？Deadly triad 是哪三件的叠加？  
13. Double DQN 拆开的是哪两件事？它和 target network 是同一帖药吗？  
14. Policy gradient 为什么不要求 environment 可微？  
15. Baseline 为什么能降方差、却不改变期望梯度？“较小但仍为正的 \(G_t\)”会不会自动削弱该动作？  
16. Actor 与 critic 分别学什么？Advantage 的符号怎样改 \(\log\pi\)？  
17. Offline 数据里从未选过 `risky` 时，为什么无法判断它是 \(+10\) 还是 \(-10\)？  
18. Decision Transformer 的 desired return 超出 dataset support 后，为什么不保证可实现？

---

## 16. 合上书再看一眼

RL 是一个 agent–environment 闭环。Agent 选动作，动作改写下一格、下一份奖励，以及下一批评训数据。一步 reward 不够，目标是长期贴现回报；value 是这份回报的期望；Bellman 只是把期望按“现在 + 打折的下一步”拆开。

信息够，就在脑子里做 DP；只能跑完整故事，就用 Monte Carlo；不想等终点，就用一步 TD 去 bootstrap。要控制时，SARSA 评价实际会走的路，Q-learning 评价 greedy 会走的路。表存不下，把 \(Q\) 交给网络，于是出现 replay、target network、Double、deadly triad——全是在对抗“标签会自己跑、数据由政策产生”。不想先学 \(Q\)，就直接推 \(\log\pi\)；方差太高，就请一个 critic 来报 advantage。

若连试都试不了，coverage 变成定理边界：没做过的动作，价值不可识别。RL 最不像前面章节的地方，始终是同一句——模型不只学习数据，它还通过动作决定下一批数据从哪里来。

第 2–19 章把系统造齐了。下一章离开「怎样造」，改问这件事按理为什么不该成功，却成功了。
