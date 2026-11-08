# AGENTS.md — understand_deeplearning 工作区记忆

## 2026-11-08 会话：sequence v5 教学重构

### 完成的工作

| 源文件 | 产出 | 说明 |
|---|---|---|
| sequence_models_evolution_v4.md（重构） | sequence_models_evolution_v5.md → sequence_v5_am_draft.md → sequence_models_evolution_v5.html | **教学重构**：每章重排成参照 `RNN的记忆、序列、时间与状态.md` 的「六拍」模板；v4 技术内容一字未删 |

### v5 的六拍模板（每章固定节拍）

1. 一句话 + 一个式子（boxed 核心循环，先说干什么）
2. 同一道题走一遍（§0.1 的 A/B 题逐步手算，第一步/第二步/第三步）
3. 对在哪，错在哪（每个错读数指认一条机制）
4. 机制细看（推导、账单、边界都在这里，不再开章就跳抽象框架）
5. 一个类比（RNN 传话游戏 / LSTM 一排账户 / Transformer 档案馆 / Linear Attn 只会加法的会计 / DeltaNet 拿到橡皮的会计 / Gated 碎纸机 / Hybrid 图书馆+笔记本）
6. 本章小结（编号列表 + 一句话收尾，明确「修好了上一章什么病、留下什么病给下一章」）

- 章首新增「本章接口」引用块，焊上章间接缝（治"太跳跃"）。
- 新增「怎么读这本书」节，把六拍模板明说给读者。
- §3.2/§7.2 给 Transformer/Hybrid 也补了同一道题的走读（v4 里这两章没有逐步走账）。
- 交叉引用全部重编号，已用脚本体检无悬空（§4.5 串味 / §4.6 上界 / §5.7 并行账单 / §7.5 怎么选）。

### 管线新经验

- 转换脚本现存 `build_am_draft_v5.py`（v4 及更早脚本已不在目录里）。
- mermaid → flow 的正则两个坑：虚线箭头是 `-.->`（正则要写 `-\.->`）；节点文本可含 `_{...}` LaTeX（方括号节点文本字符类用 `[^"\]]`，菱形用 `[^"\}]`，分开匹配）。
- am render 在有 STE 警告时不会自动打开页面，需 `--replace <同一 html>` 覆盖（会删除旧页重写）。
- 验证 HTML：Edge 无头截图 `"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless --screenshot=... --window-size=1400,3200 file:///...`。

### 待办/可能的后续

- v4 的 HTML 仍保留未动；若 v5 取代 v4，考虑是否归档 v4 三件套。
- 三份 HTML 之间尚未互加跳转链接（已提议，等 David 确认）。
- deltanet 教程的 md 源里有指向 tensor 指南的相对链接，HTML 版目前是纯文本路径。

---

（以下为 2026-11-07 会话记录，管线细节已被上面的 v5 条目更新）

## 所有者偏好（David）

- 称呼 David，教育/科研工作者，助手名 Ember。
- 语气：轻松、高能量；专业回复可带公式，术语用英文，要讲清思想、渊源、架构、流程。
- 视觉学习者：多用流程图/示意图；**输出图像内不要中文**。
- 重要概念用「互补式重复」（多角度搭桥），不用机械重复；鼓励冗余讲解。

## 2026-11-07 会话：tensor_and_memeory 教程的 md → HTML 管线

### 完成的工作

目录：`tensor_and_memeory/`（注意目录名是 memeory，拼写保持原样）。

| 源文件 | 产出 HTML | 说明 |
|---|---|---|
| tensor_and_matrix_multiplication_pytorch_guide.md | tensor_and_matrix_multiplication_pytorch_guide.html | 4345 行主线教程，全文转换 |
| deltanet_attention_family_guide.md | deltanet_attention_family_guide.html | DeltaNet 专题，全文转换 |
| sequence_models_evolution_v3.md + matrix_multiplication_view.html | sequence_models_evolution_v4.md → sequence_models_evolution_v4.html | **改写融合**：把「矩阵乘法六副眼镜」（01 点积 / 02 列 / 03 行 / 04 外积 / 05 复合 / 06 括号）织入 v3 主线 |

### v4 改写要点（sequence_models_evolution_v4.md）

- 新增 §0.4「六副眼镜」词汇表；原 §0.4/§0.5 顺延为 §0.5/§0.6。
- 各章按编号引用眼镜：§0.3=01+06+04，§1=02 列，§3=01 点积，§4.2=06 括号正面战场，§4.4 补 N=8192 差 64 倍账单，§5.1=04 外积 rank-1 地毯。
- §8 机制总结表新增「主视角」列。
- v3 原文未删一字，只做增补。

### 渲染管线（复用方法）

1. 转换脚本（同目录已有三个范例：`build_am_draft.py`、`build_am_draft_deltanet.py`、`build_am_draft_v4.py`）：
   - 加 frontmatter：`template: doc`、`title`、`subtitle`、`file: <英文短名>`、`lang: zh`；
   - 删除首个 `# ` 标题行（frontmatter 承载标题）；删除正文独立 `---` 分隔线（需跟踪代码围栏状态）；
   - mermaid 块转成 am 的 `flow` 组件（节点用 `[文本]`，边标签跟在 `: ` 后；注意 ``` 和 ~~~ 两种围栏都要处理；同名节点会合并，需改名区分）；
   - 相对 .md 链接改成「纯文本 + `代码路径`」。
2. 渲染命令：
   `node "D:/KimiData/daimon-share/daimon/runtime/kimi-code/home/plugins/managed/answer-me-with-html@answer-me-with-html/answer-me-with-html/scripts/am.mjs" render <draft.md> -o <目标.html>`
3. STE/code-length 警告属行文风格建议，教程原文不改，直接保留页面。
4. 改写用 `rewrite_v4.py` 模式：精确锚点替换 + 断言唯一，比全文重写省 token 且可复跑。

### 目录里的关键文件

- `matrix_multiplication_view.html`：交互式「六副眼镜」教学页（非 am 生成，无 am-source，内容靠文本提取）。
- `*_am_draft.md`：各 HTML 的中间草稿，改内容后用 `am render <draft> -o <同一 html>` 覆盖更新。
- 其他源文件：RNN的记忆、序列、时间与状态.md、toy_rnn_guide.md、toy_rnn.py、tensor_demo.ipynb、sequence_models_evolution.md / _v2.md（v3 的旧版）。

### 待办/可能的后续

- 三份 HTML 之间尚未互加跳转链接（已提议，等 David 确认）。
- deltanet 教程的 md 源里有指向 tensor 指南的相对链接，HTML 版目前是纯文本路径。
