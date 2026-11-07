# AGENTS.md — understand_deeplearning 工作区记忆

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
