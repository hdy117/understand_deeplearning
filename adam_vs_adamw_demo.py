"""
Adam / Adam+L2 / AdamW 对照示例

两个参数起点相同，但数据梯度量级差 100 倍：
  φ0: 大梯度（类似深层里尺度大的参数）
  φ1: 小梯度（尺度小的参数）

运行:
  python adam_vs_adamw_demo.py
"""

from __future__ import annotations

import numpy as np


def adam_like(
    mode: str,
    *,
    steps: int = 8,
    alpha: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    lam: float = 0.2,
    g_data: np.ndarray | None = None,
    phi0: np.ndarray | None = None,
) -> list[dict]:
    """
    mode:
      - "adam":     无正则
      - "adam_l2":  L2 并进梯度（耦合）
      - "adamw":    解耦 weight decay
    """
    if g_data is None:
        g_data = np.array([10.0, 0.1])
    if phi0 is None:
        phi0 = np.array([1.0, 1.0])

    phi = phi0.astype(float).copy()
    m = np.zeros_like(phi)
    v = np.zeros_like(phi)
    history: list[dict] = []

    for t in range(1, steps + 1):
        phi_before = phi.copy()

        if mode == "adam_l2":
            g = g_data + lam * phi_before
        else:
            g = g_data.copy()

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        adam_step = alpha * m_hat / (np.sqrt(v_hat) + eps)

        if mode == "adamw":
            decay = alpha * lam * phi_before
            phi = phi_before - adam_step - decay
        else:
            decay = np.zeros_like(phi)
            phi = phi_before - adam_step

        history.append(
            {
                "t": t,
                "phi_before": phi_before,
                "g": g,
                "sqrt_vhat": np.sqrt(v_hat),
                "adam_step": adam_step,
                "decay": decay,
                "phi_after": phi.copy(),
            }
        )
    return history


def print_history(title: str, history: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(
        f"{'t':>2} | {'phi_before':^18} | {'g':^18} | "
        f"{'adam_step':^18} | {'decay':^14} | {'phi_after':^18}"
    )
    for h in history:
        print(
            f"{h['t']:2d} | "
            f"{np.array2string(h['phi_before'], precision=4):18s} | "
            f"{np.array2string(h['g'], precision=4):18s} | "
            f"{np.array2string(h['adam_step'], precision=4):18s} | "
            f"{np.array2string(h['decay'], precision=4):14s} | "
            f"{np.array2string(h['phi_after'], precision=4):18s}"
        )


def demo_pure_decay() -> None:
    """
    数据梯度恒为 0：只剩正则 / weight decay。
    最能看出「耦合 L2」和「解耦 decay」的差别。
    """
    print("\n" + "#" * 72)
    print("场景 B：数据梯度 g_data = 0（只看正则怎么拉参数）")
    print("#" * 72)

    alpha, lam = 0.1, 0.1
    # 两个参数起点相同
    for mode, name in [("adam_l2", "Adam + L2（耦合）"), ("adamw", "AdamW（解耦）")]:
        hist = adam_like(
            mode,
            steps=8,
            alpha=alpha,
            lam=lam,
            g_data=np.array([0.0, 0.0]),
            phi0=np.array([1.0, 1.0]),
        )
        print_history(name, hist)

    print(
        """
解读（g_data=0）:
  Adam+L2: g = λφ，归一化后 adam_step ≈ α · sign(φ)
           → 每步大约固定减 α，不是「按比例缩小」
  AdamW:   g = 0 → adam_step ≈ 0，只剩 φ ← (1 - αλ)φ
           → 真正的指数式权重衰减
"""
    )


def demo_different_grad_scales() -> None:
    print("\n" + "#" * 72)
    print("场景 A：两参数起点相同，数据梯度差 100 倍")
    print("  g_data = [10.0, 0.1],  φ0 = [1, 1]")
    print("  α=0.05, λ=0.2, β1=0.9, β2=0.999")
    print("#" * 72)

    for mode, name in [
        ("adam", "Adam（无正则）"),
        ("adam_l2", "Adam + L2（耦合，常见错误用法）"),
        ("adamw", "AdamW（解耦 weight decay）"),
    ]:
        hist = adam_like(mode, steps=6)
        print_history(name, hist)

    print(
        """
解读（有数据梯度）:
  Adam:  两维 adam_step 几乎一样（≈α），因为 m/√v 近似只保留符号
  Adam+L2:
    大梯度维: g≈10+λφ，λφ 几乎被淹没，正则几乎改不动步长
    小梯度维: g≈0.1+λφ，λφ 占比很大，正则严重扭曲 g
    但两者最终 step 仍被 √v 压成差不多大小 → 有效衰减被扭曲
  AdamW:
    数据部分仍走 Adam（两维 step 仍≈α）
    衰减部分额外减 αλ·φ，两维同比例（起点相同时 decay 相同）
"""
    )


def demo_handcalc_step1() -> None:
    """打印 t=1 的手算过程，方便对照笔记。"""
    print("\n" + "#" * 72)
    print("场景 C：t=1 手算核对")
    print("#" * 72)
    alpha, beta1, beta2, lam = 0.05, 0.9, 0.999, 0.2
    phi = np.array([1.0, 1.0])
    g_data = np.array([10.0, 0.1])

    print("公共设定: φ=[1,1], g_data=[10, 0.1], α=0.05, β1=0.9, β2=0.999, λ=0.2")
    print()

    # Adam
    g = g_data
    m = (1 - beta1) * g
    v = (1 - beta2) * (g * g)
    mhat = m / (1 - beta1)
    vhat = v / (1 - beta2)
    step = alpha * mhat / np.sqrt(vhat)
    print("Adam:")
    print(f"  g={g}")
    print(f"  m=(1-β1)g={m}")
    print(f"  m̂=m/(1-β1)={mhat}  （bias correction 后恢复为 g）")
    print(f"  step=α·m̂/√v̂={step}")
    print(f"  φ←φ-step={phi - step}")
    print()

    # Adam+L2
    g = g_data + lam * phi
    m = (1 - beta1) * g
    v = (1 - beta2) * (g * g)
    mhat = m / (1 - beta1)
    vhat = v / (1 - beta2)
    step = alpha * mhat / np.sqrt(vhat)
    print("Adam+L2:")
    print(f"  g=∇L+λφ={g}   ← 注意小梯度维从 0.1 变成 0.3")
    print(f"  step={step}")
    print(f"  φ←φ-step={phi - step}")
    print("  （正则已混进 g，再被 √v 归一，看不出独立的「缩权重」）")
    print()

    # AdamW
    g = g_data
    m = (1 - beta1) * g
    v = (1 - beta2) * (g * g)
    mhat = m / (1 - beta1)
    vhat = v / (1 - beta2)
    step = alpha * mhat / np.sqrt(vhat)
    decay = alpha * lam * phi
    print("AdamW:")
    print(f"  g=∇L={g}  （不含 λφ）")
    print(f"  adam_step={step}")
    print(f"  decay=αλ·φ={decay}")
    print(f"  φ←φ-step-decay={phi - step - decay}")
    print("  （数据自适应 + 独立按比例衰减）")


if __name__ == "__main__":
    demo_handcalc_step1()
    demo_different_grad_scales()
    demo_pure_decay()
