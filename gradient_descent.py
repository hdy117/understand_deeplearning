import argparse

import matplotlib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def plt_data(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    show: bool = False,
    output_path: str = "gradient_descent.png",
) -> None:
    matplotlib.use("TkAgg" if show else "Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.scatter(x.numpy(), y.numpy())
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("Scatter Plot of Data")

    if show:
        plt.show()
    else:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"Plot saved to {output_path}")


class DataGenerator:
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.start = -10.0
        self.end = 10.0
        self.x = torch.linspace(self.start, self.end, steps=self.num_samples)
        self.y = (
            1.0
            + 2 * self.x
            + 3 * self.x * self.x
            + 4 * self.x * self.x * self.x
            + torch.randn(num_samples)
        )  # Adding noise for demonstration

    def __len__(self):
        return self.num_samples

    def get_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y


class ManualGradientDescent(torch.nn.Module):
    def __init__(self, learning_rate=0.01):
        super().__init__()

        self.learning_rate = learning_rate

        # prameters to fit the model
        self.parameter_a = 0.0
        self.parameter_b = 0.0
        self.parameter_c = 0.0
        self.parameter_d = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.parameter_a
            + self.parameter_b * x
            + self.parameter_c * x * x
            + self.parameter_d * x * x * x
        )

    def f_to_a(self, x):
        return 1.0

    def f_to_b(self, x):
        return x

    def f_to_c(self, x):
        return x * x

    def f_to_d(self, x):
        return x * x * x

    def gradient_descent(
        self, x_input: torch.Tensor, y_label: torch.Tensor, epoches=1000
    ):
        # tensorboard writer
        writer = SummaryWriter("run/exp_4")

        # to cuda
        x_input = x_input.to(torch.device("cuda"))
        y_label = y_label.to(torch.device("cuda"))

        # train
        for i in range(epoches):
            y_pred = self.forward(x_input)

            # MAE: L = mean(|f - y|)
            # ∂L/∂f = sign(f - y)  （f=y 时 subgradient 取 0）
            # 注意：不是 2(f-y)（那是 MSE），也不是 |f-y|
            # loss = torch.mean(torch.abs(y_pred - y_label))
            # loss_2_f = torch.sign(y_pred - y_label)

            # 若要做 MSE，应改为：
            loss = torch.mean((y_pred - y_label) ** 2)
            loss_2_f = 2 * (y_pred - y_label)

            f_2_a = self.f_to_a(x_input)
            f_2_b = self.f_to_b(x_input)
            f_2_c = self.f_to_c(x_input)
            f_2_d = self.f_to_d(x_input)

            # 链式法则: ∂L/∂θ = mean( (∂L/∂f) * (∂f/∂θ) )
            l_2_a = torch.mean(loss_2_f * f_2_a)
            l_2_b = torch.mean(loss_2_f * f_2_b)
            l_2_c = torch.mean(loss_2_f * f_2_c)
            l_2_d = torch.mean(loss_2_f * f_2_d)

            self.parameter_a = self.parameter_a - 1e1 * self.learning_rate * l_2_a
            self.parameter_b = self.parameter_b - 1e3 * self.learning_rate * l_2_b
            self.parameter_c = self.parameter_c - self.learning_rate * l_2_c
            self.parameter_d = self.parameter_d - self.learning_rate * l_2_d

            # add to tensorboard
            writer.add_scalar("loss", torch.mean(loss), i + 1)
            writer.add_scalar("loss_2_f", torch.mean(loss_2_f), i + 1)
            # writer.add_scalar("l_2_a", l_2_a, i + 1)
            # writer.add_scalar("l_2_b", l_2_b, i + 1)
            # writer.add_scalar("l_2_c", l_2_c, i + 1)
            # writer.add_scalar("l_2_d", l_2_d, i + 1)
            writer.add_scalar("f_2_a", f_2_a, i + 1)
            writer.add_scalar("f_2_b", torch.mean(f_2_b), i + 1)
            writer.add_scalar("f_2_c", torch.mean(f_2_c), i + 1)
            writer.add_scalar("f_2_d", torch.mean(f_2_d), i + 1)
            writer.add_scalar("f_2_a_abs", f_2_a, i + 1)
            writer.add_scalar("f_2_b_abs", torch.mean(torch.abs(f_2_b)), i + 1)
            writer.add_scalar("f_2_c_abs", torch.mean(torch.abs(f_2_c)), i + 1)
            writer.add_scalar("f_2_d_abs", torch.mean(torch.abs(f_2_d)), i + 1)
            writer.add_scalar("parameter_a", self.parameter_a, i + 1)
            writer.add_scalar("parameter_b", self.parameter_b, i + 1)
            writer.add_scalar("parameter_c", self.parameter_c, i + 1)
            writer.add_scalar("parameter_d", self.parameter_d, i + 1)

            # if (i + 1) % 100 == 0:
            #     print(f"==================")
            #     print(f"loss:{torch.mean(loss)}")
            #     print(f"l_2_f:{torch.mean(loss_2_f)}")
            #     print(f"********")
            #     print(f"l_2_a:{l_2_a}, f_2_a:{f_2_a}")
            #     print(f"l_2_b:{l_2_b}, f_2_b:{torch.mean(f_2_b)}")
            #     print(f"l_2_c:{l_2_c}, f_2_c:{torch.mean(f_2_c)}")
            #     print(f"l_2_d:{l_2_d}, f_2_d:{torch.mean(f_2_d)}")
            #     print(f"self.parameter_a: {self.parameter_a}")
            #     print(f"self.parameter_b: {self.parameter_b}")
            #     print(f"self.parameter_c: {self.parameter_c}")
            #     print(f"self.parameter_d: {self.parameter_d}")

        writer.close()
        y_pred = self.forward(x_input)

        x = x_input.cpu()
        y_pred = y_pred.cpu()

        plt_data(
            x,
            y_pred,
            show=False,
            output_path="gradient_descent.png",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot with Tk instead of saving it",
    )
    args = parser.parse_args()

    data_generator = DataGenerator(num_samples=1000)
    x, y = data_generator.get_data()
    # print(f"x:[x]\ny:[{y}]")
    plt_data(x, y, show=args.show, output_path="original_data.png")

    manual_gd = ManualGradientDescent(learning_rate=1e-3)
    manual_gd = manual_gd.to(torch.device("cuda"))
    manual_gd.gradient_descent(x, y, epoches=3000)
