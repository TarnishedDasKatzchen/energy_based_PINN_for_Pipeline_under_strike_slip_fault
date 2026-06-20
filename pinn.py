"""
Mimikyu

With the help of Claude, GPT and Deepseek
"""

import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PINN(torch.nn.Module):
    """
    原始 PINN 网络结构 backbone
    输出 2 个标量：[N_u, N_w]，由外层硬约束包装组合成最终 [u, w]。
    """

    def __init__(self, layers=[64, 64, 64, 64], max_a=1000, device=None):
        super(PINN, self).__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.max_a = max_a

        self.input_layer = torch.nn.Linear(1, layers[0])

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden_layers.append(torch.nn.Linear(layers[i], layers[i + 1]))

        self.output_layer = torch.nn.Linear(layers[-1], 2)

        self.activations = torch.nn.ModuleList()
        for i in range(len(layers)):
            self.activations.append(torch.nn.Tanh())

        self.alphas = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(1) * 1.0) for _ in range(len(layers))
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.xavier_normal_(self.input_layer.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(self.input_layer.bias)

        for layer in self.hidden_layers:
            torch.nn.init.xavier_normal_(layer.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(layer.bias)

        torch.nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        torch.nn.init.zeros_(self.output_layer.bias)

    def forward(self, x_in):
        if not torch.is_tensor(x_in):
            x_in = torch.from_numpy(x_in).float().to(device)

        x_normalized = x_in / self.max_a

        x = self.input_layer(x_normalized)
        x = self.alphas[0] * self.activations[0](x)

        for i, (hidden_layer, activation, alpha) in enumerate(
            zip(self.hidden_layers, self.activations[1:], self.alphas[1:])
        ):
            residual = x
            x = hidden_layer(x)
            x = alpha * activation(x)
            if residual.shape == x.shape and i % 2 == 1:
                x = x + residual

        x_out = self.output_layer(x)
        return x_out


# =============================================================
#                  硬约束包装
# =============================================================
class HardConstraintPINN(torch.nn.Module):
    """
    在原始 PINN 外包一层function：
        u_tilde(x) = A_u(x; lf) + B(x) * N_u(x)
        w_tilde(x) = A_w(x; lf) + B(x) * N_w(x)

    ----
    backbone = PINN(layers=[64,64,64,64], max_a=2000)
    model = HardConstraintPINN(backbone, L=2000.0,
                               disp_grud_x=disp_grud_x,
                               disp_grud_y=disp_grud_y).to(device)

    """

    def __init__(self, backbone, L,
                 disp_grud_x, disp_grud_y,
                 device=None):
        super(HardConstraintPINN, self).__init__()
        self.backbone = backbone
        self.L = float(L)
        # 把远场位移做成 buffer，方便切换设备
        self.register_buffer('disp_x', torch.tensor(float(disp_grud_x)))
        self.register_buffer('disp_y', torch.tensor(float(disp_grud_y)))
        # load_factor 用 buffer 而非 Parameter，外部直接 set
        self.register_buffer('lf', torch.tensor(0.05))

    def set_load_factor(self, lf):
        """与 trainer.update_curriculum 同步使用。"""
        self.lf = torch.tensor(float(lf), device=self.lf.device, dtype=self.lf.dtype)

    @staticmethod
    def _s(xi):
        """Hermite 形函数：s(-1)=0, s(+1)=1, s'(±1)=0。
        s(xi) = 1/2 + 3/4 * xi - 1/4 * xi**3
        """
        return 0.5 + 0.75 * xi - 0.25 * xi ** 3

    @staticmethod
    def _B(xi):
        """函数：B(±1)=0, B'(±1)=0。
        B(xi) = (1 - xi**2)**2
        在中心 xi=0 处取最大值 1，逐渐衰减到边界为 0。
        """
        return (1.0 - xi ** 2) ** 2

    def forward(self, x_in):
        if not torch.is_tensor(x_in):
            x_in = torch.from_numpy(x_in).float().to(self.lf.device)

        # 网络原始输出 [N_u, N_w]
        N = self.backbone(x_in)
        N_u = N[:, 0:1]
        N_w = N[:, 1:2]

        # 归一化坐标 xi = x / L ∈ [-1, 1]
        xi = x_in / self.L

        s_val = self._s(xi)            # [N,1]
        B_val = self._B(xi)            # [N,1]

        lf = self.lf
        A_u = self.disp_x * lf * s_val
        A_w = self.disp_y * lf * s_val

        u = A_u + B_val * N_u
        w = A_w + B_val * N_w
        # u = A_u * N_u
        # w = A_w * N_w
        return torch.cat([u, w], dim=1)
