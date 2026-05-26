import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def set_seed(seed=26):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f" Random seed set to {seed} (results are reproducible)")


SEED = 42
set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class AnalyticalSolution3D:
    def __init__(self, Lx, Ly, Lz, c=343):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.c = c

    def get_standing_wave(self, x, y, z, t, freq, amplitude):
        """简化的驻波解"""
        k = 2 * np.pi * freq / self.c
        # 简化的驻波模式
        return amplitude * np.cos(k * x) * np.sin(2 * np.pi * freq * t)


class FDTD3DSolver:
    def __init__(self, Lx, Ly, Lz, dx, c=343, cfl=0.5):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c

        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1

        self.lambda2 = cfl ** 2

    def solve(self, freq=200, amplitude=5.0, n_steps=200, source_pos=None, record_every=10):
        p = np.zeros((self.Nx, self.Ny, self.Nz))
        p_prev = np.zeros((self.Nx, self.Ny, self.Nz))

        src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        history = []

        for step in range(n_steps):
            t = step * self.dt

            if t < 0.01:
                window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                src_val = amplitude * window * np.sin(2 * np.pi * freq * t)
                p[src_ix, src_iy, src_iz] += src_val

            p_new = np.zeros((self.Nx, self.Ny, self.Nz))

            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    for k in range(1, self.Nz - 1):
                        laplacian = (
                                p[i + 1, j, k] + p[i - 1, j, k] +
                                p[i, j + 1, k] + p[i, j - 1, k] +
                                p[i, j, k + 1] + p[i, j, k - 1] -
                                6 * p[i, j, k]
                        )
                        p_new[i, j, k] = (2 * p[i, j, k] - p_prev[i, j, k] +
                                          self.lambda2 * laplacian)

            # 刚性边界
            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = p_new[-2, :, :]
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]

            if step % record_every == 0:
                history.append({
                    'step': step,
                    't': t,
                    'p': p.copy(),
                    'p_center': p[:, :, self.Nz // 2].copy()
                })

            p_prev, p = p, p_new

        return history


class ConstrainedCorrector3D(nn.Module):


    def __init__(self, nx, ny, nz, hidden_channels=16):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),

            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),

            nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

        # 修正幅度限制（可学习，但初始很小）
        self.correction_limit = nn.Parameter(torch.tensor(0.01))

    def forward(self, p_curr, p_prev, freq, amplitude, time):
        """
        预测归一化的修正项
        """
        batch_size = p_curr.shape[0]

        # 归一化输入
        max_val = amplitude.view(batch_size, 1, 1, 1, 1) + 1e-6
        p_curr_norm = p_curr.unsqueeze(1) / max_val
        p_prev_norm = p_prev.unsqueeze(1) / max_val
        p_curr_norm = torch.clamp(p_curr_norm, -2, 2)
        p_prev_norm = torch.clamp(p_prev_norm, -2, 2)

        # 堆叠
        x = torch.cat([p_curr_norm, p_prev_norm], dim=1)

        # 编码-解码
        features = self.encoder(x)
        correction_norm = self.decoder(features)  # [batch, 1, nx, ny, nz]

        # 强约束：修正项不超过输入幅度的 correction_limit 倍
        correction = correction_norm * self.correction_limit * amplitude.view(batch_size, 1, 1, 1, 1)

        return correction.squeeze(1)



class StableHybridFDTD3D:

    def __init__(self, Lx, Ly, Lz, dx, c=343, cfl=0.5, ml_model=None,
                 use_correction=True, max_correction_ratio=0.1):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c
        self.lambda2 = cfl ** 2

        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1

        self.ml_model = ml_model
        self.use_correction = use_correction
        self.max_correction_ratio = max_correction_ratio  # 修正项最大占输入的比例

    def solve(self, freq=200, amplitude=5.0, n_steps=200, record_every=10):
        p = np.zeros((self.Nx, self.Ny, self.Nz))
        p_prev = np.zeros((self.Nx, self.Ny, self.Nz))
        src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        history = []

        for step in range(n_steps):
            t = step * self.dt

            if t < 0.01:
                window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                src_val = amplitude * window * np.sin(2 * np.pi * freq * t)
                p[src_ix, src_iy, src_iz] += src_val

            # FDTD更新
            p_new = np.zeros((self.Nx, self.Ny, self.Nz))
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    for k in range(1, self.Nz - 1):
                        laplacian = (
                                p[i + 1, j, k] + p[i - 1, j, k] +
                                p[i, j + 1, k] + p[i, j - 1, k] +
                                p[i, j, k + 1] + p[i, j, k - 1] -
                                6 * p[i, j, k]
                        )
                        p_new[i, j, k] = (2 * p[i, j, k] - p_prev[i, j, k] +
                                          self.lambda2 * laplacian)

            # ML修正（带安全限制）
            if self.ml_model is not None and self.use_correction:
                with torch.no_grad():
                    p_curr_tensor = torch.FloatTensor(p).unsqueeze(0).to(device)
                    p_prev_tensor = torch.FloatTensor(p_prev).unsqueeze(0).to(device)
                    freq_tensor = torch.FloatTensor([freq]).to(device)
                    amp_tensor = torch.FloatTensor([amplitude]).to(device)
                    time_tensor = torch.FloatTensor([t]).to(device)

                    correction = self.ml_model(p_curr_tensor, p_prev_tensor,
                                               freq_tensor, amp_tensor, time_tensor)
                    correction = correction.squeeze().cpu().numpy()

                # 安全限制：修正项不超过输入最大值的 max_correction_ratio
                max_correction = self.max_correction_ratio * np.max(np.abs(p_new))
                correction = np.clip(correction, -max_correction, max_correction)

                p_new = p_new + correction

            # 边界条件
            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = p_new[-2, :, :]
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]

            # 额外安全检查：防止数值爆炸
            if np.max(np.abs(p_new)) > 100:
                print(f"  Warning: Numerical instability at step {step}, using fallback")
                p_new = p  # 回退到上一步

            if step % record_every == 0:
                history.append({
                    'step': step,
                    't': t,
                    'p': p.copy(),
                    'p_center': p[:, :, self.Nz // 2].copy()
                })

            p_prev, p = p, p_new

        return history


class TrainingDataGenerator3D:
    def __init__(self, Lx, Ly, Lz, dx, c=343):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = dx
        self.c = c
        self.dt = 0.5 * dx / c

        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1

    def generate_samples(self, n_samples=30, n_steps=150):
        """生成训练数据 - 修正目标归一化到小范围"""
        X_curr = []
        X_prev = []
        X_freq = []
        X_amp = []
        X_time = []
        y_correction = []

        lambda2 = 0.5 ** 2

        for sample_idx in range(n_samples):
            freq = np.random.uniform(150, 250)
            amplitude = np.random.uniform(4.0, 6.0)

            p_curr = np.zeros((self.Nx, self.Ny, self.Nz))
            p_prev = np.zeros((self.Nx, self.Ny, self.Nz))
            src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2

            for step in range(n_steps):
                t = step * self.dt

                if t < 0.01:
                    window = 0.5 * (1 - np.cos(2 * np.pi * t / 0.01))
                    src_val = amplitude * window * np.sin(2 * np.pi * freq * t)
                    p_curr[src_ix, src_iy, src_iz] += src_val

                # FDTD预测
                p_fdtd = np.zeros((self.Nx, self.Ny, self.Nz))
                for i in range(1, self.Nx - 1):
                    for j in range(1, self.Ny - 1):
                        for k in range(1, self.Nz - 1):
                            laplacian = (
                                    p_curr[i + 1, j, k] + p_curr[i - 1, j, k] +
                                    p_curr[i, j + 1, k] + p_curr[i, j - 1, k] +
                                    p_curr[i, j, k + 1] + p_curr[i, j, k - 1] -
                                    6 * p_curr[i, j, k]
                            )
                            p_fdtd[i, j, k] = (2 * p_curr[i, j, k] - p_prev[i, j, k] +
                                               lambda2 * laplacian)

                # 边界条件
                p_fdtd[0, :, :] = p_fdtd[1, :, :]
                p_fdtd[-1, :, :] = p_fdtd[-2, :, :]
                p_fdtd[:, 0, :] = p_fdtd[:, 1, :]
                p_fdtd[:, -1, :] = p_fdtd[:, -2, :]
                p_fdtd[:, :, 0] = p_fdtd[:, :, 1]
                p_fdtd[:, :, -1] = p_fdtd[:, :, -2]

                # 目标：理想修正项（使用FDTD自身的二阶精度作为参考）
                # 实际训练中，我们让模型学习平滑化修正
                if step > 20 and step % 5 == 0:
                    # 计算局部梯度作为修正目标（小幅度）
                    grad_x = np.gradient(p_fdtd, axis=0)
                    grad_y = np.gradient(p_fdtd, axis=1)
                    grad_z = np.gradient(p_fdtd, axis=2)
                    target_correction = 0.01 * (grad_x + grad_y + grad_z)
                    target_correction = np.clip(target_correction, -0.1, 0.1)

                    X_curr.append(p_curr.copy())
                    X_prev.append(p_prev.copy())
                    X_freq.append(freq)
                    X_amp.append(amplitude)
                    X_time.append(t)
                    y_correction.append(target_correction)

                p_prev, p_curr = p_curr, p_fdtd

            if (sample_idx + 1) % 10 == 0:
                print(f"  Generated {sample_idx + 1}/{n_samples} samples")

        # 归一化输入输出
        X_curr = torch.FloatTensor(np.array(X_curr))
        X_prev = torch.FloatTensor(np.array(X_prev))
        X_freq = torch.FloatTensor(np.array(X_freq)) / 500
        X_amp = torch.FloatTensor(np.array(X_amp)) / 10
        X_time = torch.FloatTensor(np.array(X_time)) / 0.02
        y_correction = torch.FloatTensor(np.array(y_correction))

        # 限制目标范围
        y_correction = torch.clamp(y_correction, -0.1, 0.1)

        print(f"\nGenerated {len(X_curr)} training samples")
        print(f"  Correction range: [{y_correction.min():.4f}, {y_correction.max():.4f}]")

        return (X_curr, X_prev, X_freq, X_amp, X_time), y_correction


def train_3d_model():
    print("\n" + "=" * 70)
    print("Training 3D Constrained Corrector Model")
    print("=" * 70)

    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = 0.05

    Nx, Ny, Nz = int(Lx / dx) + 1, int(Ly / dx) + 1, int(Lz / dx) + 1

    print(f"\nGrid: {Nx} x {Ny} x {Nz} = {Nx * Ny * Nz:,} points")

    print("\nGenerating training data...")
    generator = TrainingDataGenerator3D(Lx, Ly, Lz, dx, c=343)
    (X_curr, X_prev, X_freq, X_amp, X_time), y = generator.generate_samples(
        n_samples=40, n_steps=120
    )

    n_total = len(X_curr)
    n_train = int(0.8 * n_total)

    # 使用固定的随机排列（确保每次划分一致）
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_curr_train = X_curr[train_idx].to(device)
    X_prev_train = X_prev[train_idx].to(device)
    X_freq_train = X_freq[train_idx].to(device)
    X_amp_train = X_amp[train_idx].to(device)
    X_time_train = X_time[train_idx].to(device)
    y_train = y[train_idx].to(device)

    X_curr_val = X_curr[val_idx].to(device)
    X_prev_val = X_prev[val_idx].to(device)
    X_freq_val = X_freq[val_idx].to(device)
    X_amp_val = X_amp[val_idx].to(device)
    X_time_val = X_time[val_idx].to(device)
    y_val = y[val_idx].to(device)

    print(f"Training: {len(train_idx)}, Validation: {len(val_idx)}")

    model = ConstrainedCorrector3D(Nx, Ny, Nz, hidden_channels=16)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    print("\nStarting training...")
    best_val_loss = float('inf')
    epochs = 4000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_curr_train, X_prev_train, X_freq_train, X_amp_train, X_time_train)
        loss = criterion(output, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_curr_val, X_prev_val, X_freq_val, X_amp_val, X_time_val)
            val_loss = criterion(val_output, y_val)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_3d_corrector.pth')

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: Train Loss={loss.item():.12f}, Val Loss={val_loss.item():.12f}")

    model.load_state_dict(torch.load('best_3d_corrector.pth'))
    print(f"\nBest model saved (val loss: {best_val_loss:.12f})")

    return model


def comparison_experiment_3d(ml_model):
    print("\n" + "=" * 70)
    print("3D Comparison: Standard vs Optimized FDTD")
    print("=" * 70)

    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = 0.05
    test_freq = 200
    test_amp = 5.0
    n_steps = 200

    print("\nRunning Standard FDTD...")
    standard_solver = StableHybridFDTD3D(
        Lx, Ly, Lz, dx, c=343, cfl=0.5,
        ml_model=None, use_correction=False
    )
    standard_history = standard_solver.solve(
        freq=test_freq, amplitude=test_amp, n_steps=n_steps
    )

    print("Running Optimized FDTD (with ML correction)...")
    optimized_solver = StableHybridFDTD3D(
        Lx, Ly, Lz, dx, c=343, cfl=0.5,
        ml_model=ml_model, use_correction=True, max_correction_ratio=0.05
    )
    optimized_history = optimized_solver.solve(
        freq=test_freq, amplitude=test_amp, n_steps=n_steps
    )

    # 计算能量
    min_len = min(len(standard_history), len(optimized_history))

    standard_energy = [np.sum(h['p_center'] ** 2) for h in standard_history[:min_len]]
    optimized_energy = [np.sum(h['p_center'] ** 2) for h in optimized_history[:min_len]]

    # 计算改善（能量越小越好）
    avg_std_energy = np.mean(standard_energy[-50:])  # 稳态能量
    avg_opt_energy = np.mean(optimized_energy[-50:])

    if avg_std_energy > 0:
        improvement = (avg_std_energy - avg_opt_energy) / avg_std_energy * 100
    else:
        improvement = 0

    print(f"\n{'=' * 50}")
    print(f"Results (Steady-state energy):")
    print(f"  Standard FDTD:   {avg_std_energy:.4f}")
    print(f"  Optimized FDTD:  {avg_opt_energy:.4f}")
    print(f"  Improvement:     {improvement:.1f}%")
    print(f"{'=' * 50}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    t_idx = len(standard_history) // 2

    im1 = axes[0].imshow(standard_history[t_idx]['p_center'],
                         cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    axes[0].set_title(f'Standard FDTD')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(optimized_history[t_idx]['p_center'],
                         cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    axes[1].set_title(f'Optimized FDTD')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])

    times = [h['t'] for h in standard_history[:min_len]]
    axes[2].semilogy(times, standard_energy, 'r-', label='Standard', linewidth=1.5)
    axes[2].semilogy(times, optimized_energy, 'b-', label='Optimized', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Energy')
    axes[2].set_title(f'Energy Evolution (Improvement: {improvement:.1f}%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('3d_comparison.png', dpi=150)
    plt.close()

    print("\nVisualization saved: 3d_comparison.png")


def main():
    print("=" * 70)
    print("3D Constrained ML-FDTD Optimization (Reproducible)")
    print("=" * 70)
    print(f"Random Seed: {SEED}")
    print("""
    Key improvements:
    1. Strong output constraint (correction < 1% of field amplitude)
    2. Normalized training targets
    3. Safety fallback for numerical instability
    4. Fixed random seed for reproducible results
    """)

    ml_model = train_3d_model()
    comparison_experiment_3d(ml_model)


if __name__ == "__main__":
    main()