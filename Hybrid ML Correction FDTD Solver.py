import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class AnalyticalSolution:
    def __init__(self, Lx, Ly, Lz, c=343):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.c = c

    def get_pressure(self, x, y, z, t, freq, amplitude):
        return amplitude * np.cos(2 * np.pi * freq / self.c * x) * np.sin(2 * np.pi * freq * t)


class StandardFDTD3DGPU:
    """Standard FDTD solver"""

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

        print(f"Grid: {self.Nx} x {self.Ny} x {self.Nz} = {self.Nx * self.Ny * self.Nz:,} points")
        print(f"dt: {self.dt * 1e6:.2f} μs")

    def step(self, p, p_prev, src_val, src_ix, src_iy, src_iz):
        p_new = torch.zeros_like(p)
        p_new[1:-1, 1:-1, 1:-1] = (
                2 * p[1:-1, 1:-1, 1:-1] - p_prev[1:-1, 1:-1, 1:-1] +
                self.lambda2 * (
                        p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                        p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                        p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] -
                        6 * p[1:-1, 1:-1, 1:-1]
                )
        )

        p_new[0, :, :] = p_new[1, :, :]
        p_new[-1, :, :] = p_new[-2, :, :]
        p_new[:, 0, :] = p_new[:, 1, :]
        p_new[:, -1, :] = p_new[:, -2, :]
        p_new[:, :, 0] = p_new[:, :, 1]
        p_new[:, :, -1] = p_new[:, :, -2]
        p_new[src_ix, src_iy, src_iz] += src_val

        return p_new

    def solve(self, freq, amplitude, n_steps, source_pos=None, receiver_pos=None, record_every=10):
        p = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)
        p_prev = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)

        if source_pos is None:
            src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        else:
            src_ix, src_iy, src_iz = source_pos

        if receiver_pos is None:
            rx, ry, rz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        else:
            rx, ry, rz = receiver_pos

        history = []
        omega = 2 * np.pi * freq

        for step in range(n_steps):
            t = step * self.dt
            src_val = amplitude * np.sin(omega * t)
            src_tensor = torch.tensor(src_val, dtype=torch.float32, device=device)

            p_new = self.step(p, p_prev, src_tensor, src_ix, src_iy, src_iz)

            if step % record_every == 0:
                history.append({
                    'step': step,
                    't': t,
                    'p_receiver': p_new[rx, ry, rz].cpu().item(),
                })

            p_prev, p = p, p_new

        return history


class SimpleMLCorrector(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        self.strength = nn.Parameter(torch.tensor(0.05))

    def forward(self, p_curr, p_prev, freq, amplitude, time):
        freq_norm = freq / 500.0
        amp_norm = amplitude / 10.0
        time_norm = time / 0.02
        p_curr_norm = p_curr / (amplitude + 1e-6)
        p_prev_norm = p_prev / (amplitude + 1e-6)

        x = torch.cat([freq_norm, amp_norm, time_norm, p_curr_norm, p_prev_norm], dim=1)
        ratio = self.net(x)  # Correction ratio [-1, 1]

        # Correction value = ratio * strength * amplitude
        correction = ratio * self.strength * amplitude
        return correction


class TrainingDataGeneratorSimple:
    """Generate simple training data - record only source point data"""

    def __init__(self, Lx, Ly, Lz, dx, c=343):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = dx
        self.c = c
        self.dt = 0.5 * dx / c
        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1

    def generate_training_data(self, n_samples=200, n_steps=200):
        """Generate training data - record only source point values"""
        print("Generating training data (source point only)...")
        start_time = time.time()

        src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        lambda2 = 0.5 ** 2

        # Storage for data
        p_curr_list = []
        p_prev_list = []
        freq_list = []
        amp_list = []
        time_list = []
        target_correction_list = []

        # Precompute coordinate grids for analytical solution
        x = torch.linspace(0, self.Lx, self.Nx, device=device)
        y = torch.linspace(0, self.Ly, self.Ny, device=device)
        z = torch.linspace(0, self.Lz, self.Nz, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        for sample_idx in range(n_samples):
            freq = np.random.uniform(100, 300)
            amplitude = np.random.uniform(3.0, 7.0)
            omega = 2 * np.pi * freq

            freq_tensor = torch.tensor(freq, dtype=torch.float32, device=device)
            amp_tensor = torch.tensor(amplitude, dtype=torch.float32, device=device)

            p = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)
            p_prev = torch.zeros_like(p)

            for step in range(n_steps):
                t = step * self.dt
                src_val = amplitude * np.sin(omega * t)
                src_tensor = torch.tensor(src_val, dtype=torch.float32, device=device)

                # FDTD update
                p_new = torch.zeros_like(p)
                p_new[1:-1, 1:-1, 1:-1] = (
                        2 * p[1:-1, 1:-1, 1:-1] - p_prev[1:-1, 1:-1, 1:-1] +
                        lambda2 * (
                                p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                                p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                                p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] -
                                6 * p[1:-1, 1:-1, 1:-1]
                        )
                )
                p_new[0, :, :] = p_new[1, :, :]
                p_new[-1, :, :] = p_new[-2, :, :]
                p_new[:, 0, :] = p_new[:, 1, :]
                p_new[:, -1, :] = p_new[:, -2, :]
                p_new[:, :, 0] = p_new[:, :, 1]
                p_new[:, :, -1] = p_new[:, :, -2]
                p_new[src_ix, src_iy, src_iz] += src_tensor

                # Analytical solution
                t_next = t + self.dt
                exact = amp_tensor * torch.cos(2 * np.pi * freq_tensor / self.c * X) * torch.sin(
                    2 * np.pi * freq_tensor * (t + self.dt)
                )
                exact_src = exact[src_ix, src_iy, src_iz].item()
                fdtd_src = p_new[src_ix, src_iy, src_iz].item()

                # Required correction value
                needed_correction = exact_src - fdtd_src

                # Collect data after steady state
                if step > 50 and step % 5 == 0:
                    p_curr_list.append(p[src_ix, src_iy, src_iz].item())
                    p_prev_list.append(p_prev[src_ix, src_iy, src_iz].item())
                    freq_list.append(freq)
                    amp_list.append(amplitude)
                    time_list.append(t)
                    target_correction_list.append(needed_correction)

                p_prev, p = p, p_new

            if (sample_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Generated {sample_idx + 1}/{n_samples} samples (elapsed: {elapsed:.1f}s)")

        # Convert to tensors
        p_curr = torch.FloatTensor(p_curr_list).view(-1, 1)
        p_prev = torch.FloatTensor(p_prev_list).view(-1, 1)
        freqs = torch.FloatTensor(freq_list).view(-1, 1)
        amps = torch.FloatTensor(amp_list).view(-1, 1)
        times = torch.FloatTensor(time_list).view(-1, 1)
        targets = torch.FloatTensor(target_correction_list).view(-1, 1)

        # Normalize target
        target_mean = targets.mean().item()
        target_std = targets.std().item()
        if target_std < 1e-6:
            target_std = 1.0
        targets_norm = (targets - target_mean) / target_std

        elapsed = time.time() - start_time
        print(f"\nGenerated {len(p_curr)} training samples in {elapsed:.1f}s")
        print(f"Target - mean: {target_mean:.6f}, std: {target_std:.6f}")
        print(f"Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")

        return (p_curr, p_prev, freqs, amps, times), targets_norm, target_mean, target_std


def train_model():
    print("\n" + "=" * 70)
    print("Training Simple ML Corrector (Point-wise Prediction)")
    print("=" * 70)

    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = 0.05

    print("\nGenerating training data...")
    generator = TrainingDataGeneratorSimple(Lx, Ly, Lz, dx, c=343)
    (p_curr, p_prev, freqs, amps, times), targets_norm, target_mean, target_std = generator.generate_training_data(
        n_samples=200, n_steps=200
    )

    # Split dataset
    n_total = len(p_curr)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    p_curr_train = p_curr[train_idx].to(device)
    p_prev_train = p_prev[train_idx].to(device)
    freqs_train = freqs[train_idx].to(device)
    amps_train = amps[train_idx].to(device)
    times_train = times[train_idx].to(device)
    targets_train = targets_norm[train_idx].to(device)

    p_curr_val = p_curr[val_idx].to(device)
    p_prev_val = p_prev[val_idx].to(device)
    freqs_val = freqs[val_idx].to(device)
    amps_val = amps[val_idx].to(device)
    times_val = times[val_idx].to(device)
    targets_val = targets_norm[val_idx].to(device)

    print(f"Training: {len(train_idx)}, Validation: {len(val_idx)}")

    # Create simple model
    model = SimpleMLCorrector(input_dim=5, hidden_dim=64)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    batch_size = 256
    train_dataset = TensorDataset(p_curr_train, p_prev_train, freqs_train, amps_train, times_train, targets_train)
    val_dataset = TensorDataset(p_curr_val, p_prev_val, freqs_val, amps_val, times_val, targets_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_losses = []
    val_losses = []

    print(f"\nStarting training...")
    best_val_loss = float('inf')
    epochs = 300

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            p_c, p_p, f, a, t, target = batch

            optimizer.zero_grad()
            output = model(p_c, p_p, f, a, t)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                p_c, p_p, f, a, t, target = batch
                output = model(p_c, p_p, f, a, t)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'target_mean': target_mean,
                'target_std': target_std,
            }, 'best_simple_corrector.pth')

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1:4d}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Strength={model.strength.item():.4f}")

    # Load best model
    checkpoint = torch.load('best_simple_corrector.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']

    print(f"\n✓ Best model loaded (val_loss: {best_val_loss:.6f})")
    print(f"  Correction strength: {model.strength.item():.4f}")
    print(f"  Target mean: {target_mean:.6f}, std: {target_std:.6f}")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Training Loss', alpha=0.7)
    axes[0].plot(val_losses, label='Validation Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Prediction vs truth
    model.eval()
    with torch.no_grad():
        val_outputs = model(p_curr_val, p_prev_val, freqs_val, amps_val, times_val)
        val_outputs_denorm = val_outputs.cpu().numpy() * target_std + target_mean
        targets_denorm = targets_val.cpu().numpy() * target_std + target_mean

        axes[1].scatter(targets_denorm, val_outputs_denorm, alpha=0.3, s=1)
        lim = max(abs(targets_denorm).max(), abs(val_outputs_denorm).max())
        axes[1].plot([-lim, lim], [-lim, lim], 'r--', label='Ideal')
        axes[1].set_xlabel('True Correction (Pa)')
        axes[1].set_ylabel('Predicted Correction (Pa)')
        axes[1].set_title(f'Prediction vs Truth')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_simple.png', dpi=150)
    plt.close()

    return model, target_mean, target_std


class OptimizedFDTD3DGPU:
    def __init__(self, Lx, Ly, Lz, dx, c=343, cfl=0.5, ml_model=None, target_mean=0, target_std=1):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c
        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1
        self.lambda2 = cfl ** 2
        self.ml_model = ml_model
        self.target_mean = target_mean
        self.target_std = target_std
        if ml_model is not None:
            self.ml_model.eval()

    def solve(self, freq, amplitude, n_steps, source_pos=None, receiver_pos=None,
              record_every=10, use_correction=True):
        p = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)
        p_prev = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)

        if source_pos is None:
            src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        else:
            src_ix, src_iy, src_iz = source_pos

        if receiver_pos is None:
            rx, ry, rz = src_ix, src_iy, src_iz
        else:
            rx, ry, rz = receiver_pos

        history = []
        omega = 2 * np.pi * freq

        for step in range(n_steps):
            t = step * self.dt
            src_val = amplitude * np.sin(omega * t)
            src_tensor = torch.tensor(src_val, dtype=torch.float32, device=device)

            p_new = torch.zeros_like(p)
            p_new[1:-1, 1:-1, 1:-1] = (
                    2 * p[1:-1, 1:-1, 1:-1] - p_prev[1:-1, 1:-1, 1:-1] +
                    self.lambda2 * (
                            p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
                            p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
                            p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2] -
                            6 * p[1:-1, 1:-1, 1:-1]
                    )
            )

            p_new[0, :, :] = p_new[1, :, :]
            p_new[-1, :, :] = p_new[-2, :, :]
            p_new[:, 0, :] = p_new[:, 1, :]
            p_new[:, -1, :] = p_new[:, -2, :]
            p_new[:, :, 0] = p_new[:, :, 1]
            p_new[:, :, -1] = p_new[:, :, -2]
            p_new[src_ix, src_iy, src_iz] += src_tensor

            if use_correction and self.ml_model is not None and step > 50:
                with torch.no_grad():
                    p_curr_tensor = torch.tensor([[p[src_ix, src_iy, src_iz].item()]], device=device)
                    p_prev_tensor = torch.tensor([[p_prev[src_ix, src_iy, src_iz].item()]], device=device)
                    freq_tensor = torch.tensor([[freq]], device=device)
                    amp_tensor = torch.tensor([[amplitude]], device=device)
                    t_tensor = torch.tensor([[t]], device=device)

                    # Predict normalized correction
                    norm_correction = self.ml_model(p_curr_tensor, p_prev_tensor, freq_tensor, amp_tensor, t_tensor)
                    # Denormalize
                    correction = norm_correction.item() * self.target_std + self.target_mean
                    # Limit correction amplitude
                    correction = np.clip(correction, -amplitude * 0.2, amplitude * 0.2)

                    p_new[src_ix, src_iy, src_iz] += correction

            if step % record_every == 0:
                history.append({
                    'step': step,
                    't': t,
                    'p_receiver': p_new[rx, ry, rz].cpu().item(),
                })

            p_prev, p = p, p_new

        return history


def comparison_experiment(ml_model, target_mean, target_std):
    print("\n" + "=" * 70)
    print("Comparison: Analytical vs Standard FDTD vs Optimized FDTD")
    print("=" * 70)

    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = 0.05
    c = 343

    test_freq = 171.5
    test_amp = 5.0
    n_steps = 400

    print(f"\nTest: f={test_freq}Hz, A={test_amp}Pa")

    analytical = AnalyticalSolution(Lx, Ly, Lz, c)
    src_pos = (int(Lx / dx) // 2, int(Ly / dx) // 2, int(Lz / dx) // 2)

    print("\n1. Running Standard FDTD...")
    fdtd = StandardFDTD3DGPU(Lx, Ly, Lz, dx, c=c, cfl=0.5)
    standard_history = fdtd.solve(freq=test_freq, amplitude=test_amp, n_steps=n_steps,
                                  source_pos=src_pos, record_every=5)

    print("2. Running Optimized FDTD...")
    optimized = OptimizedFDTD3DGPU(Lx, Ly, Lz, dx, c=c, cfl=0.5,
                                   ml_model=ml_model, target_mean=target_mean, target_std=target_std)
    optimized_history = optimized.solve(freq=test_freq, amplitude=test_amp, n_steps=n_steps,
                                        source_pos=src_pos, use_correction=True, record_every=5)

    times = []
    analytical_pressure = []
    std_pressure = []
    opt_pressure = []

    min_len = min(len(standard_history), len(optimized_history))
    for i in range(min_len):
        t = standard_history[i]['t']
        times.append(t)
        analytical_pressure.append(analytical.get_pressure(0, 0, 0, t, test_freq, test_amp))
        std_pressure.append(standard_history[i]['p_receiver'])
        opt_pressure.append(optimized_history[i]['p_receiver'])

    start_idx = -100
    mae_standard = np.mean(np.abs(np.array(analytical_pressure[start_idx:]) - np.array(std_pressure[start_idx:])))
    mae_optimized = np.mean(np.abs(np.array(analytical_pressure[start_idx:]) - np.array(opt_pressure[start_idx:])))
    improvement = (mae_standard - mae_optimized) / mae_standard * 100

    print(f"\n{'=' * 60}")
    print(f"Results at Source Point:")
    print(f"  Standard FDTD MAE:      {mae_standard:.4f} Pa")
    print(f"  Optimized FDTD MAE:     {mae_optimized:.4f} Pa")
    print(f"  Improvement:            {improvement:.1f}%")
    print(f"{'=' * 60}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(times, analytical_pressure, 'g-', label='Analytical', linewidth=2)
    axes[0, 0].plot(times, std_pressure, 'r-', label='Standard FDTD', linewidth=1.5)
    axes[0, 0].plot(times, opt_pressure, 'b-', label='Optimized FDTD', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Pressure (Pa)')
    axes[0, 0].set_title('Pressure at Source Point')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times[start_idx:], analytical_pressure[start_idx:], 'g-')
    axes[0, 1].plot(times[start_idx:], std_pressure[start_idx:], 'r-')
    axes[0, 1].plot(times[start_idx:], opt_pressure[start_idx:], 'b-')
    axes[0, 1].set_title(f'Steady State (Improvement: {improvement:.1f}%)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(times, np.abs(np.array(analytical_pressure) - np.array(std_pressure)), 'r-')
    axes[1, 0].semilogy(times, np.abs(np.array(analytical_pressure) - np.array(opt_pressure)), 'b-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Absolute Error (Pa)')
    axes[1, 0].set_title('Error Evolution')
    axes[1, 0].grid(True, alpha=0.3)

    methods = ['Standard FDTD', 'Optimized FDTD']
    mae_values = [mae_standard, mae_optimized]
    colors = ['red', 'blue']
    bars = axes[1, 1].bar(methods, mae_values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Mean Absolute Error (Pa)')
    axes[1, 1].set_title('MAE Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mae_values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('optimized_comparison.png', dpi=300)
    plt.close()

    return improvement


def main():
    print("=" * 70)
    print("Simple ML Corrector for FDTD")
    print("=" * 70)


    ml_model, target_mean, target_std = train_model()
    improvement = comparison_experiment(ml_model, target_mean, target_std)

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {improvement:.1f}% Improvement")
    print("=" * 70)


if __name__ == "__main__":
    main()
