import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def set_seed(seed=31):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(seed=31)


class AnalyticalSolution3D:
    def __init__(self, Lx, Ly, Lz, dx, c=343):
        self.dx = dx
        self.c = c
        self.sx, self.sy, self.sz = Lx / 2, Ly / 2, Lz / 2
        self.r0 = dx

    def ricker(self, tau, freq):
        a = torch.pi * freq * (tau - 1.0 / freq)
        return (1.0 - 2.0 * a * a) * torch.exp(-(a * a))

    def field(self, X, Y, Z, t, freq, strength):
        r = torch.sqrt((X - self.sx) ** 2 + (Y - self.sy) ** 2 + (Z - self.sz) ** 2) + self.r0
        tau = t - r / self.c
        return torch.where(tau > 0, strength * self.ricker(tau, freq) / (4.0 * torch.pi * r), torch.zeros_like(r))

    def get_pressure_at_point(self, x, y, z, t, freq, strength):
        r = np.sqrt((x - self.sx) ** 2 + (y - self.sy) ** 2 + (z - self.sz) ** 2) + self.dx
        tau = t - r / self.c
        if tau <= 0:
            return 0.0
        a = np.pi * freq * (tau - 1.0 / freq)
        ricker_val = (1.0 - 2.0 * a * a) * np.exp(-(a * a))
        return strength * ricker_val / (4.0 * np.pi * r)


class FDTD3DSolverGPU:
    def __init__(self, Lx, Ly, Lz, dx, c=343, cfl=0.5):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dx = dx
        self.c = c
        self.cfl = cfl
        self.dt = cfl * dx / c
        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1
        self.lambda2 = cfl ** 2

        print(f"Grid: {self.Nx} x {self.Ny} x {self.Nz}")
        print(f"Total points: {self.Nx * self.Ny * self.Nz:,}")
        print(f"dt = {self.dt * 1e6:.3f} us")
        print("Boundary conditions: RIGID")
        print("Source type: POINT with Ricker wavelet")

    def step(self, p, p_prev, src, ix, iy, iz):
        p_new = torch.zeros_like(p)

        p_new[1:-1, 1:-1, 1:-1] = (
                2.0 * p[1:-1, 1:-1, 1:-1]
                - p_prev[1:-1, 1:-1, 1:-1]
                + self.lambda2 * (
                        p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
                        + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
                        + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
                        - 6.0 * p[1:-1, 1:-1, 1:-1]
                )
        )

        # Rigid boundary conditions
        p_new[0, :, :] = p_new[1, :, :]
        p_new[-1, :, :] = p_new[-2, :, :]
        p_new[:, 0, :] = p_new[:, 1, :]
        p_new[:, -1, :] = p_new[:, -2, :]
        p_new[:, :, 0] = p_new[:, :, 1]
        p_new[:, :, -1] = p_new[:, :, -2]

        # Point source
        p_new[ix, iy, iz] += src

        return p_new

    def solve(self, freq, source_strength, n_steps, source_pos=None, receiver_pos=None, record_every=10):
        p = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)
        p_prev = torch.zeros_like(p)

        if source_pos is None:
            src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        else:
            src_ix, src_iy, src_iz = source_pos

        if receiver_pos is None:
            rx, ry, rz = src_ix, src_iy, src_iz
        else:
            rx, ry, rz = receiver_pos

        history = []
        all_fields = []

        # Precompute constant tensors
        freq_t = torch.tensor(freq, dtype=torch.float32, device=device)
        strength_t = torch.tensor(source_strength, dtype=torch.float32, device=device)
        dt_t = torch.tensor(self.dt, dtype=torch.float32, device=device)
        t0 = 1.0 / freq  # Peak time of Ricker wavelet

        for step in range(n_steps):
            t = step * self.dt
            tau = t - t0
            a = torch.pi * freq_t * tau
            src = strength_t * (1.0 - 2.0 * a * a) * torch.exp(-(a * a))

            p_new = self.step(p, p_prev, src, src_ix, src_iy, src_iz)

            if step % record_every == 0:
                history.append({
                    "step": step,
                    "t": t,
                    "p_receiver": p_new[rx, ry, rz].cpu().item()
                })
                all_fields.append(p_new.detach().cpu().numpy())

            p_prev = p
            p = p_new

        return history, all_fields


class SimpleFullFieldCorrector(nn.Module):
    def __init__(self, nx, ny, nz, hidden_channels):
        super().__init__()

        self.nx, self.ny, self.nz = nx, ny, nz

        self.enc1 = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose3d(hidden_channels * 8, hidden_channels * 4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(hidden_channels * 8, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose3d(hidden_channels * 4, hidden_channels * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose3d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels // 2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.correction_strength = nn.Parameter(torch.tensor(0.05))

    def forward(self, p_curr, p_prev, freq, source_strength, time):
        p_curr = p_curr.float()
        p_prev = p_prev.float()
        freq = freq.float()
        source_strength = source_strength.float()
        time = time.float()

        batch_size = p_curr.shape[0]

        max_val = source_strength.view(batch_size, 1, 1, 1, 1) + 1e-6
        p_curr_norm = p_curr.unsqueeze(1) / max_val
        p_prev_norm = p_prev.unsqueeze(1) / max_val
        p_curr_norm = torch.clamp(p_curr_norm, -2, 2)
        p_prev_norm = torch.clamp(p_prev_norm, -2, 2)

        x = torch.cat([p_curr_norm, p_prev_norm], dim=1)

        e1 = self.enc1(x)
        d1 = self.pool1(e1)

        e2 = self.enc2(d1)
        d2 = self.pool2(e2)

        e3 = self.enc3(d2)
        d3 = self.pool3(e3)

        b = self.bottleneck(d3)

        freq_norm = (freq / 500).view(batch_size, 1)
        strength_norm = (source_strength / 10).view(batch_size, 1)
        time_norm = (time / 0.02).view(batch_size, 1)
        param = torch.cat([freq_norm, strength_norm, time_norm], dim=1)
        param_cond = self.param_encoder(param)
        param_cond = param_cond.view(batch_size, -1, 1, 1, 1)

        b = b + param_cond * 0.1

        u3 = self.up3(b)
        if u3.shape[-3:] != e3.shape[-3:]:
            u3 = torch.nn.functional.interpolate(u3, size=e3.shape[-3:], mode='trilinear', align_corners=False)
        u3 = torch.cat([u3, e3], dim=1)
        d3_out = self.dec3(u3)

        u2 = self.up2(d3_out)
        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = torch.nn.functional.interpolate(u2, size=e2.shape[-3:], mode='trilinear', align_corners=False)
        u2 = torch.cat([u2, e2], dim=1)
        d2_out = self.dec2(u2)

        u1 = self.up1(d2_out)
        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = torch.nn.functional.interpolate(u1, size=e1.shape[-3:], mode='trilinear', align_corners=False)
        u1 = torch.cat([u1, e1], dim=1)
        d1_out = self.dec1(u1)

        norm_correction = self.output(d1_out)
        correction = norm_correction * self.correction_strength * source_strength.view(batch_size, 1, 1, 1, 1)

        return correction.squeeze(1)


class TrainingDataGenerator3D:
    """Generate complete 3D training data"""

    def __init__(self, Lx, Ly, Lz, dx, c=343):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dx = dx
        self.c = c
        self.dt = 0.5 * dx / c
        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1
        self.src_ix = self.Nx // 2
        self.src_iy = self.Ny // 2
        self.src_iz = self.Nz // 2

    def generate_training_data(self, n_samples, n_steps=150):
        print(f"Generating full 3D training data...")
        start_time = time.time()

        x = torch.linspace(0, self.Lx, self.Nx, device=device)
        y = torch.linspace(0, self.Ly, self.Ny, device=device)
        z = torch.linspace(0, self.Lz, self.Nz, device=device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        lambda2 = 0.5 ** 2

        p_curr_list = []
        p_prev_list = []
        freq_list = []
        strength_list = []
        time_list = []
        target_correction_list = []

        analytical = AnalyticalSolution3D(self.Lx, self.Ly, self.Lz, self.dx, self.c)

        with tqdm(total=n_samples, desc="Generating samples", unit="sample", ncols=100) as pbar:
            for sample_idx in range(n_samples):
                freq = np.random.uniform(50, 150)
                source_strength = np.random.uniform(0.5, 1.5)
                omega = 2 * np.pi * freq

                freq_t = torch.tensor(freq, dtype=torch.float32, device=device)
                strength_t = torch.tensor(source_strength, dtype=torch.float32, device=device)
                t0 = 1.0 / freq

                p = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)
                p_prev = torch.zeros_like(p)

                for step in range(n_steps):
                    t = step * self.dt

                    # Ricker wavelet source
                    tau = t - t0
                    a = np.pi * freq * tau
                    src_val = source_strength * (1.0 - 2.0 * a * a) * np.exp(-(a * a))
                    src_tensor = torch.tensor(src_val, dtype=torch.float32, device=device)

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
                    p_new[self.src_ix, self.src_iy, self.src_iz] += src_tensor

                    t_next = t + self.dt
                    exact = analytical.field(X, Y, Z, t_next, freq_t, strength_t)

                    needed_correction = exact - p_new
                    needed_correction = torch.clamp(needed_correction, -source_strength * 0.3, source_strength * 0.3)

                    if step > 50 and step % 5 == 0:
                        p_curr_list.append(p.cpu().numpy().copy())
                        p_prev_list.append(p_prev.cpu().numpy().copy())
                        freq_list.append(freq)
                        strength_list.append(source_strength)
                        time_list.append(t)
                        target_correction_list.append(needed_correction.cpu().numpy().copy())

                    p_prev, p = p, p_new

                pbar.update(1)
                pbar.set_postfix({
                    'freq': f'{freq:.0f}Hz',
                    'strength': f'{source_strength:.1f}',
                    'collected': len(p_curr_list)
                })

                if (sample_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

        p_curr = torch.FloatTensor(np.array(p_curr_list))
        p_prev = torch.FloatTensor(np.array(p_prev_list))
        freqs = torch.FloatTensor(np.array(freq_list)).view(-1, 1)
        strengths = torch.FloatTensor(np.array(strength_list)).view(-1, 1)
        times = torch.FloatTensor(np.array(time_list)).view(-1, 1)
        targets = torch.FloatTensor(np.array(target_correction_list))

        elapsed = time.time() - start_time
        print(f"\nGenerated {len(p_curr)} training samples in {elapsed:.1f}s")

        target_mean = targets.mean().item()
        target_std = targets.std().item()
        if target_std < 1e-6:
            target_std = 1.0
        targets_norm = (targets - target_mean) / target_std

        return (p_curr, p_prev, freqs, strengths, times), targets_norm, target_mean, target_std


def train_model():
    print("\n" + "=" * 70)
    print("Training Full Field Corrector")
    print("=" * 70)

    Lx, Ly, Lz = 2.0, 2.0, 2.0  # Expanded computational domain
    dx = 0.05
    Nx, Ny, Nz = int(Lx / dx) + 1, int(Ly / dx) + 1, int(Lz / dx) + 1
    print(f"Grid: {Nx} x {Ny} x {Nz} = {Nx * Ny * Nz:,} points")

    print("\nGenerating training data...")
    generator = TrainingDataGenerator3D(Lx, Ly, Lz, dx, c=343)
    (p_curr, p_prev, freqs, strengths, times), targets_norm, target_mean, target_std = generator.generate_training_data(
        n_samples=30, n_steps=150
    )

    n_total = len(p_curr)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    p_curr_train = p_curr[train_idx].to(device)
    p_prev_train = p_prev[train_idx].to(device)
    freqs_train = freqs[train_idx].to(device)
    strengths_train = strengths[train_idx].to(device)
    times_train = times[train_idx].to(device)
    targets_train = targets_norm[train_idx].to(device)

    p_curr_val = p_curr[val_idx].to(device)
    p_prev_val = p_prev[val_idx].to(device)
    freqs_val = freqs[val_idx].to(device)
    strengths_val = strengths[val_idx].to(device)
    times_val = times[val_idx].to(device)
    targets_val = targets_norm[val_idx].to(device)

    print(f"Training: {len(train_idx)}, Validation: {len(val_idx)}")

    model = SimpleFullFieldCorrector(nx=Nx, ny=Ny, nz=Nz, hidden_channels=32)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    batch_size = 64
    train_dataset = TensorDataset(p_curr_train, p_prev_train, freqs_train, strengths_train, times_train, targets_train)
    val_dataset = TensorDataset(p_curr_val, p_prev_val, freqs_val, strengths_val, times_val, targets_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_losses = []
    val_losses = []

    print(f"\nStarting training...")
    best_val_loss = float('inf')
    epochs = 1500
    print_interval = 50

    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0
        for batch in train_loader:
            p_c, p_p, f, s, t, target = batch

            optimizer.zero_grad()
            output = model(p_c, p_p, f, s, t)
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
                p_c, p_p, f, s, t, target = batch
                output = model(p_c, p_p, f, s, t)
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
            }, 'best_full_field_corrector.pth')

        epoch_duration = time.time() - epoch_start

        if (epoch + 1) % print_interval == 0:
            avg_epoch_time = (time.time() - total_start_time) / (epoch + 1)
            remaining_epochs = epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs

            print(f"Epoch {epoch + 1:4d}/{epochs}: "
                  f"Train Loss={train_loss:.6f}, "
                  f"Val Loss={val_loss:.6f}, "
                  f"Strength={model.correction_strength.item():.4f}, "
                  f"Time={epoch_duration:.2f}s, "
                  f"ETA={eta / 60:.1f}min")

    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f}s ({total_time / 60:.2f}min)")

    checkpoint = torch.load('best_full_field_corrector.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n✓ Best model loaded (val_loss: {best_val_loss:.6f})")
    print(f"  Correction strength: {model.correction_strength.item():.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('training_history_3d.png', dpi=150)
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

    def solve(self, freq, source_strength, n_steps, source_pos=None, receiver_pos=None,
              record_every=10, use_correction=True):
        p = torch.zeros((self.Nx, self.Ny, self.Nz), dtype=torch.float32, device=device)
        p_prev = torch.zeros_like(p)

        if source_pos is None:
            src_ix, src_iy, src_iz = self.Nx // 2, self.Ny // 2, self.Nz // 2
        else:
            src_ix, src_iy, src_iz = source_pos

        if receiver_pos is None:
            rx, ry, rz = src_ix, src_iy, src_iz
        else:
            rx, ry, rz = receiver_pos

        history = []
        all_fields = []

        freq_t = torch.tensor(freq, dtype=torch.float32, device=device)
        strength_t = torch.tensor(source_strength, dtype=torch.float32, device=device)
        t0 = 1.0 / freq

        for step in range(n_steps):
            t = step * self.dt
            tau = t - t0
            a = torch.pi * freq_t * tau
            src = strength_t * (1.0 - 2.0 * a * a) * torch.exp(-(a * a))

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
            p_new[src_ix, src_iy, src_iz] += src

            if use_correction and self.ml_model is not None and step > 50:
                with torch.no_grad():
                    p_curr_tensor = p.unsqueeze(0).float()
                    p_prev_tensor = p_prev.unsqueeze(0).float()
                    freq_tensor = torch.tensor([[freq]], dtype=torch.float32, device=device)
                    strength_tensor = torch.tensor([[source_strength]], dtype=torch.float32, device=device)
                    t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)

                    norm_correction = self.ml_model(p_curr_tensor, p_prev_tensor, freq_tensor, strength_tensor,
                                                    t_tensor)
                    correction_field = norm_correction
                    max_correction = source_strength * 0.05
                    correction_field = torch.clamp(correction_field, -max_correction, max_correction)
                    p_new = p_new + correction_field.squeeze(0)

            if step % record_every == 0:
                history.append({
                    'step': step,
                    't': t,
                    'p_receiver': p_new[rx, ry, rz].cpu().item(),
                })
                all_fields.append(p_new.detach().cpu().numpy())

            p_prev = p
            p = p_new

        return history, all_fields


def save_separate_slices(analytical_fields, standard_fields, optimized_fields, times,
                         test_freq, source_strength, dx, src_pos, save_dir='pressure_slices'):
    os.makedirs(save_dir, exist_ok=True)

    if len(analytical_fields) == 0:
        print("No fields to save!")
        return

    mid_z = analytical_fields[0].shape[2] // 2
    src_x = src_pos[0] * dx
    src_y = src_pos[1] * dx
    num_frames = len(analytical_fields)

    print(f"Saving {num_frames} image sets to {save_dir}...")

    for idx in range(num_frames):
        t = times[idx]

        # Extract slices from each field
        analytical_slice = analytical_fields[idx][:, :, mid_z]
        standard_slice = standard_fields[idx][:, :, mid_z]
        optimized_slice = optimized_fields[idx][:, :, mid_z]

        # Compute their respective ranges
        vmin_ana, vmax_ana = np.min(analytical_slice), np.max(analytical_slice)
        vmin_std, vmax_std = np.min(standard_slice), np.max(standard_slice)
        vmin_opt, vmax_opt = np.min(optimized_slice), np.max(optimized_slice)

        # Symmetrize (optional, facilitates observation of positive and negative)
        max_abs_ana = max(abs(vmin_ana), abs(vmax_ana))
        max_abs_std = max(abs(vmin_std), abs(vmax_std))
        max_abs_opt = max(abs(vmin_opt), abs(vmax_opt))

        vmin_ana, vmax_ana = -max_abs_ana, max_abs_ana
        vmin_std, vmax_std = -max_abs_std, max_abs_std
        vmin_opt, vmax_opt = -max_abs_opt, max_abs_opt

        # Set default range if too small
        if max_abs_ana < 1e-6:
            vmin_ana, vmax_ana = -1, 1
        if max_abs_std < 1e-6:
            vmin_std, vmax_std = -1, 1
        if max_abs_opt < 1e-6:
            vmin_opt, vmax_opt = -1, 1

        # 1. Analytical solution
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7))
        im1 = ax1.imshow(analytical_slice.T, origin='lower', cmap='seismic', aspect='auto',
                         extent=[0, 4.0, 0, 4.0], vmin=vmin_ana, vmax=vmax_ana)
        ax1.set_title(f'Analytical Solution\nTime: {t:.4f}s', fontsize=12)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.scatter(src_x, src_y, c='red', s=50, marker='*',
                    edgecolors='white', linewidths=1, label='Source')
        ax1.legend(loc='upper right')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Pressure (Pa)', fontsize=10)
        fig1.suptitle(f'Pressure Field at z = {mid_z * dx:.3f}m (Middle Slice)\n'
                      f'Frequency: {test_freq:.1f}Hz, Source Strength: {source_strength}\n'
                      f'Range: [{vmin_ana:.4f}, {vmax_ana:.4f}]', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'Analytical_t_{idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Standard FDTD
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        im2 = ax2.imshow(standard_slice.T, origin='lower', cmap='seismic', aspect='auto',
                         extent=[0, 4.0, 0, 4.0], vmin=vmin_std, vmax=vmax_std)
        ax2.set_title(f'Standard FDTD\nTime: {t:.4f}s', fontsize=12)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.scatter(src_x, src_y, c='red', s=50, marker='*',
                    edgecolors='white', linewidths=1, label='Source')
        ax2.legend(loc='upper right')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Pressure (Pa)', fontsize=10)
        fig2.suptitle(f'Pressure Field at z = {mid_z * dx:.3f}m (Middle Slice)\n'
                      f'Frequency: {test_freq:.1f}Hz, Source Strength: {source_strength}\n'
                      f'Range: [{vmin_std:.4f}, {vmax_std:.4f}]', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'Standard_FDTD_t_{idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Optimized FDTD
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 7))
        im3 = ax3.imshow(optimized_slice.T, origin='lower', cmap='seismic', aspect='auto',
                         extent=[0, 4.0, 0, 4.0], vmin=vmin_opt, vmax=vmax_opt)
        ax3.set_title(f'Optimized FDTD\nTime: {t:.4f}s', fontsize=12)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.scatter(src_x, src_y, c='red', s=50, marker='*',
                    edgecolors='white', linewidths=1, label='Source')
        ax3.legend(loc='upper right')
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_label('Pressure (Pa)', fontsize=10)
        fig3.suptitle(f'Pressure Field at z = {mid_z * dx:.3f}m (Middle Slice)\n'
                      f'Frequency: {test_freq:.1f}Hz, Source Strength: {source_strength}\n'
                      f'Range: [{vmin_opt:.4f}, {vmax_opt:.4f}]', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'Optimized_FDTD_t_{idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{num_frames} image sets")

    print(f"Completed: Saved {num_frames} images each")


def comparison_experiment(ml_model, target_mean, target_std):
    print("\n" + "=" * 70)
    print("Comparison: Analytical vs Standard FDTD vs Optimized FDTD")
    print("=" * 70)

    Lx, Ly, Lz = 2.0, 2.0, 2.0  # Expanded computational domain
    dx = 0.05
    c = 343

    test_freq = 100.0  # Using frequency from example
    source_strength = 50.0  # Using strength from example
    n_steps = 150
    record_every = 1  # Record every step, reduce data amount

    print(f"\nTest frequency: {test_freq} Hz")
    print(f"Source Strength: {source_strength}")
    print("Boundary conditions: RIGID")
    print("Source type: POINT with Ricker wavelet")

    analytical = AnalyticalSolution3D(Lx, Ly, Lz, dx, c)
    src_pos = (int(Lx / dx) // 2, int(Ly / dx) // 2, int(Lz / dx) // 2)

    x = torch.linspace(0, Lx, int(Lx / dx) + 1, device=device)
    y = torch.linspace(0, Ly, int(Ly / dx) + 1, device=device)
    z = torch.linspace(0, Lz, int(Lz / dx) + 1, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    print("\n1. Running Standard FDTD with Ricker wavelet...")
    fdtd = FDTD3DSolverGPU(Lx, Ly, Lz, dx, c=c, cfl=0.5)
    standard_history, standard_fields = fdtd.solve(freq=test_freq, source_strength=source_strength, n_steps=n_steps,
                                                   source_pos=src_pos, record_every=record_every)

    print("\n2. Running Optimized FDTD (Full Field Correction)...")
    optimized = OptimizedFDTD3DGPU(Lx, Ly, Lz, dx, c=c, cfl=0.5,
                                   ml_model=ml_model, target_mean=target_mean, target_std=target_std)
    optimized_history, optimized_fields = optimized.solve(freq=test_freq, source_strength=source_strength,
                                                          n_steps=n_steps,
                                                          source_pos=src_pos, use_correction=True,
                                                          record_every=record_every)

    print("\n3. Computing analytical fields...")
    analytical_fields = []
    times = []
    freq_t = torch.tensor(test_freq, dtype=torch.float32, device=device)
    strength_t = torch.tensor(source_strength, dtype=torch.float32, device=device)

    for record in standard_history:
        t = record['t']
        times.append(t)
        analytical_field = analytical.field(X, Y, Z, t, freq_t, strength_t).cpu().numpy()
        analytical_fields.append(analytical_field)

    src_x = src_pos[0] * dx
    src_y = src_pos[1] * dx
    src_z = src_pos[2] * dx

    # Compute source point pressure
    analytical_pressure = []
    std_pressure = []
    opt_pressure = []
    compare_times = []

    min_len = min(len(standard_history), len(optimized_history))

    print("\n" + "=" * 80)
    print("Source Point Pressure Comparison")
    print("=" * 80)
    print(f"{'Step':>6} {'Time(s)':>10} {'Analytical(Pa)':>14} {'Std FDTD(Pa)':>14} {'Opt FDTD(Pa)':>14}")
    print("-" * 70)

    for i in range(min_len):
        t = standard_history[i]['t']
        analytical_val = analytical.get_pressure_at_point(src_x, src_y, src_z, t, test_freq, source_strength)
        fdtd_val = standard_history[i]['p_receiver']
        opt_val = optimized_history[i]['p_receiver']

        compare_times.append(t)
        analytical_pressure.append(analytical_val)
        std_pressure.append(fdtd_val)
        opt_pressure.append(opt_val)

        if i % 50 == 0 or i == min_len - 1:
            print(f"{i:6d} {t:10.4f} {analytical_val:14.6f} {fdtd_val:14.6f} {opt_val:14.6f}")

    print("-" * 70)

    start_idx = int(len(compare_times) * 0.3)
    if start_idx >= len(compare_times):
        start_idx = 0

    mae_standard = np.mean(np.abs(np.array(analytical_pressure[start_idx:]) - np.array(std_pressure[start_idx:])))
    mae_optimized = np.mean(np.abs(np.array(analytical_pressure[start_idx:]) - np.array(opt_pressure[start_idx:])))
    mae_improvement = (mae_standard - mae_optimized) / mae_standard * 100 if mae_standard > 0 else 0

    print(f"\nSource Point Error Statistics:")
    print(f"  Standard FDTD MAE:      {mae_standard:.6f} Pa")
    print(f"  Optimized FDTD MAE:     {mae_optimized:.6f} Pa")
    print(f"  MAE Improvement:        {mae_improvement:.1f}%")

    # Plot source point pressure comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(compare_times, analytical_pressure, 'g-', label='Analytical', linewidth=2)
    axes[0, 0].plot(compare_times, std_pressure, 'r-', label='Standard FDTD', linewidth=1.5)
    axes[0, 0].plot(compare_times, opt_pressure, 'b-', label='Optimized FDTD', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Pressure (Pa)')
    axes[0, 0].set_title('Pressure at Source Point')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    if len(compare_times) > 100:
        steady_start = int(len(compare_times) * 0.7)
        axes[0, 1].plot(compare_times[steady_start:], analytical_pressure[steady_start:], 'g-', linewidth=2)
        axes[0, 1].plot(compare_times[steady_start:], std_pressure[steady_start:], 'r-', linewidth=1.5)
        axes[0, 1].plot(compare_times[steady_start:], opt_pressure[steady_start:], 'b-', linewidth=1.5)
        axes[0, 1].set_title(f'Steady State (MAE Improvement: {mae_improvement:.1f}%)')
    else:
        axes[0, 1].plot(compare_times[start_idx:], analytical_pressure[start_idx:], 'g-', linewidth=2)
        axes[0, 1].plot(compare_times[start_idx:], std_pressure[start_idx:], 'r-', linewidth=1.5)
        axes[0, 1].plot(compare_times[start_idx:], opt_pressure[start_idx:], 'b-', linewidth=1.5)
        axes[0, 1].set_title(f'Steady State (MAE Improvement: {mae_improvement:.1f}%)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Pressure (Pa)')
    axes[0, 1].grid(True)

    # Error evolution
    std_errors = np.abs(np.array(analytical_pressure) - np.array(std_pressure))
    opt_errors = np.abs(np.array(analytical_pressure) - np.array(opt_pressure))
    axes[1, 0].semilogy(compare_times, std_errors, 'r-', label='Standard FDTD Error', linewidth=1.5)
    axes[1, 0].semilogy(compare_times, opt_errors, 'b-', label='Optimized FDTD Error', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Absolute Error (Pa)')
    axes[1, 0].set_title('Error Evolution at Source Point')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # MAE comparison bar chart
    methods = ['Standard FDTD', 'Optimized FDTD']
    mae_values = [mae_standard, mae_optimized]
    bars = axes[1, 1].bar(methods, mae_values, color=['red', 'blue'], alpha=0.7)
    axes[1, 1].set_ylabel('Mean Absolute Error (Pa)')
    axes[1, 1].set_title(f'Source Point MAE Comparison (Improvement: {mae_improvement:.1f}%)')
    axes[1, 1].grid(True, axis='y')
    for bar, val in zip(bars, mae_values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{val:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('source_point_comparison.png', dpi=150)
    plt.close()
    print("Saved: source_point_comparison.png")

    # Save pressure field images (every 5 time steps)
    print("\n4. Saving pressure field images (every 5 time steps)...")

    total_frames = len(analytical_fields)
    save_indices = list(range(0, total_frames, 5))

    print(f"Total frames: {total_frames}")
    print(f"Saving {len(save_indices)} image sets (every 5 time steps)")

    analytical_fields_to_save = [analytical_fields[i] for i in save_indices]
    standard_fields_to_save = [standard_fields[i] for i in save_indices]
    optimized_fields_to_save = [optimized_fields[i] for i in save_indices]
    times_to_save = [times[i] for i in save_indices]

    save_separate_slices(
        analytical_fields_to_save,
        standard_fields_to_save,
        optimized_fields_to_save,
        times_to_save,
        test_freq, source_strength, dx, src_pos,
        save_dir='pressure_slices'
    )

    print(f"\nAll images saved to 'pressure_slices' directory")

    return mae_improvement


def main():
    print("=" * 70)
    print("Full Field ML-Enhanced FDTD Solver (Ricker Wavelet)")
    print("=" * 70)


    ml_model, target_mean, target_std = train_model()
    improvement = comparison_experiment(ml_model, target_mean, target_std)

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {improvement:.1f}% Improvement")
    print("=" * 70)


if __name__ == "__main__":
    main()
