import numpy as np
import matplotlib.pyplot as plt

def visual_comparison(X1, X2, titles=None, lims=(-3,3), limit=1000):
    if titles is None:
        titles = ["Pierwszy zbiór danych", "Drugi zbiór danych"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(X1[:limit, 0], X1[:limit, 1], s=7)
    plt.xlim(*lims)
    plt.ylim(*lims)
    plt.title(titles[0])

    plt.subplot(1, 3, 2)
    plt.scatter(X2[:limit, 0], X2[:limit, 1], color='orange', s=7)
    plt.xlim(*lims)
    plt.ylim(*lims)
    plt.title(titles[1])

    plt.subplot(1, 3, 3)
    plt.scatter(X1[:limit, 0], X1[:limit, 1], label=titles[0], s=7)
    plt.scatter(X2[:limit, 0], X2[:limit, 1], label=titles[1], color='orange', s=7)
    plt.xlim(*lims)
    plt.ylim(*lims)
    plt.title("Porównanie")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_loss_curve(loss_logs):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_logs)
    plt.title("Funkcja straty")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji straty")
    plt.grid()
    plt.show()

def plot_trajectories(ds, trajectories, limit=1000):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(ds[:limit, 0], ds[:limit, 1])
    plt.plot(trajectories[:, 0, 0], trajectories[:, 0, 1], color='black')
    plt.title("Jedna Trajektoria")

    plt.subplot(1, 2, 2)
    plt.scatter(ds[:limit, 0], ds[:limit, 1])
    plt.plot(trajectories[:, 1:11, 0], trajectories[:, 1:11, 1], color='black')
    plt.title("Dziesięć Trajektorii")

    plt.show()

def show_images(generated_ddpm, generated_ddim, n_col=16, n_row=3):
    generated_ddpm = (generated_ddpm + 1) / 2  # Normalize to range (0,1)
    generated_ddim = (generated_ddim + 1) / 2  # Normalize to range (0,1)

    generated_ddpm = generated_ddpm.clip(0, 1)
    generated_ddim = generated_ddim.clip(0, 1)

    def concatenate_images(images):
        rows = [np.concatenate(images[i * n_col:(i + 1) * n_col], axis=2) for i in range(n_row)]
        return np.concatenate(rows, axis=1)

    ddpm_concat = concatenate_images(generated_ddpm)
    ddim_concat = concatenate_images(generated_ddim)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6))

    axes[0].imshow(np.transpose(ddpm_concat, (1, 2, 0)))
    axes[0].axis('off')
    axes[0].set_title("Zdjęcia wygenerowane przez DDPM", fontsize=12, fontweight='bold')

    axes[1].imshow(np.transpose(ddim_concat, (1, 2, 0)))
    axes[1].axis('off')
    axes[1].set_title("Zdjęcia wygenerowane przez DDIM", fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    return fig

def build_trajectory(imgs):
    timesteps = np.linspace(0, imgs.shape[0] - 1, 10, endpoint=True).astype(np.int32)

    columns = []
    for t in timesteps:
        col = np.concatenate([imgs[t, i, ...] for i in range(3)], axis=1)
        columns.append(col)

    columns = np.concatenate(columns, axis=2)

    columns = (columns + 1.0) / 2.0
    columns = columns.clip(0, 1)
    columns = np.transpose(columns, (1, 2, 0))

    return columns

def plot_image_trajectories(traj_ddpm, traj_ddim):
    ddpm_img = build_trajectory(traj_ddpm)
    ddim_img = build_trajectory(traj_ddim)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    print(ddpm_img.shape)
    
    axes[0].imshow(ddpm_img)
    axes[0].axis('off')
    axes[0].set_title("Trajektorie DDPM", fontsize=12, fontweight='bold')
    
    axes[1].imshow(ddim_img)
    axes[1].axis('off')
    axes[1].set_title("Trajektorie DDIM", fontsize=12, fontweight='bold')

    plt.tight_layout()

    return fig