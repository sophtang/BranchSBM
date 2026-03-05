import numpy as np
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import umap
import scanpy as sc
from sklearn.decomposition import PCA

import ot as pot
from tqdm import tqdm
from functools import partial
from typing import Optional

from matplotlib.colors import LinearSegmentedColormap


def set_seed(seed):
    """
    Sets the seed for reproducibility in PyTorch, Numpy, and Python's Random.

    Parameters:
    seed (int): The seed for the random number generators.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # CPU and GPU (deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # CUDA
        torch.cuda.manual_seed_all(seed)  # all GPU devices
        torch.backends.cudnn.deterministic = True  # CuDNN behavior
        torch.backends.cudnn.benchmark = False


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY**2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0)
        )
    )
    return mmd2, var_est


def plot_lidar(ax, dataset, xs=None, S=25, branch_idx=None):
    # Combine the dataset and trajectory points for sorting
    combined_points = []
    combined_colors = []
    combined_sizes = []
    
    
    custom_colors_1 = ["#05009E", "#A19EFF", "#50B2D7"]
    custom_colors_2 = ["#05009E", "#A19EFF", "#D577FF"]
    
    custom_cmap_1 = LinearSegmentedColormap.from_list("my_cmap", custom_colors_1)
    custom_cmap_2 = LinearSegmentedColormap.from_list("my_cmap", custom_colors_2)

    # Normalize the z-coordinates for alpha scaling
    z_coords = (
        dataset[:, 2].numpy() if torch.is_tensor(dataset[:, 2]) else dataset[:, 2]
    )
    z_min, z_max = z_coords.min(), z_coords.max()
    z_norm = (z_coords - z_min) / (z_max - z_min)

    # Add surface points with a lower z-order
    for i, point in enumerate(dataset):
        grey_value = 0.95 - 0.7 * z_norm[i]
        combined_points.append(point.numpy())
        combined_colors.append(
            (
                grey_value,
                grey_value,
                grey_value,
                1.0
            )
        )  # Grey color with transparency
        combined_sizes.append(0.1)

    # Add trajectory points with a higher z-order
    if xs is not None:
        if branch_idx == 0:
            cmap = custom_cmap_1
        else:
            cmap = custom_cmap_2
            
        B, T, D = xs.shape
        steps_to_log = np.linspace(0, T - 1, S).astype(int)
        xs = xs.cpu().detach().clone()
        for idx, step in enumerate(steps_to_log):
            for point in xs[:512, step]:
                combined_points.append(
                    point.numpy() if torch.is_tensor(point) else point
                )
                combined_colors.append(cmap(idx / (len(steps_to_log) - 1)))
                combined_sizes.append(0.8)

    # Convert to numpy array for easier manipulation
    combined_points = np.array(combined_points)
    combined_colors = np.array(combined_colors)
    combined_sizes = np.array(combined_sizes)

    # Sort by z-coordinate (depth)
    sorted_indices = np.argsort(combined_points[:, 2])
    combined_points = combined_points[sorted_indices]
    combined_colors = combined_colors[sorted_indices]
    combined_sizes = combined_sizes[sorted_indices]

    # Plot the sorted points
    ax.scatter(
        combined_points[:, 0],
        combined_points[:, 1],
        combined_points[:, 2],
        s=combined_sizes,
        c=combined_colors,
        depthshade=True,
    )

    ax.set_xlim3d(left=-4.8, right=4.8)
    ax.set_ylim3d(bottom=-4.8, top=4.8)
    ax.set_zlim3d(bottom=0.0, top=2.0)
    ax.set_zticks([0, 1.0, 2.0])
    ax.grid(False)
    plt.axis("off")

    return ax


def plot_images_trajectory(trajectories, vae, processor, num_steps):

    # Compute trajectories for each image
    t_span = torch.linspace(0, trajectories.shape[1] - 1, num_steps)
    t_span = [int(t) for t in t_span]
    num_images = trajectories.shape[0]

    # Decode images at each step in each trajectory
    decoded_images = [
        [
            processor.postprocess(
                vae.decode(
                    trajectories[i_image, traj_step].unsqueeze(0)
                ).sample.detach()
            )[0]
            for traj_step in t_span
        ]
        for i_image in range(num_images)
    ]

    # Plotting
    fig, axes = plt.subplots(
        num_images, num_steps, figsize=(num_steps * 2, num_images * 2)
    )
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable
    for img_idx, img_traj in enumerate(decoded_images):
        for step_idx, img in enumerate(img_traj):
            ax = axes[img_idx][step_idx] if num_images > 1 else axes[step_idx]
            if (
                isinstance(img, np.ndarray) and img.shape[0] == 3
            ):  # Assuming 3 channels (RGB)
                img = img.transpose(1, 2, 0)
            ax.imshow(img)
            ax.axis("off")
            if img_idx == 0:
                ax.set_title(f"t={t_span[step_idx]/t_span[-1]:.2f}")
    plt.tight_layout()
    return fig


def plot_growth(dataset, growth_nets, xs, output_file='plot.pdf'):
    x0s = [dataset["x0"][0]]
    w0s = [dataset["x0"][1]]
    x1s_list = [[dataset["x1_1"][0]], [dataset["x1_2"][0]]]
    w1s_list = [[dataset["x1_1"][1]], [dataset["x1_2"][1]]] 
    
    
    
    plt.show()