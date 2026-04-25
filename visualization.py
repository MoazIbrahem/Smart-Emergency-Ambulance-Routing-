import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. LEARNING CURVE

def plot_learning_curve(rewards: list, algo_name: str, window: int = 30, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards, alpha=0.3, color="steelblue", label=f"{algo_name} (raw)")

    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(smooth, color="crimson", linewidth=2,
                label=f"Moving Average ({window})")

    ax.set_title(f"{algo_name} — Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"{algo_name.lower().replace(' ', '_')}_learning_curve.png")
        fig.savefig(path, dpi=120)
        print(f"[viz] Saved: {path}")

    plt.show()
    plt.close(fig)


# 2. Q-VALUE / VALUE HEATMAP

def plot_value_heatmap(value_grid: np.ndarray, algo_name: str,
                       title_suffix: str = "Max Q-Value",
                       cmap: str = "RdYlGn",
                       env=None,
                       save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(value_grid, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label=title_suffix)

    # Optional: overlay key positions if the env exposes them
    if env is not None:
        _overlay_env_markers(ax, env)

    ax.set_title(f"{algo_name} — {title_suffix} Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR,
                            f"{algo_name.lower().replace(' ', '_')}_heatmap.png")
        fig.savefig(path, dpi=120)
        print(f"[viz] Saved: {path}")

    plt.show()
    plt.close(fig)


# 3. POLICY VISUALIZATION (arrow map)

# Action → (Δrow, Δcol) — adjust if your env uses different action encoding
_ACTION_ARROWS = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}   # R L U D
_ACTION_LABELS = {0: "→", 1: "←", 2: "↑", 3: "↓"}


def plot_policy(policy_grid: np.ndarray, algo_name: str,
                env=None, save: bool = True):
    rows, cols = policy_grid.shape
    fig, ax = plt.subplots(figsize=(max(6, cols * 0.5), max(6, rows * 0.5)))

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(rows):
        for c in range(cols):
            action = int(policy_grid[r, c])
            symbol = _ACTION_LABELS.get(action, "?")
            ax.text(c, r, symbol, ha="center", va="center",
                    fontsize=max(6, min(12, 120 // cols)),
                    color="navy")

    if env is not None:
        _overlay_env_markers(ax, env)

    ax.set_title(f"{algo_name} — Learned Policy")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR,
                            f"{algo_name.lower().replace(' ', '_')}_policy.png")
        fig.savefig(path, dpi=120)
        print(f"[viz] Saved: {path}")

    plt.show()
    plt.close(fig)


# 4. STABILITY PLOT  (variance across multiple runs)
def plot_stability(all_rewards: list[list], algo_name: str,
                   window: int = 30, save: bool = True):
    if not all_rewards:
        print("[viz] No data provided for stability plot.")
        return

    # Smooth each run then stack
    smoothed = []
    for run in all_rewards:
        if len(run) >= window:
            s = np.convolve(run, np.ones(window) / window, mode="valid")
        else:
            s = np.array(run, dtype=float)
        smoothed.append(s)

    min_len = min(len(s) for s in smoothed)
    arr = np.array([s[:min_len] for s in smoothed])   # shape: (n_runs, episodes)

    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    xs   = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, mean, color="steelblue", linewidth=2, label="Mean reward")
    ax.fill_between(xs, mean - std, mean + std, alpha=0.25,
                    color="steelblue", label="±1 std")

    for i, run in enumerate(smoothed):
        ax.plot(run[:min_len], alpha=0.15, linewidth=0.8, color="grey")

    ax.set_title(f"{algo_name} — Stability ({len(all_rewards)} runs)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Smoothed Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR,
                            f"{algo_name.lower().replace(' ', '_')}_stability.png")
        fig.savefig(path, dpi=120)
        print(f"[viz] Saved: {path}")

    plt.show()
    plt.close(fig)



# 5. MASTER ENTRY-POINT  ←  call this from your algorithm file


def run_visualizations(
    algo_name: str,
    rewards: list,
    q_table_or_values: np.ndarray = None,
    env=None,
    all_rewards: list[list] = None,
    grid_size: int = None,
    window: int = 30,
    save: bool = True,
):
    
    print(f"\n{'='*50}")
    print(f"  Generating visualizations for: {algo_name}")
    print(f"{'='*50}\n")

    # ── 1. Learning curve ────────────────────────────────────────────────────
    plot_learning_curve(rewards, algo_name, window=window, save=save)

    # ── 2. Value / Q-value heatmap ───────────────────────────────────────────
    if q_table_or_values is not None:
        size = grid_size or (env.size if env is not None else None)

        arr = np.array(q_table_or_values)

        if arr.ndim == 2 and arr.shape[1] > 1:
            # Flat Q-table  (n_states, n_actions)  → max Q per state → grid
            if size is None:
                size = int(np.sqrt(arr.shape[0]))
            value_grid = arr.max(axis=1).reshape(size, size)
            title_sfx  = "Max Q-Value"

            # ── 3. Policy map (only makes sense for Q-tables) ─────────────
            policy_flat = arr.argmax(axis=1).reshape(size, size)
            plot_policy(policy_flat, algo_name, env=env, save=save)

        else:
            # Already a 2-D value array
            value_grid = arr if arr.ndim == 2 else arr.reshape(size, size)
            title_sfx  = "State Value V(s)"

        plot_value_heatmap(value_grid, algo_name,
                           title_suffix=title_sfx, env=env, save=save)

    # ── 4. Stability plot ────────────────────────────────────────────────────
    if all_rewards is not None:
        plot_stability(all_rewards, algo_name, window=window, save=save)

    print(f"\n[viz] All plots saved to ./{OUTPUT_DIR}/\n")

# Internal helper

def _overlay_env_markers(ax, env):
    """Overlay ambulance start and hospital position if the env exposes them."""
    markers = []

    # Ambulance / agent start
    if hasattr(env, "ambulance_pos"):
        r, c = env.ambulance_pos
        ax.plot(c, r, marker="*", markersize=14, color="gold",
                markeredgecolor="black", zorder=5)
        markers.append(mpatches.Patch(color="gold", label="Ambulance"))

    # Hospital / goal
    if hasattr(env, "hospital_pos"):
        r, c = env.hospital_pos
        ax.plot(c, r, marker="H", markersize=14, color="red",
                markeredgecolor="black", zorder=5)
        markers.append(mpatches.Patch(color="red", label="Hospital"))

    if markers:
        ax.legend(handles=markers, loc="upper right", fontsize=8)
