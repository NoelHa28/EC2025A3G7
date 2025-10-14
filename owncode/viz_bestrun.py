"""Assignment 3 – visualize path from a .npy file (no robot object)."""

# --- your template imports ---
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer

# --- DATA SETUP (unchanged) ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


def show_xpos_history(
    history: list[float],
    spawn_position: list[float],
    target_position: list[float],
    *,
    save: bool = True,
    show: bool = True,
) -> None:
    # (YOUR FUNCTION BODY UNCHANGED)
    mj.set_mjcb_control(None)
    world = OlympicArena(load_precompiled=False)

    start_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="green_sphere" size=".1" rgba="0 1 0 1"/>
        </worldbody>
    </mujoco>
    """
    end_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="red_sphere" size=".1" rgba="1 0 0 1"/>
        </worldbody>
    </mujoco>
    """
    target_box = r"""
    <mujoco>
        <worldbody>
            <geom name="magenta_box" size=".1 .1 .1" type="box" rgba="1 0 1 0.75"/>
        </worldbody>
    </mujoco>
    """
    spawn_box = r"""
    <mujoco>
        <worldbody>
            <geom name="gray_box" size=".1 .1 .1" type="box" rgba="0.5 0.5 0.5 0.5"/>
        </worldbody>
    </mujoco>
    """

    pos_data = np.array(history)

    adjustment = np.array((0, 0, target_position[2] + 1))
    world.spawn(
        mj.MjSpec.from_string(start_sphere),
        position=pos_data[0] + (adjustment * 1.5),
        correct_collision_with_floor=False,
    )

    world.spawn(
        mj.MjSpec.from_string(end_sphere),
        position=pos_data[-1] + (adjustment * 2),
        correct_collision_with_floor=False,
    )

    world.spawn(
        mj.MjSpec.from_string(target_box),
        position=np.array(target_position) + adjustment,
        correct_collision_with_floor=False,
    )

    world.spawn(
        mj.MjSpec.from_string(spawn_box),
        position=np.array(spawn_position),
        correct_collision_with_floor=False,
    )

    smooth = np.linspace(0, 1, len(pos_data))
    inv_smooth = 1 - smooth
    smooth_rise = np.linspace(1.25, 1.95, len(pos_data))
    for i in range(1, len(pos_data)):
        pos_i = pos_data[i]
        pos_j = pos_data[i - 1]

        distance = pos_i - pos_j
        minimum_size = 0.05
        geom_size = np.array(
            [
                max(abs(distance[0]) / 2, minimum_size),
                max(abs(distance[1]) / 2, minimum_size),
                max(abs(distance[2]) / 2, minimum_size),
            ]
        )
        geom_size_str: str = f"{geom_size[0]} {geom_size[1]} {geom_size[2]}"

        half_way_point = (pos_i + pos_j) / 2
        geom_pos_str = f"{half_way_point[0]} {half_way_point[1]} {half_way_point[2]}"

        geom_rgba = f"{smooth[i]} {inv_smooth[i]} 0 0.75"
        path_box = rf"""
        <mujoco>
            <worldbody>
                <geom name="yellow_sphere"
                      type="box"
                      pos="{geom_pos_str}"
                      size="{geom_size_str}"
                      rgba="{geom_rgba}"/>
            </worldbody>
        </mujoco>
        """
        world.spawn(
            mj.MjSpec.from_string(path_box),
            position=(adjustment * smooth_rise[i]),
            correct_collision_with_floor=False,
        )

    _, ax = plt.subplots()
    plt.rc("legend", fontsize="small")
    red_patch = mpatches.Patch(color="red", label="End Position")
    gray_patch = mpatches.Patch(color="gray", label="Spawn Position")
    green_patch = mpatches.Patch(color="green", label="Start Position")
    magenta_patch = mpatches.Patch(color="magenta", label="Target Position")
    yellow_patch = mpatches.Patch(color="yellow", label="Robot Path")
    ax.legend(
        handles=[green_patch, red_patch, magenta_patch, gray_patch, yellow_patch],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )

    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.title("Robot Path in XY Plane")

    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
        width=200,
        height=600,
        cam_fovy=8,
        cam_pos=[2.1, 0, 50],
        cam_quat=[-0.7071, 0, 0, 0.7071],
    )

    img = plt.imread(save_path)
    ax.imshow(img)

    if save:
        fig_path = DATA / "robot_path.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()


# ------------------------------
# Only NEW code below this line:
# ------------------------------


def load_positions_from_npy(path: str | Path) -> np.ndarray:
    """Load (N,3) positions from a .npy file and return as a float array."""
    arr = np.load(str(path), allow_pickle=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        # Allow a flat vector that should be multiples of 3
        if arr.size % 3 != 0:
            raise ValueError(f"Flat array of size {arr.size} is not divisible by 3.")
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (N,3) positions, got shape {arr.shape}.")
    return arr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize a path stored in a .npy file."
    )
    parser.add_argument(
        "--npy",
        type=str,
        default="owncode/hei.npy",
        help="Path to the .npy file containing (N,3) positions.",
    )
    parser.add_argument(
        "--spawn",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional explicit spawn position. Defaults to the first row in the file.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional explicit target position. Defaults to the last row in the file.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the rendered figure."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the figure window."
    )
    args = parser.parse_args()

    positions = load_positions_from_npy(args.npy)

    # Use data-only: spawn = first sample, target = last sample (unless overridden)
    spawn_position = [-0.8, 0, 0.1]
    target_position = [5, 0, 0.5]

    show_xpos_history(
        history=positions.tolist(),
        spawn_position=spawn_position,
        target_position=target_position,
        save=not args.no_save,
        show=not args.no_show,
    )
