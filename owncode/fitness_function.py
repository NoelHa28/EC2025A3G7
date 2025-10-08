import numpy as np

TARGET_POSITION = [5, 0, 0.5]

# def fitness(history: list[tuple[float, float, float]]) -> float:
#     xt, yt, zt = TARGET_POSITION
#     xc, yc, zc = history[-1] # type: ignore

#     # Minimize the distance --> maximize the negative distance
#     cartesian_distance = np.sqrt(
#         (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
#     )
#     return -cartesian_distance

def fitness(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    x3, y3, z3 = history[3]
    x_end, y_end, z_end = history[-1]

    start_to_target = np.sqrt((xt - x3)**2 + (yt - y3)**2 + (zt - z3)**2)
    end_to_target   = np.sqrt((xt - x_end)**2 + (yt - y_end)**2 + (zt - z_end)**2)

    # Positive if we got closer to the target
    progress = start_to_target - end_to_target
    return progress
