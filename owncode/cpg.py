import numpy as np

class CPG:
    """
    Randomly initialized CPG (Central Pattern Generator) for locomotion.
    Each joint can have a different frequency and phase offset.
    """
    def __init__(
        self, 
        n_joints: int, 
        dt: float = 0.1,
        omega_range: tuple[float, float] = (100.0, 300.0)  # Min and max frequency (rad/s)
    ):
        self.n_joints = n_joints
        self.dt = dt
        # Random frequencies for each joint within the given range
        self.omega = np.random.uniform(omega_range[0], omega_range[1], size=n_joints)
        # Random initial phases
        self.phase = np.random.uniform(0, 2*np.pi, size=n_joints)
        # Random phase offsets for coordination between joints
        self.offset = np.random.uniform(0, 2*np.pi, size=n_joints)
        # Fixed amplitude between -pi/2 and pi/2
        self.amplitude = np.pi / 2

    def step(self) -> np.ndarray:
        """
        Advance the CPG by one timestep and return joint angles.
        """
        self.phase += self.omega * self.dt
        self.phase = np.mod(self.phase, 2*np.pi)
        angles = self.amplitude * np.sin(self.phase + self.offset)
        return angles