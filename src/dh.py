from dataclasses import dataclass
from transformations import quaternion_from_matrix
from tpk4170.models import Axes
import numpy as np

# This class is borrowd from the course resources

@dataclass
class DHLink:
    a: float
    alpha: float
    d: float
    theta: float

    def matrix(self) -> np.array:
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, self.a * ct],
                [st, ct * ca, -ct * sa, self.a * st],
                [0.0, sa, ca, self.d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

class DHFrame(Axes):
    def __init__(self, trf):
        Axes.__init__(self, 0.1)
        self.quaternion = np.roll(quaternion_from_matrix(trf), -1).tolist()
        self.position = (trf[:3, 3]).tolist()