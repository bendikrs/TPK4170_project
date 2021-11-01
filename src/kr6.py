import numpy as np
import modern_robotics as mr

if __name__ == "__main__":
    from dh import DHLink
else:
    from lib.dh import DHLink

PI_HALF = np.pi / 2
PI = np.pi

class KR6:


    def __init__(self, theta = np.array([0,0,0,0,0,0])):

        self.d1, self.d2, self.d3, self.d4 = 0.400, 0.455, 0.420, 0.080 #distansene mellom jointsene som ikke er 0
        self.a1, self.a2 = 0.025, 0.035 #offsetsene 

        self._M = np.array(
            [
                [0,  0,  1, self.a1+self.d2+self.d3+self.d4],
                [0, -1,  0,                               0],
                [1,  0,  0,                 self.a2+self.d1],
                [0,  0,  0,                               1]
            ]
        )

        self._Slist = np.array(
            [
                [0, 0,-1,                  0,                 0,                       0],
                [0, 1, 0,           -self.d1,                 0,                 self.a1],
                [0, 1, 0,           -self.d1,                 0,         self.a1+self.d2],
                [-1,0, 0,                  0,-(self.a2+self.d1),                       0],
                [0, 1, 0, -(self.d1+self.a2),                 0, self.a1+self.d2+self.d3],
                [-1,0, 0,                  0,-(self.a2+self.d1),                       0]
            ]
        ).T

        self.theta = np.atleast_2d(theta).T
        
        self.thetaDH0 = np.array([ [0],     # from space frame to joint 1
                                [0],        # from joint 1 to joint 2
                                [0],        # from joint 2 to joint 3
                                [-PI_HALF], # from joint 3 to joint 4
                                [0],        # from joint 4 to joint 5
                                [0],        # from joint 5 to joint 6
                                [0]])       # from joint 6 to body frame
        
        for i in range(1, len(self.thetaDH0)):
            self.thetaDH0[i] += self.theta[i-1]

        self.alpha = np.array([[PI],
                        [PI_HALF],
                        [0],
                        [PI_HALF],
                        [-PI_HALF],
                        [PI_HALF],
                        [PI]])

        self.d = np.array([[0],
                    [-self.d1],
                    [0],
                    [0],
                    [-self.d3],
                    [0],
                    [-self.d4]])

        self.a = np.array([[0],
                    [self.a1],
                    [self.d2],
                    [self.a2],
                    [0],
                    [0],
                    [0]])

        self.dhList = np.concatenate((self.a, self.alpha, self.d, self.thetaDH0), axis=1)

        self._Blist = mr.Adjoint(np.linalg.inv(self._M)) @ self._Slist

        self.linkList = [] # Contains all the DHlinks for the robot
        for i in range(len(self.dhList)):
            self.linkList.append(DHLink(*self.dhList[i]))

        self.fk_dh_zero = self.linkList[0].matrix() # DH zero configuration
        self.Tlist = [self.linkList[0].matrix()]    # Contains all the transformation matrices

        for i in range(1, len(self.dhList)):
            self.fk_dh_zero = self.fk_dh_zero @ self.linkList[i].matrix()
            self.Tlist.append(self.Tlist[i-1] @ self.linkList[i].matrix())
            

        self.T_sb_poe = mr.FKinSpace(self._M, self._Slist, self.theta)


