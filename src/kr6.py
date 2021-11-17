import numpy as np
import modern_robotics as mr
from tpk4170.visualization import Viewer
from src.dh import DHFrame
import src.tools as tools

if __name__ == "__main__":
    from dh import DHLink
else:
    from src.dh import DHLink

# Constants
PI_HALF = np.pi / 2
PI = np.pi

class KR6:
    '''Robot class for the  KUKA KR6 R900 sixx (Agilus) robot
    '''
    def __init__(self, theta = np.array([0,0,0,0,0,0]), thetalist0 = [0,0,0,0,0,0], eomg=0.01, ev=0.01):
        ''' Constructor for the KR6 robot class.
        :param theta: 6-element list of desired joint angles
        :param thetalist0: 6-element list of initial joint angles, used for the numerical IK
        :param eomg: Maximum allowable error in the end-effector orientation
        :param ev: Maximum allowable end-effector linear position error
        '''
        # Defining the physical parameters of the robot
        self.d1, self.d2, self.d3, self.d4 = 0.400, 0.455, 0.420, 0.080 
        self.a1, self.a2 = 0.025, 0.035

        # Defining the M and S matrices
        self._M = np.array(
            [
                [0,  0,  1, self.a1+self.d2+self.d3+self.d4],
                [0, -1,  0,                               0],
                [1,  0,  0,                 self.a2+self.d1],
                [0,  0,  0,                               1]
            ])
        self._Slist = np.array(
            [
                [0, 0,-1,                  0,                 0,                       0],
                [0, 1, 0,           -self.d1,                 0,                 self.a1],
                [0, 1, 0,           -self.d1,                 0,         self.a1+self.d2],
                [-1,0, 0,                  0,-(self.a2+self.d1),                       0],
                [0, 1, 0, -(self.d1+self.a2),                 0, self.a1+self.d2+self.d3],
                [-1,0, 0,                  0,-(self.a2+self.d1),                       0]
            ]).T
        
        # self._Slist2 = np.array([[0, 0, 0, 1, 0, 1],
        #                          [0, 1, 1, 0, 1, 0],
        #                          [1, 0, 0, 0, 0, 0],
        #                          [0,-self.d1,-self.d2, 0, 0, 0],
        #                          [0, 0, 0, -self.a2, 0, 0],
        #                          [0, self.a1, 0, 0, -self.d3, 0]])

        # Transposing the theta list to a column vector
        self.theta = np.atleast_2d(theta).T
        
        # Defining the Denavit-Hartenberg  parameters
        # Joint angle
        self.thetaDH0 = np.array([ [0],     # from space frame to joint 1
                                [0],        # from joint 1 to joint 2
                                [0],        # from joint 2 to joint 3
                                [-PI_HALF], # from joint 3 to joint 4
                                [0],        # from joint 4 to joint 5
                                [0],        # from joint 5 to joint 6
                                [0]])       # from joint 6 to body frame
        # Add the input values to the joint angle list
        for i in range(1, len(self.thetaDH0)): 
            self.thetaDH0[i] += self.theta[i-1]
        # Twist angles
        self.alpha = np.array([[PI],
                        [PI_HALF],
                        [0],
                        [PI_HALF],
                        [-PI_HALF],
                        [PI_HALF],
                        [PI]])
        # Link offsets
        self.d = np.array([[0],
                    [-self.d1],
                    [0],
                    [0],
                    [-self.d3],
                    [0],
                    [-self.d4]])
        # Link lengths
        self.a = np.array([[0],
                    [self.a1],
                    [self.d2],
                    [self.a2],
                    [0],
                    [0],
                    [0]])

        # Merge the DH parameters into a single matrix
        self.dhList = np.concatenate((self.a, self.alpha, self.d, self.thetaDH0), axis=1)

        # Calculate the Body fixed screw axis matrix
        self._Blist = mr.Adjoint(np.linalg.inv(self._M)) @ self._Slist

        # Making a list of all the DH links
        self.linkList = [] # Contains all the DHlinks for the robot
        for i in range(len(self.dhList)):
            self.linkList.append(DHLink(*self.dhList[i]))

        self.Tsb_dh = self.linkList[0].matrix()     # Tsb using the DH convention
        self.Tlist = [self.linkList[0].matrix()]    # Contains all the transformation matrices, mostly used for visualization
        for i in range(1, len(self.dhList)):
            self.Tsb_dh = self.Tsb_dh @ self.linkList[i].matrix() # T06
            self.Tlist.append(self.Tlist[i-1] @ self.linkList[i].matrix()) # [T01, T02, T03, T04, T05, T06]
            
        # Tsb using the poe convention
        self.Tsb_poe = mr.FKinSpace(self._M, self._Slist, self.theta)

        # Numerical Inverse Kinematics
        self.IK = tools.IKinSpace_our(self._Slist, self._M, self.Tsb_poe, thetalist0, eomg, ev)
        self.Tsb_IK = self.IK[2][-1] # Tsb using the numerical IK

        # Calculating the Space Jacobian
        self.Js = mr.JacobianSpace(self._Slist, self.theta)

        # Analytical Inverse Kinematics
        self.aIK_thetaLists = tools.inverseKinematicsTheta123(self.Tsb_poe) # IK for the wrist position, this works
        self.Tsw = tools.makeT_SW(self.Tsb_poe) # Fetching the Tsw matrix

        # Analytical Inverse Kinematics position and orientation
        # Complete solution for the Inverse Kinematics, This currently does not work for all configurations
        if theta.all() != np.array([0,0,0,0,0,0]).all():
            self.aIK_thetaLists_Tsb = tools.inverseKinematicsTheta456(self.aIK_thetaLists, self.Tsb_poe, self._Slist, self._M) 


    def add_FK_to_viewer(self, viewer: Viewer): # Adding the DH frames to the viewer
        for i in range(len(self.Tlist)):
            viewer.add(DHFrame(self.Tlist[i]))

    def add_aIK_to_viewer(self, viewer: Viewer): # Adding the analytical IK solution to the viewer
        kr6_temp = KR6(self.aIK_thetaLists_Tsb)
        kr6_temp.add_FK_to_viewer(viewer)

    def add_nIK_to_viewer(self, viewer: Viewer): # Adding the numerical IK solution to the viewer
        kr6_temp = KR6(self.IK[0])
        kr6_temp.add_FK_to_viewer(viewer)

