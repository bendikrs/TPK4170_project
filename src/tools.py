import sympy as sp
from modern_robotics import FKinSpace, Adjoint, se3ToVec, MatrixLog6, TransInv, JacobianSpace, MatrixExp6, VecTose3
import numpy as np


def rotx(theta): # Rotation about x-axis
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return R


def roty(theta): # Rotation about y-axis
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return R


def rotz(theta): # Rotation about z-axis
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    return R

def sprotx(theta): # Sympy version of rotx
    ct = sp.cos(theta)
    st = sp.sin(theta)
    R = sp.Matrix([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return R


def sproty(theta): # Sympy version of roty
    ct = sp.cos(theta)
    st = sp.sin(theta)
    R = sp.Matrix([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
    return R


def sprotz(theta): # Sympy version of rotz
    ct = sp.cos(theta)
    st = sp.sin(theta)
    R = sp.Matrix([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    return R


def vector(x,y,z): # Vector from x, y, z position
    return np.array([[x], [y], [z]])


def exp6(twist, theta): 
    omega = skew(twist[:3])
    v = sp.Matrix(twist[3:])
    T = sp.eye(4)
    T[:3, :3] = exp3(twist[:3], theta)
    T[:3, 3] = (sp.eye(3) * theta + (1 - sp.cos(theta)) * omega +
                (theta - sp.sin(theta)) * omega * omega) * v
    return T


def skew(v):
    return sp.Matrix([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])


def exp3(omega, theta):
    omega = skew(omega)
    R = sp.eye(3) + sp.sin(theta) * omega + (1 - sp.cos(theta)) * omega * omega
    return R


def Ad(T):
    '''calculate the Adjoint matrix of a transformation matrix
    input: T: transformation matrix
    output: Ad: Adjoint matrix
    '''
    AdT = sp.zeros(6)
    R = sp.Matrix(T[:3, :3])
    AdT[:3, :3] = R
    AdT[3:, 3:] = R
    AdT[3:, :3] = skew(T[:3, 3]) * R
    return AdT


def IKinSpace_our(Slist, M, T, thetalist0, eomg, ev):
    """This function borrowed from the IKinSpace() function from the Modern Robotics library, and then modified.
        The function does now return a list of all the Tsb's, instead of just the last one.
        It also returns i, which is the number of iterations it took to find the solution.
    
    Computes inverse kinematics in the space frame for an open chain robot
    """

    Tsb = []
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb.append(FKinSpace(M,Slist, thetalist))
    Vs = np.dot(Adjoint(Tsb[0]), \
                se3ToVec(MatrixLog6(np.dot(TransInv(Tsb[0]), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
          or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(JacobianSpace(Slist, \
                                                          thetalist)), Vs)
        i = i + 1
        Tsb.append(FKinSpace(M, Slist, thetalist))
        Vs = np.dot(Adjoint(Tsb[-1]), \
                    se3ToVec(MatrixLog6(np.dot(TransInv(Tsb[-1]), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err, Tsb, i)


def pointFromT(T=np.array) -> np.array:
    '''Function to extract the position from a T matrix.
    '''
    P = np.zeros(3)
    for i in range(len(T)):
        P[i:] = T[i][3]
    return P



def makeT_SW(T_SB=np.array) -> np.array:
    '''Function to extract the transformation matrix to the wrist position from the T_sb matrix.
    input: T_SB: transformation matrix from space frame to end-effector frame
    output: T_SW: transformation matrix from space frame to the wrist position
    '''
    T_BW = np.array([[0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, -0.080],
                     [0, 0, 0, 1]])
    return np.dot(T_SB, T_BW)


def inverseKinematicsTheta123(T_SB:np.array) -> np.array: #returnerer liste med 4 alternative konfig for theta 1,2 og 3
    '''Outputs four lists of thetas giving different configurations of joint 1,2 and 3 that provide the same wrist position
    return np.array(4x3)
    input: T_SB: transformation matrix from space frame to end-effector frame
    output: np.array(3x4) four configurations of theta 1,2 and 3
    '''
    r2, r3, r4, a1, a2 = 0.400, 0.455, np.sqrt(0.420**2+0.035**2), 0.025, 0.035 # the distances of the robot
    T_SW = makeT_SW(T_SB) #creates the T-matrix from space to wrist
    P_W = pointFromT(T_SW) #extracts the origin point of the wrist frame in space coordinates



    theta1_i = np.arctan2(-P_W[1], P_W[0])
    theta1_ii = np.arctan2(P_W[1], -P_W[0])

    #forward

    Px, Py, Pz = P_W[0] - np.cos(theta1_i) * a1, \
                 P_W[1] + np.sin(theta1_i) * a1, \
                 P_W[2] - r2,

    c3 = (Px**2 + Py**2 + Pz**2 - r3**2 - r4**2) / (2 * r3 * r4)

    s3_pos = np.sqrt(1-c3**2)
    s3_neg = -np.sqrt(1-c3**2)

    theta3_i = np.arctan2(s3_neg, c3) #under elbow
    theta3_ii = np.arctan2(s3_pos, c3) #over elbow

    c2_poss3 = (np.sqrt(Px**2 + Py**2) * (r3+r4*c3) + Pz * r4 * s3_pos) / (r3**2+r4**2+2*r3*r4*c3)
    c2_negs3 = (np.sqrt(Px**2 + Py**2) * (r3+r4*c3) + Pz * r4 * s3_neg) / (r3**2+r4**2+2*r3*r4*c3)
     
    s2_minus_poss3 = (Pz * (r3 + r4*c3) - np.sqrt(Px**2 + Py**2) * r4*s3_pos) / (r3**2 + r4**2 + 2 * r3 * r4 * c3)
    s2_minus_negs3 = (Pz * (r3 + r4*c3) - np.sqrt(Px**2 + Py**2) * r4*s3_neg) / (r3**2 + r4**2 + 2 * r3 * r4 * c3)
    s2_pluss_poss3 = (Pz * (r3 + r4*c3) + np.sqrt(Px**2 + Py**2) * r4*s3_pos) / (r3**2 + r4**2 + 2 * r3 * r4 * c3)
    s2_pluss_negs3 = (Pz * (r3 + r4*c3) + np.sqrt(Px**2 + Py**2) * r4*s3_neg) / (r3**2 + r4**2 + 2 * r3 * r4 * c3)
    
    theta2_i   = np.arctan2(-s2_minus_poss3, c2_poss3)
    theta2_iii = np.arctan2(-s2_minus_negs3, c2_negs3)
     
    # backward

    Pz, Px, Py = P_W[2] - r2, \
                 P_W[0] - np.cos(theta1_ii) * a1, \
                 P_W[1] + np.sin(theta1_ii) * a1

    c3 = (Pz**2 + Px**2 + Py**2 - r3**2 - r4**2) / (2 * r3 * r4)

    s3_pos = np.sqrt(1-c3**2)
    s3_neg = -np.sqrt(1-c3**2)

    theta3_iii = -np.arctan2(s3_neg, c3)
    theta3_iv = -np.arctan2(s3_pos, c3)

    c2_poss3 = (np.sqrt(Px**2 + Py**2) * (r3+r4*c3) + Pz * r4 * s3_pos)/(r3**2+r4**2+2*r3*r4*c3)
    c2_negs3 = (np.sqrt(Px**2 + Py**2) * (r3+r4*c3) + Pz * r4 * s3_neg)/(r3**2+r4**2+2*r3*r4*c3)
     
    theta2_ii = np.arctan2(-s2_pluss_negs3, -c2_poss3)
    theta2_iv = np.arctan2(-s2_pluss_poss3, -c2_negs3)

    alt1 = np.array([theta1_i, theta2_i,    theta3_i   + np.arctan(35/420)])
    alt2 = np.array([theta1_i, theta2_iii , theta3_ii  + np.arctan(35/420)])
    alt3 = np.array([theta1_ii, theta2_ii , theta3_iii + np.arctan(35/420)])
    alt4 = np.array([theta1_ii, theta2_iv , theta3_iv  + np.arctan(35/420)])


    return np.array([alt1,alt2,alt3,alt4])



def inverseKinematicsTheta456(thetalists, T_SB, S, M):
    '''
    Inputs:
        thetalists: a list of thetas for each joint
        T_SB: a 4x4 transformation matrix from the base to the end effector
        S: a 6x6 matrix of the space fixed screw axis
        M: a 4x4 matrix of robot in its home position
    Outputs:
        thetalists: a list of thetas 1, 2, and 3
    '''
    theta = [] #initialize theta return list

    for i in thetalists: # for each theta in the list
        # calculate Transformation matrix from 4 to 6 frame
        T_46 = exp6(-S[:,2], i[2]) @ exp6(-S[:,1], i[1]) @ exp6(-S[:,0], i[0]) @ T_SB @ np.linalg.inv(M)
        R_46 = T_46[:3,:3] # extract rotation matrix

        # calculating theta 4, 5, and 6
        theta4_i = np.arctan2( float(R_46[0,1]) , float(R_46[0,2]))-np.pi/2
        theta5_i = np.arctan2(np.sqrt(float(R_46[0,1])**2 + float(R_46[0,2])),float(R_46[2,1]))-np.pi/2
        theta6_i = np.arctan2(-float(R_46[2,1]) , float(R_46[2,0]))

        # merging input solution with theta 4, 5, and 6
        thetalist = np.concatenate((i , np.array([theta4_i, theta5_i, theta6_i])))
        thetalist = np.around(thetalist*180/np.pi,2)
        theta.append(thetalist)

    # R_46sp = sprotz(th4) @ sproty(th5) @ sprotz(th6) # testing

    return np.array(theta)
