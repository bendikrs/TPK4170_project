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


def analyticalInverseKinematics(T_SB):

    # Constants (chose to remake both M and Slist to simplify testing)
    r1,r2,r3,r4,a1,a2 = 0.4, 0.455, 0.420, 0.08, 0.025, 0.035
    r3_o = np.sqrt(r3**2 + a2**2) # the distance between joint 3 and the wrist
    Slist = np.array([[0,0,-1,0,0,0],
                    [0,1,0,-r1,0,a1],
                    [0,1,0,-r1,0,(r2+a1)],
                    [-1,0,0,0,-(r1+a2),0],
                    [0,1,0,-(r1+a2),0,(r2+r3+a1)],
                    [-1,0,0,0,-(r1+a2),0]]).T

    M = np.array([[0,0,1,r2+r3+r4+a1],
                [0,1,0,0],
                [-1,0,0,r1+a2],
                [0,0,0,1]])
    
    #Translate -80mm along z-axis of endframe
    T_BW = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,-0.08],
                    [0,0,0,1]])

    T_SW = np.dot(T_SB,T_BW)
    pw = np.array([T_SW[0][3],T_SW[1][3],T_SW[2][3]])
    theta11 = -np.arctan2(pw[1],pw[0])
    theta12 = -np.arctan2(-pw[1],-pw[0])

    #Forward configurations using theta11
    Px, Py, Pz =pw[0] - np.cos(theta11) * a1, \
                pw[1] + np.sin(theta11) * a1, \
                pw[2] - r1

    c3 = (Px**2 + Py**2 + Pz**2 - r2**2 - r3_o**2) / (2 * r2 * r3_o)

    s3p = np.sqrt(1-c3**2)
    s3n = -s3p

    theta31 = np.arctan2(s3p,c3) + np.arctan2(a2,r3)
    theta32 = np.arctan2(s3n,c3) + np.arctan2(a2,r3)

    c2pp = (r2 + r3_o * c3) * np.sqrt(Px**2 + Py**2) + r3_o * s3p * Pz
    c2pn = (r2 + r3_o * c3) * np.sqrt(Px**2 + Py**2) + r3_o * s3n * Pz

    s2np = (r2 + r3_o*c3) * Pz - r3_o * s3p * np.sqrt(Px**2 + Py**2)
    s2nn = (r2 + r3_o*c3) * Pz - r3_o * s3n * np.sqrt(Px**2 + Py**2)

    theta21 = np.arctan2(s2np,c2pp) 
    theta23 = np.arctan2(s2nn,c2pn) 

    #Backward configurations using theta12
    Px, Py, Pz =pw[0] - np.cos(theta12) * a1, \
                pw[1] + np.sin(theta12) * a1, \
                pw[2] - r1

    c3 = (Px**2 + Py**2 + Pz**2 - r2**2 - r3_o**2) / (2 * r2 * r3_o)

    s3p = np.sqrt(1-c3**2)
    s3n = -s3p

    theta33 = np.arctan2(s3p,c3) + np.arctan2(a2,r3)
    theta34 = np.arctan2(s3n,c3) + np.arctan2(a2,r3)

    c2np = -(r2 + r3_o * c3) * np.sqrt(Px**2 + Py**2) + r3_o * s3p * Pz
    c2nn = -(r2 + r3_o * c3) * np.sqrt(Px**2 + Py**2) + r3_o * s3n * Pz

    s2pp = (r2 + r3_o*c3) * Pz + r3_o * s3p * np.sqrt(Px**2 + Py**2)
    s2pn = (r2 + r3_o*c3) * Pz + r3_o * s3n * np.sqrt(Px**2 + Py**2)

    theta22 = np.arctan2(s2pp,c2np)
    theta24 = np.arctan2(s2pn,c2nn) 

    config = np.array([[theta11, -theta23, theta31], # forward over elbow
                    [theta11,-theta21, theta32], # forward under elbow
                    [theta12, -theta24, theta33], # backward under elbow
                    [theta12, -theta22, theta34]]) # backward over elbow

    # for i in config:
    #     #printing if the first 3 angles create a correct wrist position
    #     test = np.array([i[0],i[1],i[2],0,0,0])
    #     print(test*180/np.pi)
    #     print(np.allclose(pointFromT(T_SW), pointFromT(np.dot(FKinSpace(M,Slist,test), T_BW))))

    thetalist = np.zeros(shape=(4,6))
    thetalist2 = np.zeros(shape=(4,6))
    for i, theta in enumerate(config):
        t1,t2,t3 = theta[0],theta[1],theta[2]
        R_S3 = rotz(t1) @ roty(t2) @ roty(t3) @ M[:3,:3]

        R_SB = T_SB[:3,:3]
        R_3B = R_S3.T @ R_SB

        theta61 = np.arctan2(R_3B[2,1],-R_3B[2,0])
        theta62 = np.arctan2(-R_3B[2,1],R_3B[2,0])

        theta51 = np.arctan2(np.sqrt(1-R_3B[2,2]**2), R_3B[2,2])
        theta52 = np.arctan2(-np.sqrt(1-R_3B[2,2]**2), R_3B[2,2])
        
        theta41 = np.arctan2(R_3B[1,2],R_3B[0,2])
        theta42 = np.arctan2(-R_3B[1,2],-R_3B[0,2])

        thetalist[i] = np.concatenate((theta,np.array([theta41,theta51,theta61])))
        thetalist2[i] = np.concatenate((theta,np.array([theta42,theta52,theta62])))

    return thetalist, thetalist2, FKinSpace(M,Slist,thetalist[0])
