import numpy as np
import itk
from itk import RTK as rtk
from scipy import interpolate
from math import *


def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = B-A
    AC = C-A
    normal = np.cross(AB, AC)
    normal /= np.linalg.norm(normal)
    d = -1*np.dot(normal, C)
    return [normal, d]


def ExtractSourcePosition(geometry0, geometry1):
    sourcePos0 = geometry0.GetSourcePosition(0)
    sourcePos1 = geometry1.GetSourcePosition(0)
    sourcePos0 = itk.GetArrayFromVnlVector(sourcePos0.GetVnlVector())[0:3]
    sourcePos1 = itk.GetArrayFromVnlVector(sourcePos1.GetVnlVector())[0:3]
    return sourcePos0, sourcePos1


def ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1):  # Compute the coordinates in the canonical frame of the detectors
    D = geometry0.GetSourceToDetectorDistances()[0]
    Rd = geometry0.GetRadiusCylindricalDetector()
    R_traj = geometry0.GetSourceToIsocenterDistances()[0]

    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    
    projIdxToCoord0 = geometry0.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord1 = geometry1.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord0 = itk.GetArrayFromVnlMatrix(projIdxToCoord0.GetVnlMatrix().as_matrix())
    projIdxToCoord1 = itk.GetArrayFromVnlMatrix(projIdxToCoord1.GetVnlMatrix().as_matrix())
    directionProj = itk.GetArrayFromMatrix(projection0.GetDirection())

    # Check for non negative spacing
    matId = np.identity(3)
    matProd = directionProj * matId != directionProj
    if (np.sum(matProd) != 0):
        print("la matrice a %f element(s) non diagonal(aux)" % (np.sum(matProd)))
    else:
        size = []
        for k in range(len(projection0.GetOrigin())):
            size.append(projection0.GetSpacing()[k]*(projection0.GetLargestPossibleRegion().GetSize()[k]-1)*directionProj[k, k])

    Det0 = np.zeros((projection0.GetLargestPossibleRegion().GetSize()[0], projection0.GetLargestPossibleRegion().GetSize()[1], 3))
    Det1 = np.zeros((projection0.GetLargestPossibleRegion().GetSize()[0], projection0.GetLargestPossibleRegion().GetSize()[1], 3))
    thetaTot = np.zeros((projection0.GetLargestPossibleRegion().GetSize()[0], projection0.GetLargestPossibleRegion().GetSize()[1]))
    Nx, Ny = Det0.shape[0:2]
    
    if Rd == 0:  # flat detector
        for j in range(Ny):
            for i in range(Nx):
                u = projection0.GetOrigin()[0] + i*projection0.GetSpacing()[0]*directionProj[0, 0]
                v = projection0.GetOrigin()[1] + j*projection0.GetSpacing()[1]*directionProj[1, 1]
                w = 0
                idx = np.array((u, v, w, 1))
                coord0 = projIdxToCoord0.dot(idx)
                coord1 = projIdxToCoord1.dot(idx)
                thetaTot[i, j] += u
                Det0[i, j, :] += coord0[0:3]
                Det1[i, j, :] += coord1[0:3]
        theta = thetaTot[:,0]
    else:  # cylindrical detector
        theta = (projection0.GetOrigin()[0] + np.arange(Nx)*projection0.GetSpacing()[0]*directionProj[0, 0]+geometry0.GetProjectionOffsetsX()[0])/D
        for j in range(Ny):
            Det0[:, j, 0] += (R_traj-Rd*np.cos(theta))*np.sin(a0+geometry0.GetSourceOffsetsX()[0]/R_traj) + Rd*np.sin(theta)*np.cos(a0 + geometry0.GetSourceOffsetsX()[0]/R_traj)  
            Det0[:, j, 1] += np.ones(Nx)*(projection0.GetOrigin()[1] + j*projection0.GetSpacing()[1]*directionProj[1, 1] + geometry0.GetProjectionOffsetsY()[0])
            Det0[:, j, 2] += (R_traj-Rd*np.cos(theta))*np.cos(a0+ geometry0.GetSourceOffsetsX()[0]/R_traj) - Rd*np.sin(theta)*np.sin(a0+ geometry0.GetSourceOffsetsX()[0]/R_traj)
            Det1[:, j, 0] = (R_traj-Rd*np.cos(theta))*np.sin(a1+ geometry1.GetSourceOffsetsX()[0]/R_traj) + Rd*np.sin(theta)*np.cos(a1+ geometry1.GetSourceOffsetsX()[0]/R_traj) 
            Det1[:, j, 1] = np.ones(Nx)*(projection1.GetOrigin()[1] + j*projection1.GetSpacing()[1]*directionProj[1, 1] + geometry1.GetProjectionOffsetsY()[0])
            Det1[:, j, 2] = (R_traj-Rd*np.cos(theta))*np.cos(a1+ geometry1.GetSourceOffsetsX()[0]/R_traj) - Rd*np.sin(theta)*np.sin(a1+ geometry1.GetSourceOffsetsX()[0]/R_traj)
    return Det0, Det1, theta


def ComputeCylindersIntersection(geometry0, geometry1):  # Compute the intersection of two cylindrical detectors
    D = geometry0.GetSourceToDetectorDistances()[0]
    s0, s1 = ExtractSourcePosition(geometry0, geometry1)
    # on définit les deux vecteurs directeurs
    u_dir = (np.array([s1[2], s1[0]])-np.array([s0[2], s0[0]]))/np.linalg.norm((np.array([s1[2], s1[0]])-np.array([s0[2], s0[0]])))
    v_dir = np.array([-u_dir[1], u_dir[0]])  # perpendiculaire
    
    mid_point = (s0+s1)/2

#     a = v_dir[0]**2+v_dir[1]**2
#     b = 2*(v_dir[0]*(mid_point[2]-s1[2]) + v_dir[1]*(mid_point[0]-s1[0]))
#     c = (mid_point[2]-s1[2])**2 + (mid_point[0]-s1[0])**2 - D**2
#     roots = np.array(np.roots([a, b, c]))
    c = (s0[2]-s1[2])**2/4 + (s0[0]-s1[0])**2/4 - D**2
    roots = np.array([-np.sqrt(np.abs(c)),np.sqrt(np.abs(c))])

    t1 = np.array([roots[0]*v_dir[1]+mid_point[0], 0, roots[0]*v_dir[0]+mid_point[2]])
    t2 = np.array([roots[1]*v_dir[1]+mid_point[0], 0, roots[1]*v_dir[0]+mid_point[2]])

    return t1, t2


def ComputeNewFrame(geometry0, geometry1):  # Compute the frame centered on each source position
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    sourceDir0 = sourcePos0[0:3] - sourcePos1[0:3]  # First direction is the line s0/s1
#     if np.abs(a1-a0) > np.pi and a0 < a1:  # Always keep angle between 0-pi
#         sourceDir0 *= -1
    if a0 < np.pi and a1 < np.pi:
        if np.dot(sourceDir0,np.array([0,0,1]))<0 and np.abs(a1-a0) <= np.pi :
            sourceDir0 *= -1
        elif np.abs(a1-a0) > np.pi and np.dot(sourceDir0,np.array([0,0,1]))>0:
            sourceDir0 *= -1
    elif a0 > np.pi and a1 > np.pi:
        if np.dot(sourceDir0,np.array([0,0,1]))>0 and np.abs(a1-a0) <= np.pi:
            sourceDir0 *= -1 
    elif np.dot(sourceDir0,np.array([0,0,1]))==0 and a0 < np.pi and a1 > np.pi and np.abs(a1-a0) > np.pi: 
        sourceDir0 *= -1 
    elif np.dot(sourceDir0,np.array([0,0,1]))==0 and a0 > np.pi and a1 < np.pi and np.abs(a1-a0) < np.pi: 
        sourceDir0 *= -1 
        
    sourceDir0[1] = 0
    sourceDir1 = np.array([0, 1, 0])  # Axial direction is kept the same
    sourceDir2 = np.cross(sourceDir0, sourceDir1)  # The third direction is obtained to have a right-handed coordinate system and points towards the detector
    sourceDir0 /= np.linalg.norm(sourceDir0)
    sourceDir2 /= np.linalg.norm(sourceDir2)
    volDir = np.vstack((sourceDir0, sourceDir1, sourceDir2))
    return volDir

def ComputeNewFrameAndMPoints(Nx, DeltaY, D0, D1, geometry0, geometry1):  # Compute the new frame and the M-points that are used to form planes with the source positions for the cylindrical detectors
    volDir = ComputeNewFrame(geometry0, geometry1)
    # Axial coordinates
    y_min = np.max([D0[Nx//2, 0, 1], D1[Nx//2, 0, 1]])
    y_max = np.min([D0[Nx//2, -1, 1], D1[Nx//2, -1, 1]])
    y_dcc = np.linspace(y_min + DeltaY/2, y_max - DeltaY/2, floor(np.abs(y_max-y_min)/DeltaY))
#     print(len(y_dcc),y_min, y_max, floor(np.abs(y_max-y_min)/DeltaY),np.abs(y_max-y_min)/DeltaY)
    
    # Radial coordinates
    ta, tb = ComputeCylindersIntersection(geometry0, geometry1)
    dist0a = np.linalg.norm(np.array([ta[0],ta[2]])-np.array([D0[Nx//2,0,0],D0[Nx//2,0,2]]))
    dist0b = np.linalg.norm(np.array([tb[0],tb[2]])-np.array([D0[Nx//2,0,0],D0[Nx//2,0,2]]))
    dist1a = np.linalg.norm(np.array([ta[0],ta[2]])-np.array([D1[Nx//2,0,0],D1[Nx//2,0,2]]))
    dist1b = np.linalg.norm(np.array([tb[0],tb[2]])-np.array([D1[Nx//2,0,0],D1[Nx//2,0,2]]))
    if dist0a <= dist0b and dist1a <= dist1b:
        intersect = ta
#         print("ta")
    else:
        intersect = tb
#         print("tb")
    PMs = np.array([intersect[0]*np.ones(len(y_dcc)), y_dcc, intersect[2]*np.ones(len(y_dcc))])
    return PMs

def CanWeApplyDirectlyTheFormula(angle, xs):  # Check if there is a singularity present in the samples
    a = np.min(angle)
    b = np.max(angle)
    if np.sign(a*b) >0 and xs < a and xs > b:
        return True
    else:
        if xs < 0 and xs < a and np.abs(xs) > b:
            return True
        elif xs > 0 and xs > np.abs(a) and xs > b:
            return True
        else:
            return False
        

def ComputeAllInOneFunction(a0, a1, sourcePos0, sourcePos1, M_points, gamma, v_det, D):
    b = sourcePos0[0:3] - sourcePos1[0:3]
    b /= np.linalg.norm(b)
    if (np.dot(b, np.array([1., 0., 0.])) < 0) :
        b *=-1
            
    n0 = []
    n1 = []
    M_ACC = []
    
    dv = np.abs(v_det[1]-v_det[0])
            
    volDir0 = np.vstack((np.array([np.cos(a0),0,-np.sin(a0)]),np.array([0,1,0]),np.array([np.sin(a0),0,np.cos(a0)])))
    b0 = np.dot(volDir0,b)
    volDir1 = np.vstack((np.array([np.cos(a1),0,-np.sin(a1)]),np.array([0,1,0]),np.array([np.sin(a1),0,np.cos(a1)])))
    b1 = np.dot(volDir1,b)
    
    for j in range(len(M_points[1])):
        n, d = ComputePlaneEquation(sourcePos0, sourcePos1, np.array([M_points[0][j], M_points[1][j], M_points[2][j]]))
#         delta = (sourcePos0[2]-sourcePos1[2])**2/4 + (sourcePos0[0]-sourcePos1[0])**2/4 - D**2
#         an1 = (sourcePos1[0]-sourcePos0[0])*(M_points[1][j]-sourcePos0[1])-(sourcePos1[1]-sourcePos0[1])*((sourcePos1[0]-sourcePos0[0])/2-np.sqrt(np.abs(delta))*(sourcePos1[2]-sourcePos0[2])/np.sqrt((sourcePos0[2]-sourcePos1[2])**2 + (sourcePos0[0]-sourcePos1[0])**2))
#         an2 = (sourcePos1[1]-sourcePos0[1])*((sourcePos1[2]-sourcePos0[2])/2-np.sqrt(np.abs(delta))*(sourcePos0[0]-sourcePos1[0])/np.sqrt((sourcePos0[2]-sourcePos1[2])**2 + (sourcePos0[0]-sourcePos1[0])**2))-(sourcePos1[2]-sourcePos0[2])*(M_points[1][j]-sourcePos0[1])
#         an3 = -np.sqrt(np.abs(delta))*np.sqrt((sourcePos0[2]-sourcePos1[2])**2 + (sourcePos0[0]-sourcePos1[0])**2)        
#         an = np.array([an1,an2,an3])/np.linalg.norm(np.array([an1,an2,an3]))
# #         print(an1,an2,an3)
        n_new0 = np.dot(volDir0,n)
        n_new1 = np.dot(volDir1,n)
        gamma_e0 = np.arctan(-n_new0[0]/n_new0[2])
        gamma_e1 = np.arctan(-n_new1[0]/n_new1[2])

        if CanWeApplyDirectlyTheFormula(gamma,gamma_e0):
            x0 = np.array([gamma[0],gamma[-1]])
        else:
            x0 = np.array([gamma[0],gamma_e0,gamma[-1]])
        if CanWeApplyDirectlyTheFormula(gamma,gamma_e1):
            x1 = np.array([gamma[0],gamma[-1]])
        else:
            x1 = np.array([gamma[0],gamma_e1,gamma[-1]])
            
        v0 = D*(-np.sin(x0)*n_new0[0]+np.cos(x0)*n_new0[2])/n_new0[1]
        v1 = D*(-np.sin(x1)*n_new1[0]+np.cos(x1)*n_new1[2])/n_new1[1]

        c0_min  = (np.min(v_det)  <= np.min(v0) and np.min(v0) <= np.max(v_det))
        c0_max = (np.min(v_det) <= np.max(v0) and np.max(v0) <= np.max(v_det))
        c1_min  = (np.min(v_det) <= np.min(v1) and np.min(v1) <= np.max(v_det))
        c1_max = (np.min(v_det) <= np.max(v1) and np.max(v1) <= np.max(v_det))

        if (c0_min and c0_max) and (c1_min and c1_max):
            M_ACC.append(np.array([M_points[0][j], M_points[1][j], M_points[2][j]]))
            n0.append(n_new0)
            n1.append(n_new1)
            
    return np.array(M_ACC), np.array(n0), b0, np.array(n1), b1
  
def ComputeAllDetectorPlanesIntersections(geometry0, geometry1, Det0, Det1, M_points, DeltaY):  # Compute and select the intersection curves between a pair of detectors and the different planes that are not truncated in the canonical frame
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)

    x0 = Det0[:, 0, 0]
    z0 = Det0[:, 0, 2]
    x1 = Det1[:, 0, 0]
    z1 = Det1[:, 0, 2]
    Nx = Det0.shape[0]

#     Inter0 = []
#     Inter1 = []
    M_ACC = []
    Normals = []

    for i in range(len(M_points[1])):
        plane = ComputePlaneEquation(sourcePos0, sourcePos1, np.array([M_points[0][i], M_points[1][i], M_points[2][i]]))
        y0 = -(plane[0][0]*x0+plane[0][2]*z0+plane[1])/plane[0][1]
        y1 = -(plane[0][0]*x1+plane[0][2]*z1+plane[1])/plane[0][1]
        c0m = (Nx == len(np.where(y0 > np.min(Det0[0, :, 1]))[0]))
        c0M = (Nx == len(np.where(y0 < np.max(Det0[0, :, 1]))[0]))
        c1m = (Nx == len(np.where(y1 > np.min(Det1[0, :, 1]))[0]))
        c1M = (Nx == len(np.where(y1 < np.max(Det1[0, :, 1]))[0]))
        if (c0m and c0M) and (c1m and c1M):
#             Inter0.append(y0)
#             Inter1.append(y1)
            M_ACC.append(np.array([M_points[0][i], M_points[1][i], M_points[2][i]]))
            Normals.append(plane[0])
#     return x1, z1, np.array(Inter1), x0, z0, np.array(Inter0), np.array(M_ACC), np.array(Normals)
    return  np.array(M_ACC), np.array(Normals)

def ChangeOfFrameForAll(a, Scenter, s0, s1, Det, x, z, y, D, M_ACC, xe):
    volDirection = np.vstack((np.array([np.cos(a),0,-np.sin(a)]),np.array([0,1,0]),np.array([np.sin(a),0,np.cos(a)])))
    # Change of frame for the source positions and the third points that formed a plane
    s1bis = np.dot(volDirection,s1-Scenter)
    s0bis = np.dot(volDirection,s0-Scenter)
    xe = np.dot(volDirection,xe)
    xe /= np.linalg.norm(xe)
    mpoints = np.zeros(M_ACC.shape)

    # Change of frame for the detector corresponding to Scenter
    Detbis = np.zeros(Det.shape)
    Interbis = np.zeros((Det.shape[0], len(y), 3))  # Coordinates of the intersection curves in the new frame
    ze = np.zeros((len(y),3))

    for j in range(Det.shape[1]):
        for i in range(Det.shape[0]):
            Detbis[i, j, :] =  np.dot(volDirection,Det[i, j, :]-Scenter)
    for k in range(y.shape[0]):
        mpoints[k, :] +=  np.dot(volDirection,M_ACC[k, :]-Scenter)
        planebis = ComputePlaneEquation(s0bis,s1bis,mpoints[k,:])
        ze[k,:]+=planebis[0]                                        
        for o in range(Det.shape[0]):
            Interbis[o, k, :] += np.dot(volDirection,np.array([x[o], y[k, o], z[o]])-Scenter)
    return Interbis, s0bis, s1bis, Detbis, mpoints, ze, xe

def ComputeMomentsOnCylindricalDetectors(geometry0, geometry1, projection0, projection1):
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    dx0, dx1 = geometry0.GetProjectionOffsetsX()[0], geometry1.GetProjectionOffsetsX()[0]
    sx0, sx1 = geometry0.GetSourceOffsetsX()[0], geometry1.GetSourceOffsetsX()[0]
    sy0, sy1 = geometry0.GetSourceOffsetsY()[0], geometry1.GetSourceOffsetsY()[0]
    dy0, dy1 = geometry0.GetProjectionOffsetsY()[0], geometry1.GetProjectionOffsetsY()[0]
    projar0 = itk.GetArrayFromImage(projection0)
    projar1 = itk.GetArrayFromImage(projection1)
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    D = geometry0.GetSourceToDetectorDistances()[0]
    R = geometry0.GetSourceToIsocenterDistances()[0]
    [DeltaX, DeltaY, DeltaZ] = projection0.GetSpacing()
    [Nx, Ny, Nz] = projection0.GetLargestPossibleRegion().GetSize()
    
    projIdxToCoord0 = geometry0.GetProjectionCoordinatesToFixedSystemMatrix(0)
    projIdxToCoord0 = itk.GetArrayFromVnlMatrix(projIdxToCoord0.GetVnlMatrix().as_matrix())
    directionProj = itk.GetArrayFromMatrix(projection0.GetDirection())
    
    # Compute the two detectors coordinates RTK angles
    Det0, Det1, gamma_RTK = ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1)
    
    # Compute intersection between the two detectors and all possible M points
    PMs = ComputeNewFrameAndMPoints(Nx, DeltaY, Det0, Det1, geometry0, geometry1)

    v_det =  projection0.GetOrigin()[1] + dy0-sy0+ (np.arange(Ny)*projection0.GetSpacing()[1])*directionProj[1, 1] 
    M_ACC, n0, b0, n1, b1 = ComputeAllInOneFunction(a0, a1, sourcePos0, sourcePos1, PMs, gamma_RTK, v_det, D)
    
    # Computation of the projection weights
    new_cos0 = np.zeros((Det0.shape[0], n0.shape[0]))
    new_cos1 = np.zeros((Det1.shape[0], n1.shape[0]))
    v0 = np.zeros((Det0.shape[0], n0.shape[0]))
    v1 = np.zeros((Det1.shape[0], n1.shape[0]))
    for j in range(new_cos0.shape[1]):
        v0[:,j] = (-n0[j,0]*D*np.sin(gamma_RTK)+n0[j,2]*D*np.cos(gamma_RTK))/n0[j,1]
        new_cos0[:, j] = np.sqrt(D**2+v0[:,j]**2)*(np.cos(gamma_RTK)*b0[0]+np.sin(gamma_RTK)*b0[2])/D
        v1[:,j] = (-n1[j,0]*D*np.sin(gamma_RTK)+n1[j,2]*D*np.cos(gamma_RTK))/n1[j,1]
        new_cos1[:, j] = np.sqrt(D**2+v1[:,j]**2)*(np.cos(gamma_RTK)*b1[0]+np.sin(gamma_RTK)*b1[2])/D

    # Axial interpolation of the projection at the different intersection points of the curve
    proj_interp0 = np.zeros(new_cos0.shape)
    proj_interp1 = np.zeros(new_cos1.shape)
    for i in range(Det0.shape[0]):  
        proj_interp0[i,:] += np.interp(v0[i,:],v_det,projar0[0, :, i])
        proj_interp1[i,:] += np.interp(v1[i,:],v_det,projar1[0, :, i])

    m0 = np.sum(proj_interp0/new_cos0, axis=0)*np.abs(gamma_RTK[1]-gamma_RTK[0])
    m1 = np.sum(proj_interp1/new_cos1, axis=0)*np.abs(gamma_RTK[1]-gamma_RTK[0])
    
    return m0, m1

def ComputeMomentsOnCylindricalDetectorsWithSingularity(geometry0, geometry1, projection0, projection1):
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    dx0, dx1 = geometry0.GetProjectionOffsetsX()[0], geometry1.GetProjectionOffsetsX()[0]
    sx0, sx1 = geometry0.GetSourceOffsetsX()[0], geometry1.GetSourceOffsetsX()[0]
    sy0, sy1 = geometry0.GetSourceOffsetsY()[0], geometry1.GetSourceOffsetsY()[0]
    dy0, dy1 = geometry0.GetProjectionOffsetsY()[0], geometry1.GetProjectionOffsetsY()[0]
    projar0 = itk.GetArrayFromImage(projection0)
    projar1 = itk.GetArrayFromImage(projection1)
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    D = geometry0.GetSourceToDetectorDistances()[0]
    R = geometry0.GetSourceToIsocenterDistances()[0]
    [DeltaX, DeltaY, DeltaZ] = projection0.GetSpacing()
    [Nx, Ny, Nz] = projection0.GetLargestPossibleRegion().GetSize()
    
    directionProj = itk.GetArrayFromMatrix(projection0.GetDirection())
    
    # Compute the two detectors coordinates RTK angles
    Det0, Det1, gamma_RTK = ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1)

    # Compute intersection between the two detectors and all possible M points
    PMs = ComputeNewFrameAndMPoints(Nx, DeltaY, Det0, Det1, geometry0, geometry1)

    v_det =  projection0.GetOrigin()[1] + dy0-sy0 + (np.arange(Ny)*projection0.GetSpacing()[1])*directionProj[1, 1] 
    M_ACC, n0, b0, n1, b1 = ComputeAllInOneFunction(a0, a1, sourcePos0, sourcePos1, PMs, gamma_RTK, v_det, D)

    
    # Axial interpolation of the projection at the different intersection points of the curve
    proj_interp0 = np.zeros((Det0.shape[0], n0.shape[0]))
    proj_interp1 = np.zeros((Det1.shape[0], n1.shape[0]))
    v0 = np.zeros((Det0.shape[0], n0.shape[0]))
    v1 = np.zeros((Det1.shape[0], n1.shape[0]))
    for i in range(Det0.shape[0]):  
        v0[i,:] = (-n0[:,0]*D*np.sin(gamma_RTK[i])+n0[:,2]*D*np.cos(gamma_RTK[i]))/n0[:,1]
        v1[i,:] = (-n1[:,0]*D*np.sin(gamma_RTK[i])+n1[:,2]*D*np.cos(gamma_RTK[i]))/n1[:,1]
        proj_interp0[i,:] += np.interp(v0[i,:],v_det,projar0[0, :, i])
        proj_interp1[i,:] += np.interp(v1[i,:],v_det,projar1[0, :, i])
          
    m0_trpz, m1_trpz = np.zeros(v0.shape[1]), np.zeros(v1.shape[1])
    norm0, norm1 = np.zeros(v0.shape[1]), np.zeros(v1.shape[1])
    grad0, grad1 = np.zeros(v0.shape[1]), np.zeros(v1.shape[1])
    
    da = np.abs(a1-a0)
    
    for j in range(v1.shape[1]):
        xs0 = np.arctan(-b0[0]/b0[2])
        if CanWeApplyDirectlyTheFormula(gamma_RTK, xs0):
            new_cos0 = np.sqrt(D**2+v0[:,j]**2)*(np.cos(gamma_RTK)*b0[0]+np.sin(gamma_RTK)*b0[2])/D
            m0_trpz[j], norm0[j], grad0[j] = np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp0[:, j]/new_cos0), np.abs(np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp0[:, j]/new_cos0)),0
        else:
            m0_trpz[j], norm0[j], grad0[j] = TrapIntegration(da, xs0, gamma_RTK, proj_interp0[:, j], v0[:,j], b0, D)
        xs1 = np.arctan(-b1[0]/b1[2])
        if CanWeApplyDirectlyTheFormula(gamma_RTK, xs1):
            new_cos1 = np.sqrt(D**2+v1[:,j]**2)*(np.cos(gamma_RTK)*b1[0]+np.sin(gamma_RTK)*b1[2])/D
            m1_trpz[j], norm1[j], grad1[j] = np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp1[:, j]/new_cos1), np.abs(np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp1[:, j]/new_cos1)),0
        else:
            m1_trpz[j], norm1[j], grad1[j] = TrapIntegration(da, xs1, gamma_RTK, proj_interp1[:, j], v1[:,j], b1, D)
    return m0_trpz, m1_trpz, norm0, norm1, grad0, grad1


def TestForSingularity(angle, xs):  #Find the indices of the angles surrounding the singularity
    a = 0
    b = 0
    if len(np.where(angle >= xs)[0]) != len(angle) and len(np.where(angle <= xs)[0]) != len(angle):
        if angle[1]-angle[0] >0:
            a = np.where(angle <= xs)[-1][-1]
            b = np.where(angle >= xs)[-1][0]
        else : 
            a = np.where(angle <= xs)[-1][0]
            b = np.where(angle >= xs)[-1][-1]
    return a, b

def TrapIntegration(da, xs, gamma, ar, v, xe, D):  #Perform numerical integration using trapezoidale rule
    h = np.abs(gamma[1]-gamma[0])
    g = D*ar/(np.sqrt(D**2+v**2)*(np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2]))
    ii, iii = TestForSingularity(gamma, xs)
    f_lin = D * ar / np.sqrt(D**2+v**2)
    f_int = interpolate.interp1d(gamma,f_lin)
#     print(gamma[1]-gamma[0])

    grad = (f_int(xs+h/2)-f_int(xs-h/2))/h
    
    if np.abs(gamma[ii]-xs)<np.abs(gamma[iii]-xs):
        summ = np.trapz(g[0:ii],gamma[0:ii],gamma[1]-gamma[0])+np.trapz(g[iii:],gamma[iii:],gamma[1]-gamma[0])
#         summ = integrate.simps(g[0:ii],gamma[0:ii],gamma[1]-gamma[0]) + integrate.simps(g[iii:],gamma[iii:],gamma[1]-gamma[0])
#         summ = GaussIntegration(f_int,gamma[0],gamma[ii-1],50) + GaussIntegration(f_int,gamma[iii],gamma[-1],50)
        a = (f_lin[iii]-f_lin[ii-1])/(gamma[iii]-gamma[ii-1])
        b = f_lin[iii]-a*gamma[iii]
        c = ((np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2])-(np.cos(gamma[ii-1])*xe[0]+np.sin(gamma[ii-1])*xe[2]))/(gamma[iii]-gamma[ii-1])
        d = (np.cos(gamma[iii])*xe[0]+np.sin(gamma[iii])*xe[2]) - c*gamma[iii]
        rest_ii = (a*(c*gamma[ii-1]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii-1]+d)))
        rest_iii = (a*(c*gamma[iii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(np.trapz(g[0:ii],gamma[0:ii],gamma[1]-gamma[0]))+np.abs(np.trapz(g[iii:],gamma[iii:],gamma[1]-gamma[0])) + np.abs(rest)
        
#         plt.figure()
#         plt.subplot(231)
#         plt.plot(gamma,f_lin)
#         plt.plot(gamma[ii-1:iii+1],a*gamma[ii-1:iii+1]+b,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(232)
#         plt.plot(gamma,np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2])
#         plt.plot(gamma[ii-1:iii+1],c*gamma[ii-1:iii+1]+d,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(233)
#         plt.plot(gamma,g)
#         plt.plot(gamma,np.abs(g))
#         plt.plot(gamma[ii-1:iii+1],(a*gamma[ii-1:iii+1]+b)/(c*gamma[ii-1:iii+1]+d),'k--')
#         plt.plot(gamma[ii],g[ii],'.')
#         plt.plot(gamma[iii],g[iii],'.')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(234)
#         plt.plot(gamma[ii-1:iii+1],f_lin[ii-1:iii+1])
#         plt.plot(gamma[ii-1:iii+1],a*gamma[ii-1:iii+1]+b,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(235)
#         plt.plot(gamma[ii-1:iii+1],np.cos(gamma[ii-1:iii+1])*xe[0]+np.sin(gamma[ii-1:iii+1])*xe[2])
#         plt.plot(gamma[ii-1:iii+1],c*gamma[ii-1:iii+1]+d,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(236)
#         plt.plot(gamma[ii-1:iii+1],g[ii-1:iii+1])
#         plt.plot(gamma[ii-1:iii+1],(a*gamma[ii-1:iii+1]+b)/(c*gamma[ii-1:iii+1]+d),'k--')
#         plt.plot(gamma[ii],g[ii],'.')
#         plt.plot(gamma[iii],g[iii],'.')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.show()
           
    else:
        summ = np.trapz(g[0:ii+1],gamma[0:ii+1],gamma[1]-gamma[0])+np.trapz(g[iii+1:],gamma[iii+1:],gamma[1]-gamma[0])
#         summ = integrate.simps(g[0:ii+1],gamma[0:ii+1],gamma[1]-gamma[0]) + integrate.simps(g[iii+1:],gamma[iii+1:],gamma[1]-gamma[0])
#         summ = GaussIntegration(f_int,gamma[0],gamma[ii],50) + GaussIntegration(f_int,gamma[iii+1],gamma[-1],50)
        a = (f_lin[iii+1]-f_lin[ii])/(gamma[iii+1]-gamma[ii])
        b = f_lin[ii]-a*gamma[ii]
        c = ((np.cos(gamma[iii+1])*xe[0]+np.sin(gamma[iii+1])*xe[2])-(np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]))/(gamma[iii+1]-gamma[ii])
        d = (np.cos(gamma[ii])*xe[0]+np.sin(gamma[ii])*xe[2]) - c*gamma[ii]
        rest_ii = (a*(c*gamma[ii]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[ii]+d)))
        rest_iii = (a*(c*gamma[iii+1]+d)+(b*c-a*d)*np.log(np.abs(c*gamma[iii+1]+d)))
        rest = (rest_iii-rest_ii)/c**2
        norm = np.abs(np.trapz(g[0:ii+1],gamma[0:ii+1],gamma[1]-gamma[0]))+np.abs(np.trapz(g[iii+1:],gamma[iii+1:],gamma[1]-gamma[0])) + np.abs(rest)
        
#         plt.figure()
#         plt.subplot(231)
#         plt.plot(gamma,f_lin)
#         plt.plot(gamma[ii:iii+2],a*gamma[ii:iii+2]+b,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(232)
#         plt.plot(gamma,np.cos(gamma)*xe[0]+np.sin(gamma)*xe[2])
#         plt.plot(gamma[ii:iii+2],c*gamma[ii:iii+2]+d,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(233)
#         plt.plot(gamma,g)
#         plt.plot(gamma,np.abs(g))
#         plt.plot(gamma[ii:iii+2],(a*gamma[ii:iii+2]+b)/(c*gamma[ii:iii+2]+d),'k--')
#         plt.plot(gamma[ii:iii+2],np.abs((a*gamma[ii:iii+2]+b)/(c*gamma[ii:iii+2]+d)),'--')
#         plt.plot(gamma[ii],g[ii],'.')
#         plt.plot(gamma[iii],g[iii],'.')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(234)
#         plt.plot(gamma[ii:iii+2],f_lin[ii:iii+2])
#         plt.plot(gamma[ii:iii+2],a*gamma[ii:iii+2]+b,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(235)
#         plt.plot(gamma[ii:iii+2],np.cos(gamma[ii:iii+2])*xe[0]+np.sin(gamma[ii:iii+2])*xe[2])
#         plt.plot(gamma[ii:iii+2],c*gamma[ii:iii+2]+d,'k--')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.subplot(236)
#         plt.plot(gamma[ii:iii+2],g[ii:iii+2])
#         plt.plot(gamma[ii:iii+2],(a*gamma[ii:iii+2]+b)/(c*gamma[ii:iii+2]+d),'k--')
#         plt.plot(gamma[ii],g[ii],'.')
#         plt.plot(gamma[iii],g[iii],'.')
#         plt.axvline(x=xs,color='red',linewidth=0.5)
#         plt.show()

    
    return summ+rest, norm, grad



def ComputeMomentsOnFlatDetectors(geometry0, geometry1, projection0, projection1):
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    projar0 = itk.GetArrayFromImage(projection0)
    projar1 = itk.GetArrayFromImage(projection1)
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    D = geometry0.GetSourceToDetectorDistances()[0]
    [DeltaX, DeltaY, DeltaZ] = projection0.GetSpacing()
    [Nx, Ny, Nz] = projection0.GetLargestPossibleRegion().GetSize()

    # Compute the two detectors coordinates RTK angles
    Det0, Det1, u = ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1)

    # Fonction qui calcul le nouveau repère et les angles correspondant
    volDir, PMs = ComputeNewFrameAndMPointsFlat(Nx, DeltaY, Det0, Det1, geometry0, geometry1)

    # Compute all intersections in the canonical frame
    x1, z1, Inter1, x0, z0, Inter0, M_ACC = ComputeAllDetectorPlanesIntersections(geometry0, geometry1, Det0, Det1, PMs, DeltaY)

    # Change of frame for all the previous computations
    Interbis0, center0, s1bis, DETbis0, d00, PMs0 = ChangeOfFrameForAll(sourcePos0, sourcePos0, sourcePos1, Det0, volDir, x0, z0, Inter0, D, M_ACC)
    Interbis1, s0bis, center1, DETbis1, d01, PMs1 = ChangeOfFrameForAll(sourcePos1, sourcePos0, sourcePos1, Det1, volDir, x1, z1, Inter1, D, M_ACC)
    
    # Axial interpolation of the projection at the different intersection points of the curve
    proj_interp0 = np.zeros((Det0.shape[0], Interbis0.shape[1]))
    proj_interp1 = np.zeros((Det0.shape[0], Interbis0.shape[1]))

    for i in range(Det0.shape[0]):
        f0 = interpolate.interp1d(DETbis0[i, :, 1], projar0[0, :, i], kind='linear', axis=0)
        proj_interp0[i, :] += f0(Interbis0[i, :, 1])
        f1 = interpolate.interp1d(DETbis1[i, :, 1], projar1[0, :, i], kind='linear', axis=0)
        proj_interp1[i, :] += f1(Interbis1[i, :, 1])
        
    mu = (a1-a0)/2
        
    # Computation of the projection weights
    new_cos0 = np.zeros((Interbis0.shape[0], Interbis0.shape[1]))
    new_cos1 = np.zeros((Interbis0.shape[0], Interbis1.shape[1]))
    for j in range(Interbis0.shape[1]):
        new_cos0[:, j] = D/(np.sqrt(D**2+u**2)*np.abs(D*np.cos(mu)-u*np.sin(mu)))
        new_cos1[:, j] = D/(np.sqrt(D**2+u**2)*np.abs(D*np.cos(mu)-u*np.sin(mu)))
        
    m0 = np.sum(proj_interp0*new_cos0, axis=0)*np.abs(u[1]-u[0])
    m1 = np.sum(proj_interp1*new_cos1, axis=0)*np.abs(u[1]-u[0])
    
    return m0, m1
