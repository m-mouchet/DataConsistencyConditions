import numpy as np
import itk
from itk import RTK as rtk
from scipy import interpolate


def ComputePlaneEquation(A, B, C):  # compute the cartesian equation of the plane formed by the three point ABC
    AB = B-A
    AC = C-A
    normal = np.cross(AB, AC)
    d = -1*np.dot(normal, C)
    return [normal, d]


def ExtractSourcePosition(geometry0, geometry1):
    sourcePos0 = geometry0.GetSourcePosition(0)
    sourcePos1 = geometry1.GetSourcePosition(0)
    sourcePos0 = itk.GetArrayFromVnlVector(sourcePos0.GetVnlVector())[0:3]
    sourcePos1 = itk.GetArrayFromVnlVector(sourcePos1.GetVnlVector())[0:3]
    return sourcePos0, sourcePos1


def ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1):  # Compute the coordinates in the canonical frame of the detectors
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

    for j in range(Ny):
        for i in range(Nx):
            if Rd == 0:  # flat detector
                u = projection0.GetOrigin()[0] + i*projection0.GetSpacing()[0]*directionProj[0, 0]
                v = projection0.GetOrigin()[1] + j*projection0.GetSpacing()[1]*directionProj[1, 1]
                w = 0
                idx = np.array((u, v, w, 1))
                coord0 = projIdxToCoord0.dot(idx)
                coord1 = projIdxToCoord1.dot(idx)
            else:  # cylindrical detector
                theta = (projection0.GetOrigin()[0] + i*projection0.GetSpacing()[0]*directionProj[0, 0])/Rd
                x0 = (R_traj-Rd*np.cos(theta))*np.sin(a0) + Rd*np.sin(theta)*np.cos(a0)
                y0 = sourcePos0[1] + projection0.GetOrigin()[1] + j*projection0.GetSpacing()[1]*directionProj[1, 1]
                z0 = (R_traj-Rd*np.cos(theta))*np.cos(a0) - Rd*np.sin(theta)*np.sin(a0)
                coord0 = np.array([x0, y0, z0])
                x1 = (R_traj-Rd*np.cos(theta))*np.sin(a1) + Rd*np.sin(theta)*np.cos(a1)
                y1 = sourcePos1[1] + projection0.GetOrigin()[1] + j*projection0.GetSpacing()[1]*directionProj[1, 1]
                z1 = (R_traj-Rd*np.cos(theta))*np.cos(a1) - Rd*np.sin(theta)*np.sin(a1)
                coord1 = np.array([x1, y1, z1])
            Det0[i, j, :] += coord0[0:3]
            Det1[i, j, :] += coord1[0:3]
            thetaTot[i, j] += theta
    return Det0, Det1, thetaTot[:, 0]


def ComputeCylindersIntersection(geometry0, geometry1):
    D = geometry0.GetSourceToDetectorDistances()[0]
    s0, s1 = ExtractSourcePosition(geometry0, geometry1)
    # on définit les deux vecteurs directeurs
    u_dir = (np.array([s1[2], s1[0]])-np.array([s0[2], s0[0]]))
    v_dir = np.array([-u_dir[1], u_dir[0]])  # perpendiculaire
    
    mid_point = (s0+s1)/2

    a = v_dir[0]**2+v_dir[1]**2
    b = 2*(v_dir[0]*(mid_point[2]-s1[2]) + v_dir[1]*(mid_point[0]-s1[0]))
    c = (mid_point[2]-s1[2])**2 + (mid_point[0]-s1[0])**2 - D**2

    roots = np.array(np.roots([a, b, c]))
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


def ComputeAllDetectorPlanesIntersections(geometry0, geometry1, Det0, Det1, M_points, DeltaY):
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)

    x0 = Det0[:, 0, 0]
    z0 = Det0[:, 0, 2]
    x1 = Det1[:, 0, 0]
    z1 = Det1[:, 0, 2]
    Nx = Det0.shape[0]

    Inter0 = []
    Inter1 = []
    M_ACC = []

    for i in range(len(M_points[1])):
        plane = ComputePlaneEquation(sourcePos0, sourcePos1, np.array([M_points[0][i], M_points[1][i], M_points[2][i]]))
        y0 = -(plane[0][0]*x0+plane[0][2]*z0+plane[1])/plane[0][1]
        y1 = -(plane[0][0]*x1+plane[0][2]*z1+plane[1])/plane[0][1]
        c0m = (Nx == len(np.where(y0 >= np.min(Det0[0, :, 1]))[0]))
        c0M = (Nx == len(np.where(y0 <= np.max(Det0[0, :, 1]))[0]))
        c1m = (Nx == len(np.where(y1 >= np.min(Det1[0, :, 1]))[0]))
        c1M = (Nx == len(np.where(y1 <= np.max(Det1[0, :, 1]))[0]))
        if (c0m and c0M) and (c1m and c1M):
            Inter0.append(y0)
            Inter1.append(y1)
            M_ACC.append(np.array([M_points[0][i], M_points[1][i], M_points[2][i]]))
    return x1, z1, np.array(Inter1), x0, z0, np.array(Inter0), np.array(M_ACC)


def ChangeOfFrameForAll(Scenter, otherS, Det, volDirection, x, z, y, D, M_ACC):
    # Change of frame for the source positions and the third points that formed a plane
    center = np.zeros(3)
    s = np.dot(volDirection, otherS-Scenter)
    mpoints = np.zeros(M_ACC.shape)

    # Change of frame for the detector corresponding to Scenter
    Detbis = np.zeros(Det.shape)
    Interbis = np.zeros((Det.shape[0], len(y), 3))  # Coordinates of the intersection curves in the new frame
    d0 = np.zeros((len(y), 3))  # Vector that corresponds to the direction giving a null phi-angle

    for j in range(Det.shape[1]):
        for i in range(Det.shape[0]):
            Detbis[i, j, :] = np.dot(volDirection, Det[i, j, :]-Scenter)
    for k in range(y.shape[0]):
        mpoints[k, :] += np.dot(volDirection, M_ACC[k, :]-Scenter)
        planebis = ComputePlaneEquation(center, s, mpoints[k, :])
        d0x = 0
        d0z = D
        d0y = -(planebis[0][0]*d0x+planebis[0][2]*d0z-planebis[1])/planebis[0][1]
        d0[k, :] += np.array([d0x, d0y, d0z])
        d0[k, :] /= np.linalg.norm(d0[k, :])
        for o in range(Det.shape[0]):
            Interbis[o, k, :] += np.dot(volDirection, np.array([x[o], y[k, o], z[o]])-Scenter)
    return Interbis, center, s, Detbis, d0, mpoints


def ComputeDifferentialWeights(center, s, Interbis, mpoint, gamma, D):
    oneovercos = np.zeros((Interbis.shape[0], Interbis.shape[1]))
    coeff_singularity = np.zeros((Interbis.shape[1], 3))  # sin, cos, offset coefficients
    for j in range(Interbis.shape[1]):
        v1p = center-mpoint[j, :]
        v0p = s - mpoint[j, :]
        # coeff computation
        Asin = v0p[1]/v0p[0]-v0p[2]*(v1p[1]*v0p[0]-v1p[0]*v0p[1])/(v0p[0]*(v1p[2]*v0p[0]-v1p[0]*v0p[2]))
        Acos = (v1p[1]*v0p[0]-v1p[0]*v0p[1])/(v1p[2]*v0p[0]-v1p[0]*v0p[2])
        B = mpoint[j, 1] - v0p[1]*mpoint[j, 0]/v0p[0] + (mpoint[j, 0]*v0p[2]-mpoint[j, 2]*v0p[0])*(v1p[1]*v0p[0]-v1p[0]*v0p[1])/(v0p[0]*(v1p[2]*v0p[0]-v1p[0]*v0p[2]))
        coeff_singularity[j, :] += np.array([Asin, Acos, B])
        for i in range(Interbis.shape[0]):
            dx = -D*np.cos(gamma[i])
            dz = D*np.sin(gamma[i])
            dy1 = v1p[1]*dx/v1p[0]
            dyn1 = -v1p[0]*dz + v1p[2]*dx
            dyn2 = v0p[0]*v1p[1]-v1p[0]*v0p[1]
            dyd = (v1p[2]*v0p[0]-v1p[0]*v0p[2])*v0p[0]
            dy = dy1 + dyn1*dyn2/dyd
            oneovercos[i, j] += np.sqrt(1+(dy/D)**2)
    return oneovercos, coeff_singularity


def ComputeAngularSamplingNewFrame(a0, a1, gamma_RTK):
    if np.abs(a1-a0) <= np.pi:
        gamma_center = np.abs(a1-a0)*0.5
        if a0 <= a1:
            gamma0 = -gamma_center - gamma_RTK
            gamma1 = gamma_center - gamma_RTK
        else:
            gamma0 = gamma_center - gamma_RTK
            gamma1 = -gamma_center - gamma_RTK
    else:
        gamma_center = (2*np.pi-np.abs(a1-a0))*0.5
        if a0 <= a1:
            gamma1 =  -gamma_center - gamma_RTK
            gamma0 =  gamma_center  - gamma_RTK 
        else:
            gamma0 = -gamma_center - gamma_RTK
            gamma1 = gamma_center  - gamma_RTK 
    return gamma0, gamma1


def ComputeNewFrameAndMPoints(Nx, DeltaY, D0, D1, geometry0, geometry1):
    volDir = ComputeNewFrame(geometry0, geometry1)
    # Axial coordinates
    y_min = np.max([D0[Nx//2, 0, 1], D1[Nx//2, 0, 1]])
    y_max = np.min([D0[Nx//2, -1, 1], D1[Nx//2, -1, 1]])
    y_dcc = np.linspace(y_min, y_max, round(np.abs(y_max-y_min)/DeltaY))
    # Radial coordinates
    t1, t2 = ComputeCylindersIntersection(geometry0, geometry1)
    if np.dot(t1, volDir[2, :]) >= 0:
        intersect = t1
    else:
        intersect = t2
    PMs = np.array([intersect[0]*np.ones(len(y_dcc)), y_dcc, intersect[2]*np.ones(len(y_dcc))])
    return volDir, PMs


def ComputeMomentsOnCylindricalDetectors(geometry0, geometry1, projection0, projection1):
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    projar0 = itk.GetArrayFromImage(projection0)
    projar1 = itk.GetArrayFromImage(projection1)
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    D = geometry0.GetSourceToDetectorDistances()[0]
    [DeltaX, DeltaY, DeltaZ] = projection0.GetSpacing()
    [Nx, Ny, Nz] = projection0.GetLargestPossibleRegion().GetSize()

    # Compute the two detectors coordinates RTK angles
    Det0, Det1, gamma_RTK = ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1)

    # Fonction qui calcul le nouveau repère et les angles correspondant
    volDir, PMs = ComputeNewFrameAndMPoints(Nx, DeltaY, Det0, Det1, geometry0, geometry1)
    gamma0, gamma1 = ComputeAngularSamplingNewFrame(a0, a1, gamma_RTK)

    # Compute all intersections in the canonical frame
    x1, z1, Inter1, x0, z0, Inter0, M_ACC = ComputeAllDetectorPlanesIntersections(geometry0, geometry1, Det0, Det1, PMs, DeltaY)

    # Change of frame for all the previous computations
    Interbis0, center0, s1bis, DETbis0, d00, PMs0 = ChangeOfFrameForAll(sourcePos0, sourcePos1, Det0, volDir, x0, z0, Inter0, D, M_ACC)
    Interbis1, center1, s0bis, DETbis1, d01, PMs1 = ChangeOfFrameForAll(sourcePos1, sourcePos0, Det1, volDir, x1, z1, Inter1, D, M_ACC)

    # Computation of the factor dphi/dgamma
    DiffWeights0, coeff_sing0 = ComputeDifferentialWeights(center0, s1bis, Interbis0, PMs0, gamma0, D)
    DiffWeights1, coeff_sing1 = ComputeDifferentialWeights(center1, s0bis, Interbis1, PMs1, gamma1, D)

    # Computation of the projection weights
    new_cos0 = np.zeros((Interbis0.shape[0], Interbis0.shape[1]))
    new_cos1 = np.zeros((Interbis0.shape[0], Interbis1.shape[1]))
    for j in range(Interbis0.shape[1]):
        new_cos0[:, j] = (D*np.cos(gamma0)*d00[j, 2]+Interbis0[:, j, 1]*d00[j, 1])/np.sqrt(D**2+Interbis0[:, j, 1]**2)
        new_cos1[:, j] = (D*np.cos(gamma1)*d01[j, 2]+Interbis1[:, j, 1]*d01[j, 1])/np.sqrt(D**2+Interbis1[:, j, 1]**2)

    # Axial interpolation of the projection at the different intersection points of the curve
    proj_interp0 = np.zeros((Det0.shape[0], Interbis0.shape[1]))
    proj_interp1 = np.zeros((Det0.shape[0], Interbis0.shape[1]))

    for i in range(Det0.shape[0]):
        f0 = interpolate.interp1d(DETbis0[i, :, 1], projar0[0, :, i], kind='linear', axis=0)
        proj_interp0[i, :] += f0(Interbis0[i, :, 1])
        f1 = interpolate.interp1d(DETbis1[i, :, 1], projar1[0, :, i], kind='linear', axis=0)
        proj_interp1[i, :] += f1(Interbis1[i, :, 1])

    m0 = np.sum(DiffWeights0*proj_interp0/new_cos0, axis=0)*np.abs(gamma_RTK[1]-gamma_RTK[0])
    m1 = np.sum(DiffWeights1*proj_interp1/new_cos1, axis=0)*np.abs(gamma_RTK[1]-gamma_RTK[0])

    return m0, m1


def ComputeSingularityForOneCurve(coeff_sing, d0, D):
    a = (d0[1]*coeff_sing[1]+d0[2])*D
    b = coeff_sing[0]*d0[1]*D
    c = d0[1]*coeff_sing[2]
    roots = np.array(np.roots([c-a, 2*b, a+c]))
    return 2*np.arctan(roots)


def ComputeMomentsOnCylindricalDetectorsWithSingularity(geometry0, geometry1, projection0, projection1):
    a0, a1 = geometry0.GetGantryAngles()[0], geometry1.GetGantryAngles()[0]
    projar0 = itk.GetArrayFromImage(projection0)
    projar1 = itk.GetArrayFromImage(projection1)
    sourcePos0, sourcePos1 = ExtractSourcePosition(geometry0, geometry1)
    D = geometry0.GetSourceToDetectorDistances()[0]
    [DeltaX, DeltaY, DeltaZ] = projection0.GetSpacing()
    [Nx, Ny, Nz] = projection0.GetLargestPossibleRegion().GetSize()

    # Compute the two detectors coordinates RTK angles
    Det0, Det1, gamma_RTK = ComputeDetectorsPosition(geometry0, geometry1, projection0, projection1)

    # Fonction qui calcul le nouveau repère et les angles correspondant
    volDir, PMs = ComputeNewFrameAndMPoints(Nx, DeltaY, Det0, Det1, geometry0, geometry1)
    gamma0, gamma1 = ComputeAngularSamplingNewFrame(a0, a1, gamma_RTK)

    # Compute all intersections in the canonical frame
    x1, z1, Inter1, x0, z0, Inter0, M_ACC = ComputeAllDetectorPlanesIntersections(geometry0, geometry1, Det0, Det1, PMs, DeltaY)

    # Change of frame for all the previous computations
    Interbis0, center0, s1bis, DETbis0, d00, PMs0 = ChangeOfFrameForAll(sourcePos0, sourcePos1, Det0, volDir, x0, z0, Inter0, D, M_ACC)
    Interbis1, center1, s0bis, DETbis1, d01, PMs1 = ChangeOfFrameForAll(sourcePos1, sourcePos0, Det1, volDir, x1, z1, Inter1, D, M_ACC)

    # Computation of the factor dphi/dgamma
    DiffWeights0, coeff_sing0 = ComputeDifferentialWeights(center0, s1bis, Interbis0, PMs0, gamma0, D)
    DiffWeights1, coeff_sing1 = ComputeDifferentialWeights(center1, s0bis, Interbis1, PMs1, gamma1, D)

    # Axial interpolation of the projection at the different intersection points of the curve
    proj_interp0 = np.zeros((Det0.shape[0], Interbis0.shape[1]))
    proj_interp1 = np.zeros((Det0.shape[0], Interbis0.shape[1]))

    for i in range(Det0.shape[0]):
        f0 = interpolate.interp1d(DETbis0[i, :, 1], projar0[0, :, i], kind='linear', axis=0)
        proj_interp0[i, :] += f0(Interbis0[i, :, 1])
        f1 = interpolate.interp1d(DETbis1[i, :, 1], projar1[0, :, i], kind='linear', axis=0)
        proj_interp1[i, :] += f1(Interbis1[i, :, 1])

    m0 = np.zeros((Interbis0.shape[1]))
    m1 = np.zeros((Interbis1.shape[1]))

    for j in range(Interbis1.shape[1]):
        v_gamma = D*(coeff_sing0[j, 0]*np.sin(gamma0) + coeff_sing0[j, 1]*np.cos(gamma0)) + coeff_sing0[j, 2]
        sing0 = ComputeSingularityForOneCurve(coeff_sing0[j, :], d00[j, :], D)
        if CanWeApplyDirectlyTheFormula(gamma0, sing0[np.where(sing0 < 0)][0]):
            new_cos0 = (D*np.cos(gamma0)*d00[j, 2]+Interbis0[:, j, 1]*d00[j, 1])/np.sqrt(D**2+Interbis0[:, j, 1]**2)
            m0[j] += np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp0[:, j]*DiffWeights0[:, j]/new_cos0)
        else:
            if np.abs(a1-a0) <= np.pi:
                m0[j] += TrapIntegration(sing0[np.where(sing0 < 0)][0], gamma0, proj_interp0[:, j], DiffWeights0[:, j], v_gamma, d00[j], D)
            else:
                m0[j] += TrapIntegration(np.pi+sing0[np.where(sing0 < 0)][0], gamma0, proj_interp0[:, j], DiffWeights0[:, j], v_gamma, d00[j], D)
        v_gamma = D*(coeff_sing1[j, 0]*np.sin(gamma1) + coeff_sing1[j, 1]*np.cos(gamma1)) + coeff_sing1[j, 2]
        sing1 = ComputeSingularityForOneCurve(coeff_sing1[j, :], d01[j, :], D)
        if CanWeApplyDirectlyTheFormula(gamma1, sing1[np.where(sing1 > 0)][0]):
            new_cos1 = (D*np.cos(gamma1)*d01[j, 2]+Interbis1[:, j, 1]*d01[j, 1])/np.sqrt(D**2+Interbis1[:, j, 1]**2)
            m1[j] += np.abs(gamma_RTK[1]-gamma_RTK[0])*np.sum(proj_interp1[:, j]*DiffWeights1[:, j]/new_cos1)
        else:
            if np.abs(a1-a0) <= np.pi:
                m1[j] += TrapIntegration(sing1[np.where(sing1 > 0)][0], gamma1, proj_interp1[:, j], DiffWeights1[:, j], v_gamma, d01[j], D)
            else:
                m1[j] += TrapIntegration(np.pi+sing1[np.where(sing1 > 0)][0], gamma1, proj_interp1[:, j], DiffWeights1[:, j], v_gamma, d01[j], D)

    return m0, m1


def CanWeApplyDirectlyTheFormula(angle, xs):
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


def TestForSingularity(angle, xs):
    a = 0
    b = 0
    if len(np.where(angle >= xs)[0]) != len(angle) and len(np.where(angle <= xs)[0]) != len(angle):
        a = np.where(angle <= xs)[-1][0]
        b = np.where(angle >= xs)[-1][-1]
    return np.min((a, b)), np.max((a, b))


def TrapIntegration(xs, gamma, ar, DiffWeights, Interbis, d0, D):
    h = np.abs(gamma[1]-gamma[0])
    f = interpolate.interp1d(gamma, ar)
    g = ar/(D*np.cos(gamma)*d0[2]+Interbis*d0[1])*np.sqrt(D**2+Interbis**2)
    DW_interp = interpolate.interp1d(gamma, DiffWeights)
    v_interp = interpolate.interp1d(gamma, Interbis)
    ii, iii = TestForSingularity(gamma, xs)

    if ii == 0 and iii == 1:
        hh = np.abs(xs-gamma[ii])
        summ = h*(DiffWeights[1]*g[1]/2 + np.sum(g[2:-1]*DiffWeights[2:-1]) + DiffWeights[-1]*g[-1]/2)
        sing = -(ar[ii]-f(xs-hh))/2
        mida = (gamma[iii]+h/2 + xs-hh/2)/2
        dmida = (xs-hh/2) - (gamma[iii]+h/2)
        midb = (gamma[ii]+xs+hh/2)/2
        dmidb = gamma[ii]-(xs+hh/2)
        rest = dmida*f(mida)*np.sqrt(D**2+v_interp(mida)**2)/(D*np.cos(mida)*d0[2] + v_interp(mida)*d0[1]) + dmidb*f(midb)*np.sqrt(D**2+v_interp(midb)**2)/(D*np.cos(midb)*d0[2]+v_interp(midb)*d0[1])
#             rest=0
    elif (ii == np.size(gamma)-2 and iii == np.size(gamma)-1):
#         print('N-2,N-1')
        hh = np.abs(xs - gamma[iii])
        sing = -(f(xs+hh)-ar[iii])/2
        mida = (gamma[ii]-h/2 + xs + hh/2)/2
        dmida = gamma[ii]-h/2 - (xs + hh/2)
        midb = (xs-hh/2 + gamma[iii])/2
        dmidb = (xs-hh/2 - gamma[iii])
        summ = h*(DiffWeights[0]*g[0]/2 + np.sum(g[1:-2]*DiffWeights[1:-2]) + DiffWeights[-2]*g[-2]/2)
        rest = dmida*f(mida)*np.sqrt(D**2+v_interp(mida)**2)/(D*np.cos(mida)*d0[2] + v_interp(mida)*d0[1]) + dmidb*f(midb)*np.sqrt(D**2+v_interp(midb)**2)/(D*np.cos(midb)*d0[2]+v_interp(midb)*d0[1])
    else:
        if np.abs(gamma[ii]-xs) <= np.abs(gamma[iii]-xs):
            hh = 2*np.abs(xs-(gamma[ii]))
            summ = h*(np.sum(g[1:-1]*DiffWeights[1:-1])+(g[0]*DiffWeights[0]+g[-1]*DiffWeights[-1])/2-(g[ii]*DiffWeights[ii]+g[iii]*DiffWeights[iii])/2)
            hrest = ((xs-hh/2)-(gamma[iii]))
            amp = f(xs-hh/2)*np.sqrt(D**2+v_interp(xs-hh/2)**2)/(D*np.cos(xs-hh/2)*d0[2]+v_interp(xs-hh/2)*d0[1])
            rest = hrest*(amp*DW_interp(xs-hh/2)+g[iii]*DiffWeights[iii])/2
        else:
            hh = 2*np.abs(xs-(gamma[iii]))
            summ = h*(np.sum(g[1:-1]*DiffWeights[1:-1])+(g[0]*DiffWeights[0]+g[-1]*DiffWeights[-1])/2-(g[ii]*DiffWeights[ii]+g[iii]*DiffWeights[iii])/2)
            hrest = ((gamma[ii])-(xs+hh/2))
            amp = f(xs+hh/2)*np.sqrt(D**2+v_interp(xs+hh/2)**2)/(D*np.cos(xs+hh/2)*d0[2]+v_interp(xs+hh/2)*d0[1])
            rest = hrest*(amp*DW_interp(xs+hh/2)+g[ii]*DiffWeights[ii])/2
#         sing = -(f(xs+hh)-f(xs-hh))/2
        sing = 0
        # a = (ar[iii]-ar[ii])/(gamma[iii]-gamma[ii])
        # b = ar[ii]-a*gamma[ii]
        # rest = (ar[iii]-ar[ii])+b*np.log(np.abs(gamma[iii]/gamma[ii]))

    return summ+rest+sing
