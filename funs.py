################################################
# Functions for Subsea pipe case. Phantom and CT acquisition geometry.
# By Silja L. Christensen
# April 2024
################################################

import numpy as np

#%%
#=======================================================================
# Phantom
#=========================================================================

def DeepSeaOilPipe8(N,defects):

    radii  = np.array([9,11,16,17.5,23])

    domain = 55
    c = np.round(np.array([N/2,N/2]))
    axis1 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    axis2 = np.linspace(-c[0]-1,N-c[0],N, endpoint=True)
    x, y = np.meshgrid(axis1,axis2)
    center = np.array([0,0])
    phantom = 2e-2*7.9*drawPipe(N,domain,x,y,center,center,radii[0],radii[1])      # Steel (8.05g/cm^3)
    phantom = phantom+5.1e-2*0.15*drawPipe(N,domain,x,y,center,center,radii[1],radii[2])      # PE-foam
    phantom = phantom+5.1e-2*0.94*drawPipe(N,domain,x,y,center,center,radii[2],radii[3])     # PU rubber      0.93-0.97 g / cm^3 (Might be PVC, 1400 kg /m^3)
    phantom = phantom+4.56e-2*2.3*drawPipe(N,domain,x,y,center,center,radii[3],radii[4])    # Concrete 2.3 g/cm^3

    # radial cracks
    if defects == True:

        defectmask = []
        vertices = []

        # radial and angular cracks
        no = 12
        ang = np.array([-3*np.pi/9, -2*np.pi/9, -np.pi/9, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2, 2*np.pi/3, 5*np.pi/4-np.pi/9, 5*np.pi/4, 5*np.pi/4+np.pi/9])-60/180*np.pi
        dist = np.array([20.25, 20.25, 20.25, 20.25, 20.25, 16.75, 13.5, 10, 20.25, 16.75+2, 16.75, 16.75-2])/domain*N
        w = np.array([0.5, 0.4, 0.3, 0.2, 4, 4, 4, 4, 0.4, 0.4, 0.4, 0.4])/domain*N
        l = np.array([4, 4, 4, 4, 0.4, 0.4, 0.4, 0.4, 4, 4, 4, 4])/domain*N
        vals = np.zeros(no)
        vals[8] = 2e-2*7.9
        for i in range(no):
            # coordinates in (x,y), -1 to 1 system
            coordinates0 = np.array([
                [c[0]+w[i]/2, c[1]+dist[i] + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist[i] + l[i]/2],
                [c[0]-w[i]/2, c[1]+dist[i] - l[i]/2],
                [c[0]+w[i]/2, c[1]+dist[i] - l[i]/2]
            ])
            R = np.array([
                [np.cos(ang[i]), -np.sin(ang[i])],
                [np.sin(ang[i]), np.cos(ang[i])]
                ])
            # Rotate around image center
            coordinates = R @ (coordinates0.T - np.array([[c[0]],[c[1]]])) + np.array([[c[0]],[c[1]]])
            coordinates = coordinates.T

            # transform into (row, column) indicies
            vertices.append(np.ceil(np.fliplr(coordinates)))
            # create mask
            tmpmask = create_polygon([N,N], vertices[i])
            defectmask.append(np.array(tmpmask, dtype=bool))
            phantom[defectmask[i]] = vals[i]

        # Cross
        c_cross_ang = -np.pi/2
        c_cross_dist = 20.25/domain*N
        c_cross = c_cross_dist*np.array([np.cos(c_cross_ang), np.sin(c_cross_ang)])+N/2
        #np.array([c[1]-20.25/domain*N, c[0]])
        a = (2/np.sqrt(2))/domain*N
        b = (0.2/np.sqrt(2))/domain*N
        coordinates_cross1 = np.array([
            [c_cross[0]-a+b, c_cross[1]-a],
            [c_cross[0]+a, c_cross[1]+a-b],
            [c_cross[0]+a-b, c_cross[1]+a],
            [c_cross[0]-a, c_cross[1]-a+b]])
        coordinates_cross2 = np.array([
            [c_cross[0]+a-b, c_cross[1]-a],
            [c_cross[0]+a, c_cross[1]-a+b],
            [c_cross[0]-a+b, c_cross[1]+a],
            [c_cross[0]-a, c_cross[1]+a-b]])
        # transform into (row, column) indicies
        vertices.append(np.ceil(np.flipud(coordinates_cross1)))
        # create mask
        tmpmask = create_polygon([N,N], vertices[12])
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[12]] = 0
        # transform into (row, column) indicies
        vertices.append(np.ceil(np.flipud(coordinates_cross2)))
        # create mask
        tmpmask = create_polygon([N,N], vertices[13])
        defectmask.append(np.array(tmpmask, dtype=bool))
        phantom[defectmask[13]] = 0

        # Circles
        ang_circ = np.array([3*np.pi/4+np.pi/9, 3*np.pi/4+np.pi/9, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4-np.pi/9])-60/180*np.pi
        dist_circ = 20.25/domain*N
        siz = np.array([1, 0.3, 1, 0.3, 0.3])/domain*N
        val = np.array([0, 2e-2*7.9, 0, 4.56e-2*2.3, 2e-2*7.9])

        for i in range(len(ang_circ)):
            tmpmask = ((x-np.cos(ang_circ[i])*dist_circ)**2 + (y-np.sin(ang_circ[i])*dist_circ)**2 <= siz[i]**2)
            defectmask.append(np.array(tmpmask, dtype=bool))
            phantom[defectmask[14+i]] = val[i]

        center_dists = np.hstack([dist, c_cross_dist, dist_circ*np.ones(3)])
        center_x = center_dists*np.hstack([np.sin(-ang), np.sin(np.array([c_cross_ang])), np.cos(ang_circ[np.array([0,2,4])])])+N/2
        center_y = center_dists*np.hstack([np.cos(-ang), np.cos(np.array([c_cross_ang])), np.sin(ang_circ[np.array([0,2,4])])])+N/2
        centers = np.vstack([center_x, center_y])
        
        
        return phantom, radii, defectmask, vertices, centers
    else:
        return phantom, radii
    
def drawPipe(N, domain, x,y ,c1,c2, r1, r2):
    # N is number of pixels on one axis
    # domain is true size of one axis
    # x and y is a meshgrid of the domain
    # r1 and r2 are the inner and outer radii of the pipe layer
    R1 = r1/domain*N
    R2 = r2/domain*N

    M1 = (x-c1[0]/domain*N)**2+(y-c1[1]/domain*N)**2>=R1**2
    M2 = (x-c2[0]/domain*N)**2+(y-c2[1]/domain*N)**2<=R2**2

    return np.logical_and(M1, M2)

def check(p1, p2, base_array):
    """
    Source: https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array

#%%
#=======================================================================
# Acquisition geometry
#=========================================================================

def geom_Data20180911(size):
    
    offset      = 0             # angular offset
    shift       = -12.5           # source offset from center
    stc         = 60               # source to center distance
    ctd         = 50               # center to detector distance
    det_full    = 512
    startAngle  = 0
    if size == "sparseangles":
        p   = 510               # p: number of detector pixels
        q   = 36                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    if size == "sparseangles20percent":
        p   = 510               # p: number of detector pixels
        q   = 72                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    if size == "sparseangles50percent":
        p   = 510               # p: number of detector pixels
        q   = 180                # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    elif size == "full":
        p   = 510               # p: number of detector pixels
        q   = 360               # q: number of projection angles
        maxAngle    = 360               # measurement max angle
    elif size == "overfull":
        p   = 510               # p: number of detector pixels
        q   = 720               # q: number of projection angles
        maxAngle    = 364               # measurement max angle
    elif size == "limited90":
        p   = 510               # p: number of detector pixels
        q   = 90               # q: number of projection angles
        startAngle = 15
        maxAngle = 105
    elif size == "limited120":
        p   = 510               # p: number of detector pixels
        q   = 120               # q: number of projection angles
        maxAngle = 120
    elif size == "limited180":
        p   = 510               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        startAngle = 15
        maxAngle = 195
    elif size == "limited180_2":
        p   = 510               # p: number of detector pixels
        q   = 180               # q: number of projection angles
        startAngle = 180
        maxAngle = 360

    dlA         = 41.1*(p/det_full)              # full detector length
    dl          = dlA/p   # length of detector element

    # view angles in rad
    theta = np.linspace(startAngle, maxAngle, q, endpoint=False) 
    theta = theta/180*np.pi
    
    s0 = np.array([shift, -stc])
    d0 = np.array([shift, ctd])
    u0 = np.array([dl, 0])

    vectors = np.empty([q, 6])
    for i, val in enumerate(theta):
        R = np.array([[np.cos(val), -np.sin(val)], [np.sin(val), np.cos(val)]])
        s = R @ s0
        d = R @ d0
        u = R @ u0
        vectors[i, 0:2] = s
        vectors[i, 2:4] = d
        vectors[i, 4:6] = u

    return p, theta, stc, ctd, shift, vectors, dl, dlA
