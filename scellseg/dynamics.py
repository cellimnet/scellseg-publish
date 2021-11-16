import time, os

from scipy import ndimage
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage import measurements
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
from numba import njit, float32, int32, vectorize
from . import utils, metrics
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed

try:
    import torch
    from torch import optim, nn
    from . import resnet_torch
    TORCH_ENABLED = True 
    torch_GPU = torch.device('cuda')
except:
    TORCH_ENABLED = False

@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)

    Parameters
    --------------

    T: float64, array
        _ x Lx array that diffusion is run in

    y: int32, array
        pixels in y inside mask

    x: int32, array
        pixels in x inside mask

    ymed: int32
        center of mask in y

    xmed: int32
        center of mask in x

    Lx: int32
        size of x-dimension of masks

    niter: int32
        number of iterations to run diffusion

    Returns
    ---------------

    T: float64, array
        amount of diffused particles at each pixel

    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T

def labels_to_flows(labels, files=None):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell probability, flows[k][2] is Y flow, and flows[k][3] is X flow

    """

    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3:
        # print('NOTE: computing flows for labels (could be done before to save time)')
        # compute flows        
        veci = [masks_to_flows(labels[n][0])[0] for n in range(nimg)]
        # concatenate flows with cell probability
        flows = [np.concatenate((labels[n][[0]], labels[n][[0]]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        # print('flows precomputed')  # TODO：这里可以修改一下，做一个用户反馈机制
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows

def masks_to_flows(masks):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if masks.ndim > 2:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows(masks[z])[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows(masks[:,y])[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows(masks[:,:,x])[0]
            mu[[0,1], :, :, x] += mu0
        return mu, None

    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    
    nmask = masks.max()
    slices = scipy.ndimage.find_objects(masks)
    dia = utils.diameters(masks)[0]
    s2 = (.15 * dia)**2
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y,x = np.nonzero(masks[sr, sc] == (i+1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
            xmed = x[imin]
            ymed = y[imin]

            d2 = (x-xmed)**2 + (y-ymed)**2
            mu_c[sr.start+y-1, sc.start+x-1] = np.exp(-d2/s2)

            niter = 2*np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly+2)*(lx+2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), niter)
            T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])

            dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx = T[y*lx + x+1] - T[y*lx + x-1]
            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((dy,dx))

    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    return mu, mu_c

@njit(['(int16[:,:,:],float32[:], float32[:], float32[:,:])',
        '(float32[:,:,:],float32[:], float32[:], float32[:,:])'], cache=False)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1= min(Ly-1, yf+1)
        xf1= min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )

def steps2D_interp(p, dP, niter, use_gpu=False):
    shape = dP.shape[1:]
    if use_gpu and TORCH_ENABLED:
        device = torch_GPU
        pt = torch.from_numpy(p[[1,0]].T).double().to(device)
        pt = pt.unsqueeze(0).unsqueeze(0)
        pt[:,:,:,0] = (pt[:,:,:,0]/(shape[1]-1)) # normalize to between  0 and 1
        pt[:,:,:,1] = (pt[:,:,:,1]/(shape[0]-1)) # normalize to between  0 and 1
        pt = pt*2-1                       # normalize to between -1 and 1
        im = torch.from_numpy(dP[[1,0]]).double().to(device)
        im = im.unsqueeze(0)
        for k in range(2):
            im[:,k,:,:] /= (shape[1-k]-1) / 2.
        for t in range(niter):
            dPt = torch.nn.functional.grid_sample(im, pt)
            for k in range(2):
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] - dPt[:,k,:,:], -1., 1.)
        pt = (pt+1)*0.5
        pt[:,:,:,0] = pt[:,:,:,0] * (shape[1]-1)
        pt[:,:,:,1] = pt[:,:,:,1] * (shape[0]-1)
        return pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
    else:
        dPt = np.zeros(p.shape, np.float32)
        for t in range(niter):
            map_coordinates(dP, p[0], p[1], dPt)
            p[0] = np.minimum(shape[0]-1, np.maximum(0, p[0] - dPt[0]))
            p[1] = np.minimum(shape[1]-1, np.maximum(0, p[1] - dPt[1]))
        return p

@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 3D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 4D array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

    dP: float32, 4D array
        flows [axis x Lz x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 3]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 4D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j,0]
            y = inds[j,1]
            x = inds[j,2]
            p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
            p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] - dP[0,p0,p1,p2]))
            p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] - dP[1,p0,p1,p2]))
            p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] - dP[2,p0,p1,p2]))
    return p

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            p[0,y,x] = min(shape[0]-1, max(0, p[0,y,x] - dP[0,p0,p1]))
            p[1,y,x] = min(shape[1]-1, max(0, p[1,y,x] - dP[1,p0,p1]))
    return p

def follow_flows(dP, niter=200, interp=True, use_gpu=False):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)
    if len(shape)>2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        #inds = np.array(np.nonzero(dP[0]!=0)).astype(np.int32).T
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        if not interp:
            p = steps2D(p, dP, inds, niter)
        else:
            p[:,inds[:,0],inds[:,1]] = steps2D_interp(p[:,inds[:,0], inds[:,1]], 
                                                      dP, niter, use_gpu=use_gpu)
    return p

def remove_bad_flow_masks(masks, flows, threshold=0.4):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ = metrics.flow_error(masks, flows)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 

    Parameters
    ----------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].

    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.

    rpad: int (optional, default 20)
        histogram edge padding

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)

    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.

    Returns
    ---------------

    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]
    
    # remove big masks
    # print('shape0', shape0)
    # _,counts = np.unique(M0, return_counts=True)
    # big = np.prod(shape0) * 0.6  # change 0.4 to 0.6
    # for i in np.nonzero(counts > big)[0]:
    #     M0[M0==i] = 0
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if threshold is not None and threshold > 0 and flows is not None:
        M0 = remove_bad_flow_masks(M0, flows, threshold=threshold)
        _,M0 = np.unique(M0, return_inverse=True)
        M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0


def labels_to_hovers(labels, files=None):
    """ convert labels (list of masks or hovers) to hovers for training model

    if files is not None, hovers are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that hovers were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create hovers and cell probabilities.

    Returns
    --------------

    hovers: list of [4 x Ly x Lx] arrays
        hovers[k][0] is labels[k], hovers[k][1] is cell probability, hovers[k][2] is Y hover, and hovers[k][3] is X hover

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3:
        # print('NOTE: computing flows for labels (could be done before to save time) --- hover')
        # compute flows
        veci = [masks_to_hovers(labels[n][0]) for n in range(nimg)]
        # concatenate flows with cell probability
        hovers = [np.concatenate((labels[n][[0]], labels[n][[0]]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(hovers, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_hovers.tif', flow)
    else:
        # print('flows precomputed')
        hovers = [labels[n].astype(np.float32) for n in range(nimg)]
    return hovers

def masks_to_hovers(masks):
    """Input masks must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """

    if masks.ndim > 2:
        Lz, Ly, Lx = masks.shape
        hv_map = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            hv_map0 = masks_to_hovers(masks[z])
            hv_map[[1,2], z] += hv_map0
        for y in range(Ly):
            hv_map0 = masks_to_hovers(masks[:,y])
            hv_map[[0,2], :, y] += hv_map0
        for x in range(Lx):
            hv_map0 = masks_to_hovers(masks[:,:,x])
            hv_map[[0,1], :, :, x] += hv_map0
        return hv_map

    orig_mask = masks.copy()  # instance ID map
    orig_mask = np.pad(orig_mask, [[2, 2], [2, 2]], mode='constant')  # first pad the mask
    # re-cropping with fixed instance id map

    x_map = np.zeros(orig_mask.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_mask.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(orig_mask))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(orig_mask == inst_id, np.uint8)  # get single instance
        inst_box = get_bounding_box(inst_map) # bbox

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]  # crop

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([y_map[2:-2, 2:-2], x_map[2:-2, 2:-2]])
    hover = np.transpose(hv_map, (2, 0, 1))
    return hover

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def get_masks_watershed(pred, cellprob_threshold=0.5, min_size=15):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed Y-map
              channel 2 containing the regressed X-map

    """
    pred = np.array(pred, dtype=np.float32)

    h_dir_raw = pred[..., 0]
    v_dir_raw = pred[..., 1]
    cellprob = pred[..., 2]

    # processing
    blb = np.array(cellprob >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=min_size)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)  # minus background
    overall[overall < 0] = 0
    overall = np.array(overall >= 0.6, dtype=np.int32)  # get masker
    marker = blb - overall  # get inner
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)  # open
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=min_size)

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    proced_pred = watershed(dist, markers=marker, mask=blb)
    pred_masks = np.zeros_like(proced_pred)
    proced_ids = np.unique(proced_pred)
    for proced_id in range(1, len(proced_ids)):  # 0 donate backgound
        proced_id_0 = proced_ids[proced_id]
        pred_masks[proced_pred == proced_id_0] = proced_id

    return pred_masks