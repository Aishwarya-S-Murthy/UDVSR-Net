import numpy as np
import math
from scipy import ndimage
import cv2

# ==============================================================================
# UCIQE Implementation
# ==============================================================================

def calculate_uciqe(img1, img2=None, crop_border=0, **kwargs):
    """
    Calculate UCIQE (Underwater Color Image Quality Evaluation).
    No-reference metric. img2 (Ground Truth) is ignored.
    """
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    # Ensure image is uint8
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255.0).astype(np.uint8)

    # Convert BGR to CIELab
    # Assuming input is BGR as per tensor2img default in BasicSR
    lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # 1. Variance of Chroma
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)

    # 2. Contrast of Luminance (Difference between top 1% and bottom 1%)
    l_flatten = l.flatten()
    l_flatten.sort()
    idx = int(len(l_flatten) * 0.01)
    if idx == 0:
        con_l = np.max(l_flatten) - np.min(l_flatten)
    else:
        con_l = np.mean(l_flatten[-idx:]) - np.mean(l_flatten[:idx])

    # 3. Average of Saturation
    saturation = chroma / (l + 1e-6) # Add epsilon to avoid division by zero
    mu_s = np.mean(saturation)

    # UCIQE equation with standard weights
    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    
    return float(uciqe)


# ==============================================================================
# UIQM Implementation (Asymmetric Alpha-Trimmed + EME)
# ==============================================================================

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """Calculates the asymmetric alpha-trimmed mean"""
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    weight = (1/(K-T_a_L-T_a_R))
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """Enhancement measure estimation"""
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    """Underwater Image Sharpness Measure"""
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def _uiconm(x, window_size):
    """Underwater image contrast measure"""
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
    return w*val

def getUIQM(x):
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm

# ==============================================================================
# BasicSR Wrapper for UIQM
# ==============================================================================

def calculate_uiqm(img1, img2=None, crop_border=0, **kwargs):
    """
    Calculate UIQM.
    BasicSR wrapper for the provided UIQM implementation.
    """
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    # Ensure image is in 0-255 range
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255.0).astype(np.uint8)

    # BasicSR's tensor2img outputs BGR format, but the getUIQM function 
    # explicitly expects RGB (R = x[:,:,0], etc.).
    # We flip the channels from BGR to RGB here before passing it in.
    img_rgb = img1[..., ::-1]
    
    uiqm_val = getUIQM(img_rgb)
    return float(uiqm_val)