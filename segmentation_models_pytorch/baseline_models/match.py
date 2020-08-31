import numpy as np
import cv2
import math


# based on code here: https://github.com/fierval/retina/blob/master/DiabeticRetinopathy/Refactoring/kobra/imaging.py
def matched_filter(image, length = 9, sigma = 1.5, w = 31, coefficient = 1.5):
    def _filter_kernel_mf_fdog(length, sigma, t = 3, mf = True):
        dim_y = int(length)
        dim_x = int(2 * t * sigma)
        if dim_x == 0:
            dim_x = 1
        if dim_y == 0:
            dim_y = 1
        arr = np.zeros((dim_y, dim_x), 'f')
        ctr_x = dim_x / 2
        ctr_y = int(dim_y / 2.)

        # an un-natural way to set elements of the array
        # to their x coordinate.
        # x's are actually columns, so the first dimension of the iterator is used
        it = np.nditer(arr, flags=['multi_index'])
        while not it.finished:
            arr[it.multi_index] = it.multi_index[1] - ctr_x
            it.iternext()

        two_sigma_sq = 2 * sigma * sigma
        if two_sigma_sq == 0:
            two_sigma_sq = 1
        div = math.sqrt(2 * math.pi) * sigma
        if div == 0:
            div = 1.
        sqrt_w_pi_sigma = 1. / div
        if not mf:
            div = sigma ** 2
            if div == 0:
                div = 1.
            sqrt_w_pi_sigma = sqrt_w_pi_sigma / div

        #@vectorize(['float32(float32)'], target='cpu')
        def k_fun(x):
            return sqrt_w_pi_sigma * np.exp(-x * x / two_sigma_sq)
        #@vectorize(['float32(float32)'], target='cpu')
        def k_fun_derivative(x):
            return -x * sqrt_w_pi_sigma * np.exp(-x * x / two_sigma_sq)

        if mf:
            kernel = k_fun(arr)
            kernel = kernel - np.sum(kernel)/(dim_x*dim_y)
        else:
           kernel = k_fun_derivative(arr)

        # return the "convolution" kernel for filter2D
        return kernel
    def createMatchedFilterBank(kernel, n = 12):
        rotate = 180 / n
        center = (kernel.shape[1] / 2, kernel.shape[0] / 2)
        cur_rot = 0
        kernels = [kernel]

        for i in range(1, n):
            cur_rot += rotate
            r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
            k = cv2.warpAffine(kernel, r_mat, (kernel.shape[1], kernel.shape[0]))
            if np.count_nonzero(k):
                mean = np.sum(k)/np.count_nonzero(k)
            else:
                mean = 0
            for y in range(len(k)):
                for x in range(len(k[0])):
                    if k[y][x]:
                        k[y][x] -= mean
            kernels.append(k)
        return kernels
    def applyFilters(img, kernels):
        images = np.array([cv2.filter2D(img, -1, k) for k in kernels])
        return np.max(images, 0)
    def inbounds(shape, indices):
        '''
        Test if the given coordinates inside the given image.
        The first input parameter is the shape of image (height, weight) and the
        second parameter is the coordinates to be tested (y, x)
        The function returns True if the coordinates inside the image and vice versa.
        '''
        assert len(shape) == len(indices)
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= shape[i]:
                return False
        return True
    def setlable(img, labimg, x, y, label, size):
        '''
        This fucntion is used for label image.
        The first two input images are the image to be labeled and an output image with
        labeled region. "x", "y" are the coordinate to be tested, "label" is the ID
        of a region and size is used to limit maximum size of a region.
        '''
        if img[y][x] and not labimg[y][x]:
            labimg[y][x] = label
            size += 1
            if size > 500:
                    return False
            if inbounds(img.shape, (y, x+1)):
                setlable(img, labimg, x+1, y,label, size)
            if inbounds(img.shape, (y+1, x)):
                setlable(img, labimg, x, y+1,label, size)
            if inbounds(img.shape, (y, x-1)):
                setlable(img, labimg, x-1, y,label, size)
            if inbounds(img.shape, (y-1, x)):
                setlable(img, labimg, x, y-1,label, size)
            if inbounds(img.shape, (y+1, x+1)):
                setlable(img, labimg, x+1, y+1,label, size)
            if inbounds(img.shape, (y+1, x-1)):
                setlable(img, labimg, x-1, y+1,label, size)
            if inbounds(img.shape, (y-1, x+1)):
                setlable(img, labimg, x+1, y-1,label, size)
            if inbounds(img.shape, (y-1, x-1)):
                setlable(img, labimg, x-1, y-1,label, size)

    height, weight = image.shape[:2]
    image = 255 - image

    # generate Gaussian filter and first-order derivative of Gaussian filter.
    #fdog = matched.fdog_filter_kernel()
    gf = _filter_kernel_mf_fdog(length, sigma)
    fdog = _filter_kernel_mf_fdog(length, sigma, mf = False)

    # generate filter bank
    bank_gf = createMatchedFilterBank(kernel = gf)
    bank_fdog = createMatchedFilterBank(kernel = fdog)

    # obtain matched filter response. H is the MFR-G and D is MFR-FDoG
    H = applyFilters(image, bank_gf)
    D = applyFilters(image, bank_fdog)

    # compute the threshold value using MFR-FDoG
    kernel_size = 31
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)
    dm = np.zeros(D.shape,np.float32)
    DD = np.array(D, dtype='f')
    dm = cv2.filter2D(DD,-1,kernel)
    dmn = cv2.normalize(dm, 0, 1, cv2.NORM_MINMAX)
    uH = cv2.mean(H)
    Tc = coefficient * uH[0]
    T = (1+dmn) * Tc

    # threshold the MFR-G with previous threhshold value.
    out = np.zeros(H.shape)
    out[H > T] = 255

    # get rid of the segment less than 10 pixel
    lab = 1
    label = np.zeros(out.shape)
    for y in range(height):
        for x in range(weight):
            if not label[y][x] and out[y][x]:
                size = 0
                setlable(out, label, x, y, lab, size)
                lab += 1
    num = np.zeros(lab)
    for y in range(height):
        for x in range(weight):
            num[int(label[y][x]-1)] += 1
    for y in range(height):
        for x in range(weight):
            if num[int(label[y][x]-1)] <= 10:
                out[y][x] = 0

    return out/255.0
