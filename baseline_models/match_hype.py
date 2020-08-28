import numpy as np
import cv2
import math
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix


# based on code here: https://github.com/fierval/retina/blob/master/DiabeticRetinopathy/Refactoring/kobra/imaging.py
def matched_filter(img, target,mask = None, length = 9, sigma = 1.5, w = 31, coefficient = 1.5 ):
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

    #print(img.shape)
    #print(type(img))
    image = img[:,:,1] #use green channelheight, wei
    height, weight = image.shape[:2]
    image = 255 - image
    if mask is not None:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

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


    # using the mask image to truncate the value outside the reina.
    if mask is not None:
        laplacian = cv2.Laplacian(mask, cv2.CV_64F)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        laplacian = cv2.dilate(laplacian,kernel,iterations = 4)
        H[(mask == 0) + (laplacian != 0)] = 0
        out[(mask == 0) + (laplacian != 0)] = 0

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

    prediction_flat = out.flatten() / 255
    prediction_flat = prediction_flat.astype(int)

    prediction_flat = np.array(prediction_flat)
    #target = cv2.imread(sys.argv[2])
    target_flat = target[:,:,0] / 255
    #target_flat = np.round(target.flatten())
    target_flat = target_flat.astype(int)
    target_flat = target_flat.flatten()

    auc_score = roc_auc_score(target_flat, prediction_flat)

    rounded_prediction_flat = np.round(prediction_flat)
    (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                                  rounded_prediction_flat,
                                                                                  average='binary')
    tn, fp, fn, tp = confusion_matrix(target_flat, rounded_prediction_flat).ravel()
    kappa = cohen_kappa_score(target_flat, rounded_prediction_flat)
    #acc = float(tp + tn) / float(tp + tn + fp + fn)
    #specificity = float(tn) / float(tn + fp)

    #print("tn",type(tn),tn)
    #print("fp",type(fp),fp)
    #print("fn",type(fn),fn)
    #print("tp",type(tp),tp)
    #print("acc",type(acc),acc)
    #print("specificity",type(specificity),specificity)

    # generate the output images.
    #cv2.imwrite("Final_01.jpg", out)
    #cv2.imwrite("MF_01.jpg" , H)
    return out, tn, fp, fn, tp

img1 = np.array(Image.open("/Users/scalzo/Desktop/vessel-seg/data/DSA/1.jpg"))
img2 = np.array(Image.open("/Users/scalzo/Desktop/vessel-seg/data/DSA/2.jpg"))
img3 = np.array(Image.open("/Users/scalzo/Desktop/vessel-seg/data/DSA/3.jpg"))

target1 = np.array(Image.open("/Users/scalzo/Desktop/vessel-seg/data/DSA/1_target.jpg"))
target2 = np.array(Image.open("/Users/scalzo/Desktop/vessel-seg/data/DSA/2_target.jpg"))
target3 = np.array(Image.open("/Users/scalzo/Desktop/vessel-seg/data/DSA/3_target.jpg"))

acc_max = 0
length_max = 0
sigma_max = 0
threshold_max = 0

# TO DO:
# make sure to make the appropriate changes for DSA dataset
# get the correct metrics
# seems to only check a few images,  make sure to create 3 validation splits and get the average of best thresholds
# make sure to verify on test set

for length in range(3,13,1):
    for sigma in range(6,25,1):
        for threshold in range(6,36,3):
            tn1,fp1,fn1,tp1=matched_filter(img=img1, target=target1,length = length, sigma = sigma/10, coefficient = threshold/10)
            tn2,fp2,fn2,tp2=matched_filter(img=img2, target=target2,length = length, sigma = sigma/10, coefficient = threshold/10)
            tn3,fp3,fn3,tp3=matched_filter(img=img3, target=target3,length = length, sigma = sigma/10, coefficient = threshold/10)
            tn_all = tn1+tn2+tn3
            fp_all = fp1+fp2+fp3
            fn_all = fn1+fn2+fn3
            tp_all = tp1+tp2+tp3
            acc = float(tp_all + tn_all) / float(tp_all + tn_all + fp_all + fn_all)
            print("accuracy",acc)
            if acc > acc_max:
                acc_max = acc
                length_max = length
                sigma_max = sigma
                threshold_max = threshold

print("acc_max",type(acc_max),acc_max)
print("length_max",type(length_max),length_max)
print("sigma_max",type(sigma_max),sigma_max)
print("threshold_max",type(threshold_max),threshold_max)
