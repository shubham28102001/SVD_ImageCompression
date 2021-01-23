import
numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from random import normalvariate
from math import sqrt
from PIL import Image

'''
Returns transpose of A
Parameter: matrix A
'''


def transpose(A):
    rowA = len(A)
    colA = len(A[0])
    A_T = [[A[j][i] for j in range(rowA)] for i in range(colA)]
    return A_T


'''
Returns multiplication of two rectangular matrices each having more than 2 rows and columns
Parameter 1: matrix A having more than 2 rows and columns
Parameter 2: matrix B having more than 2 rows and columns
'''


def dot(A, B):
    rowA = len(A)
    rowB = len(B)
    colB = len(B[0])
    mul = [[0 for i in range(colB)] for j in range(rowA)]
    for i in range(rowA):
        for j in range(colB):
            for k in range(rowB):
                mul[i][j] += A[i][k] * B[k][j]
    return mul


''''
Returns multiplication of one rectangular matrix and 1 dimensional vector
Paramater 1: matrix A having more than 2 rows and columns
Paramater 2: 1 dimensional vector v
'''


def dotnd_1d(A, v):
    rowA = len(A)
    mul = []
    for i in range(rowA):
        mul.append(dot1d_1d(A[i], v))
    return mul


'''
Returns multiplication of two 1 dimensional vectors
# Parameter 1: 1 dimensional vector v1
# Parameter 2: 1 dimensional vector v2
'''


def dot1d_1d(v1, v2):
    rowv1 = len(v1)
    mul = []
    for i in range(rowv1):
        mul.append(v1[i] * v2[i])
    return sum(mul)


'''
Return the outerproduct of two 1 dimensional vectors
Parameter 1: 1 dimensional vector v1
Parameter 2: 1 dimensional vector v2
'''


def outerproduct(v1, v2):
    res = v1.reshape(-1, 1) * v2
    return res


'''
Returns the norm of a 1 dimensional vector
Parameter: 1 dimensional vector v
'''


def norm1d(v):
    return sqrt(sum(i ** 2 for i in v))


'''
Returns unit vector having each entry random normal distribution floating number having mean = 0 and sigma = 1
Parameter: min(rows, cols) of an image
'''


def randomUnitVector(n):
    unnormalizedVector = [normalvariate(0, 1) for i in range(n)]
    unnormalizedVectorNorm = sqrt(sum(v * v for v in unnormalizedVector))
    return [v / unnormalizedVectorNorm for v in unnormalizedVector]


'''
Performs one dimensional SVD and returns a vector
'''


def svd_1d(A, epsilon=1e-10):
    row, col = A.shape
    v = randomUnitVector(min(row, col))
    currentV = v
    previousV = None
    A_T = transpose(A)
    if row > col:
        B = dot(A_T, A)
    else:
        B = dot(A, A_T)

    while (1):
        previousV = currentV
        currentV = dotnd_1d(B, previousV)
        n = np.float64(norm1d(currentV))
        currentV = currentV / n

        if abs(dot1d_1d(currentV, previousV)) > 1 - epsilon:  # If cosine of angle between the vectors is close to 1
            return currentV


'''
Performs SVD using Power method and returns matrix U, Sigma and transpose of V
Parameter 1: Matrix A
Parameter 2: k is the number largest of singular values you want to compute of input matrix
                if k is None than it computes:
                    matrix U of dimension row x min(row, col)
                    matrix V of dimension min(row, col) x col
                    array Sigma of dimension 1 x min(row, col)

                else:
                    matrix U of dimension row x k
                    matrix V of dimension k x col
                    array Sigma of dimension 1 x k
Parameter 3: Value of epsilon 1e-10
'''


def svd(A, k=None, epsilon=1e-10):
    A = np.array(A, dtype=float)
    row, col = A.shape
    svdCurrent = []
    if k is None:
        k = min(row, col)

    for i in range(k):
        svd1dMatrix = A.copy()

        for u, singularValue, v in svdCurrent[:i]:
            svd1dMatrix -= singularValue * outerproduct(u, v)

        if row > col:
            v = svd_1d(svd1dMatrix, epsilon=epsilon)
            u = dot(A, v)
            sigma = np.float64(norm1d(u))
            u = u / sigma
        else:
            u = svd_1d(svd1dMatrix, epsilon=epsilon)
            v = dotnd_1d(transpose(A), u)
            sigma = np.float64(norm1d(v))
            v = v / sigma

        svdCurrent.append((u, sigma, v))

    us, singularValues, vs = [np.array(x) for x in zip(*svdCurrent)]
    us_T = np.array(transpose(us))
    return us_T, singularValues, vs


if __name__ == "__main__":

    image = np.array(Image.open('image.bmp'))
    # image = np.array(Image.open('Final.jpeg'))

    image = image / 255  # Normalizing the intensity values in each pixel
    row, col, channels = image.shape  # Channels: number of components used to represent each pixel
    print("The input image is of: ", row, "x", col, " pixels")

    fig1 = plt.figure(figsize=(15, 10))
    f1 = fig1.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image)
    plot1 = plt.imshow(image)
    f1.set_title('Original Image')
    plt.show()

    imageRed = image[:, :, 0]  # Extracting red color component from the original image
    imageGreen = image[:, :, 1]  # Extracting green color component from the original image
    imageBlue = image[:, :, 2]  # Extracting blue color component from the original image

    originalsize = image.nbytes
    print("The bytes required to store the original image: ", originalsize)

    n = int(input("Enter the desired rank to diagnolise the matrix: "))
    U_r, d_r, V_r = svd(imageRed, n)  # Perform SVD on red color component separately
    U_g, d_g, V_g = svd(imageGreen, n)  # Perform SVD on green color component separately
    U_b, d_b, V_b = svd(imageBlue, n)  # Perform SVD on blue color component separately

    k = int(input("Enter the desired rank for the output image less than {}: ".format(n)))
    while k > n:
        print("The rank of diagnolised matrix of original image is less than the desired rank of the output image")
        k = int(input("Enter the desired rank for the output image less than {}: ".format(n)))

    U_r_k = U_r[:, 0:k]  # Reconstruct matrix U_r_k to rows x k
    V_r_k = V_r[0:k, :]  # Reconstruct matrix V_r_k to k x cols
    U_g_k = U_g[:, 0:k]  # Reconstruct matrix U_r_k to rows x k
    V_g_k = V_g[0:k, :]  # Reconstruct matrix U_r_k to k x cols
    U_b_k = U_b[:, 0:k]  # Reconstruct matrix U_r_k to rows x k
    V_b_k = V_b[0:k, :]  # Reconstruct matrix U_r_k to k x cols

    # Extracting k largest singular values
    d_r_k = d_r[0:k]
    d_g_k = d_g[0:k]
    d_b_k = d_b[0:k]

    compressedsize = sum([matrix.nbytes for matrix in [U_r_k, d_r_k, V_r_k, U_g_k, d_g_k, V_g_k, U_b_k, d_b_k, V_b_k]])

    ratio = compressedsize / originalsize
    print("The compression ratio between the original image size and the total size of the compressed factors is",
          ratio)

    compressedsizeImageRed = dot(U_r_k, dot(np.diag(d_r_k),
                                            V_r_k))  # Reconstructing red component by multiplying U_r_k, d_r_k and V_r_k
    compressedsizeImageGreen = dot(U_g_k, dot(np.diag(d_g_k),
                                              V_g_k))  # Reconstructing green component by multiplying U_g_k, d_g_k and V_g_k
    compressedsizeImageBlue = dot(U_b_k, dot(np.diag(d_b_k),
                                             V_b_k))  # Reconstructing blue component by multiplying U_b_k, d_b_k and V_b_k

    compressedImage = np.zeros((row, col, 3))

    # Merging red, green and blue color component into single martix
    compressedImage[:, :, 0] = compressedsizeImageRed
    compressedImage[:, :, 1] = compressedsizeImageGreen
    compressedImage[:, :, 2] = compressedsizeImageBlue

    # Correcting the pixels whose intensity is less than 0 and greater than 1
    compressedImage[compressedImage < 0] = 0
    compressedImage[compressedImage > 1] = 1

    fig2 = plt.figure(figsize=(15, 10))
    f2 = fig2.add_subplot(1, 1, 1)
    plot2 = plt.imshow(compressedImage)
    f2.set_title("Compressed image having rank: {}".format(k))
    plt.show()