import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# classical image processing from scratch

# box_filter
def box_kernel(k, m):
  b_kernel = np.ones([k,k]) * 1/m

  return b_kernel

# gaussian_filter
def gaussian_kernel1(k, std):
  g_kernel = np.zeros([k,k])
  k_h = k //2

  for y in range(k):
    for x in range(k):
      ky = y - k_h
      kx = x - k_h
      g_kernel[y,x] = np.exp(-(ky**2 + kx**2)/2*std**2)
  g_kernel /= np.sum(g_kernel)

  return g_kernel
  
def gaussian_kernel2(k, std):
  k_h = k //2
  [kx, ky] = np.ogrid[-k_h:k_h+1, -k_h:k_h+1]
  g_kernel = np.exp(-(kx**2+ky**2)/2*std**2)
  g_kernel /= np.sum(g_kernel)

  return g_kernel  

# Make_noise
def noise_image(M,N, type_of_noise):
  if (type_of_noise == 'uniform'):
    noise = np.random.uniform(0,100,size=(M,N))
  elif(type_of_noise == 'gaussian'):
    noise = np.random.normal(0,50,size=(M,N))
  else :
    print("error")
  return noise
  
  
# evaluation
def MSE(image1, image2):
  return np.mean((image1-image2)**2)

def PSNR(image1, image2):
  MAX = 255.0
  PSNR = 20*math.log10(MAX/np.sqrt(MSE(image1, image2)))
  return PSNR
  
  
# filtering_function
def filter_out(image, filter):
  H,W = image.shape
  k = filter.shape[0]
  image_out = np.zeros_like(image, dtype=np.float32)
  k_h = k//2

  # edge padding
  image = np.pad(image, ((k_h,k_h),(k_h,k_h)), mode = 'edge')  # (위,아래), (왼쪽,오른쪽) mode='edge'->원본 array에서 가장 가까운 모서리(edge)에 있는 값으로 테두리 데이터를 추가

  for y in range(H):
    for x in range(W):
      image_out[y,x] = np.sum(image[y:y+k, x:x+k] * filter)   # 필터 크기만큼의 이미지와 필터를 곱한 후 전체 시그마
  image_out /= np.sum(image_out)
  return image_out
  
  
# Scaling_funcion
def scaling1(image):
    H,W = image.shape
    image_out = np.zeros([2*H,2*W])
    image = np.float32(image)   # 오버플로우 막기 위해 float32로 변환

    for sy in range(2*H):
        for sx in range(2*W):
            y = int(sy/2)   # backward mapping (0,1->0에 맵핑, 2,3->1에 맵핑)
            x = int(sx/2)
            #image_out[sy,sx] = image[y,x] #-> 'Nearest Neighborhood' 
            # 1023 // 2 = 511 이어서 +1하면 512가 되어 이미지의 index범위 벗어남
            if((sy%2==0 and sx%2==0) or (sy==2*H-1) or (sx==2*W-1)):
                image_out[sy,sx] = image[y,x]
            elif(sy%2==1 and sx%2==0):
                image_out[sy,sx] = (image[y,x] + image[y+1,x]) // 2
            elif(sy%2==0 and sx%2==1):
                image_out[sy,sx] = (image[y,x] + image[y,x+1]) // 2
            elif(sy%2==1 and sx%2==1):
                image_out[sy,sx] = (image[y,x]+image[y+1,x]+image[y,x+1]+image[y+1,x+1])//4                        

    return image_out
    
def scaling2(image, N):
    H, W = image.shape
    image_out = np.zeros([N*H,N*W])

    for sy in range(N*H):
        for sx in range(N*W):
            a = sy/N-int(sy/N)
            b = sx/N-int(sx/N)

            y = int(sy/N) # backward mapping
            x = int(sx/N)
            y1 = np.clip(int(sy/N)+1, a_min=0, a_max=H-1)
            x1 = np.clip(int(sx/N)+1, a_min=0, a_max=W-1)

            image_out[sy,sx] =  (1-a)*((1-b)*image[y,x]+b*image[y,x1]) + a*((1-b)*image[y1,x]+b*image[y1,x1])
                
    return image_out 


# edge_detection_sobel_prewitt

def edge_dectection(image, type_of_detection):
    image = np.pad(image, ((1,1),(1,1)), mode='edge')
    dy_image = np.zeros([512,512])
    dx_image = np.zeros([512,512])
    edge_strength_map = np.zeros([512,512])

    if(type_of_detection == 'prewitt'):
        dy = np.array([-1, -1, -1, 0, 0, 0, 1, 1 ,1]).reshape(3,3)
        dx = dy.T
        for j in range(512):
            for i in range(512):
                dy_image[j,i] = np.sum(image[j:j+3, i:i+3] * dy)
                dx_image[j,i] = np.sum(image[j:j+3, i:i+3] * dx)
        edge_strength_map = np.sqrt(np.power(dy_image,2) + np.power(dx_image,2))


    elif(type_of_detection == 'sobel'):
        dy = np.array([-1, -2, -1, 0, 0, 0, 1, 2 ,1]).reshape(3,3)
        dx = dy.T
        for j in range(512):
            for i in range(512):
                dy_image[j,i] = np.sum(image[j:j+3, i:i+3] * dy)
                dx_image[j,i] = np.sum(image[j:j+3, i:i+3] * dx)
        edge_strength_map = np.sqrt(np.power(dy_image,2) + np.power(dx_image,2))

    else:
        print("Error!")
    
    return dy_image, dx_image, edge_strength_map
    

# Harris_coner_detector

def filter_out(image, filter):
    H,W = image.shape
    k = filter.shape[0]
    image_out = np.zeros_like(image, dtype=np.float32)
    k_h = k//2

    image = np.pad(image, ((k_h,k_h),(k_h,k_h)), mode='edge')

    for y in range(H):
        for x in range(W):
            image_out[y,x] = np.sum(image[y:y+2*k_h+1, x:x+2*k_h+1]*filter)
    
    return image_out

def gaussian_kernel2(k,std):
    k_h = k//2
    [kx, ky] = np.ogrid[-k_h:k_h+1, -k_h:k_h+1]
    g_kernel = np.exp(-(kx**2+ky**2)/2*std**2)
    g_kernel /= np.sum(g_kernel)

    return g_kernel

def Harris_corner(image):
    hor_dif = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ver_dif = hor_dif.T

    x_diff = filter_out(image, hor_dif)
    y_diff = filter_out(image, ver_dif)
    x_2_diff = x_diff**2
    y_2_diff = y_diff**2
    xy_diff = x_diff * y_diff

    k = 0.04
    gaussian = gaussian_kernel2(3,0.5)
    g_x_2_diff = filter_out(x_2_diff, gaussian)
    g_y_2_diff = filter_out(y_2_diff, gaussian)
    g_xy_diff = filter_out(xy_diff, gaussian)

    det = g_x_2_diff*g_y_2_diff - g_xy_diff**2
    trace = g_x_2_diff + g_y_2_diff
    c = det - k*trace**2
    c /= np.max(c)
    x = np.where(c>0.1)

    return x
  
  
# Moravec_algorithm
def Moravec(image):
    H,W = image.shape
    image = np.pad(image, ((1,1),(1,1)), mode='edge')
    image_out = np.zeros_like(image, dtype=np.float32)
    threshold = 150

    for y in range(H):
        for x in range(W):
            a = (image[y-1,x] - image[y,x])**2
            b = (image[y,x-1] - image[y,x])**2
            c = (image[y,x+1] - image[y,x])**2
            d = (image[y+1,x] - image[y,x])**2
            min = np.min([a,b,c,d])

            if min > threshold:
                image_out[y,x] = 255
   
    x = np.where(image_out>200)
    return x
