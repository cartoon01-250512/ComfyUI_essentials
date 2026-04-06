from .utils import max_, min_
from nodes import MAX_RESOLUTION
import comfy.utils
from nodes import SaveImage
from node_helpers import pillow
from PIL import Image, ImageOps

import kornia
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

#import warnings
#warnings.filterwarnings('ignore', module="torchvision")
import math
import os
import numpy as np
import folder_paths
from pathlib import Path
import random

import cv2  # for face_align, mediapipe
import dlib # for face_align
from PIL import Image as PILImage # for face_align
from urllib.parse import urlparse # for face_align, mediapipe
from torch.hub import download_url_to_file # for face_align, mediapipe
import subprocess # for face_align
from collections import Counter # for find_bbox_of_src_in_dest
import json # for Load Pose keypoint
import hashlib # for Load Pose keypoint
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Image analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageEnhanceDifference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "exponent": ("FLOAT", { "default": 0.75, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image analysis"

    def execute(self, image1, image2, exponent):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.permute([0,3,1,2]), image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        diff_image = image1 - image2
        diff_image = torch.pow(diff_image, exponent)
        diff_image = torch.clamp(diff_image, 0, 1)

        return(diff_image,)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Batch tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageBatchMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], { "default": "lanczos" }),
            }, "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image_1, method, image_2=None, image_3=None, image_4=None, image_5=None):
        out = image_1

        if image_2 is not None:
            if image_1.shape[1:] != image_2.shape[1:]:
                image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((image_1, image_2), dim=0)
        if image_3 is not None:
            if image_1.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_3), dim=0)
        if image_4 is not None:
            if image_1.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_4), dim=0)
        if image_5 is not None:
            if image_1.shape[1:] != image_5.shape[1:]:
                image_5 = comfy.utils.common_upscale(image_5.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_5), dim=0)

        return (out,)


class ImageExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image, size, method):
        orig_size = image.shape[0]

        if orig_size == size:
            return (image,)

        if size <= 1:
            return (image[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = image[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = image[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = image.repeat([math.ceil(size / image.shape[0])] + [1] * (len(image.shape) - 1))[:size]
        elif 'first' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat([image[:1].repeat(size-image.shape[0], 1, 1, 1), image], dim=0)
        elif 'last' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat((image, image[-1:].repeat((size-image.shape[0], 1, 1, 1))), dim=0)

        return (out,)

class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": -1, "min": -1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image, start, length):
        if length<0:
            length = image.shape[0]
        start = min(start, image.shape[0]-1)
        length = min(image.shape[0]-start, length)
        return (image[start:start + length], )


class ImageListToBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    INPUT_IS_LIST = True
    CATEGORY = "essentials/image batch"

    def execute(self, image):
        shape = image[0].shape[1:3]
        out = []

        for i in range(len(image)):
            img = image[i]
            if image[i].shape[1:3] != shape:
                img = comfy.utils.common_upscale(img.permute([0,3,1,2]), shape[1], shape[0], upscale_method='bicubic', crop='center').permute([0,2,3,1])
            out.append(img)

        out = torch.cat(out, dim=0)

        return (out,)

class ImageBatchToList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image):
        return ([image[i].unsqueeze(0) for i in range(image.shape[0])], )


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Image manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageCompositeFromMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "mask": ("MASK", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image_from, image_to, mask):
        frames = mask.shape[0]

        if image_from.shape[1] != image_to.shape[1] or image_from.shape[2] != image_to.shape[2]:
            image_to = comfy.utils.common_upscale(image_to.permute([0,3,1,2]), image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        if frames < image_from.shape[0]:
            image_from = image_from[:frames]
        elif frames > image_from.shape[0]:
            image_from = torch.cat((image_from, image_from[-1].unsqueeze(0).repeat(frames-image_from.shape[0], 1, 1, 1)), dim=0)

        mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        if image_from.shape[1] != mask.shape[1] or image_from.shape[2] != mask.shape[2]:
            mask = comfy.utils.common_upscale(mask.permute([0,3,1,2]), image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        out = mask * image_to + (1 - mask) * image_from

        return (out, )

class ImageComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, destination, source, x, y, offset_x, offset_y, mask=None):    
        ## ##################################################################################
        ## code added by dev_aj
        if source is not None:
            print(f"b4 source ndim ={source.ndim}, shape={source.shape}")
        if destination is not None:
            print(f"b4 destination ndim ={destination.ndim}, shape={destination.shape}")
        if mask is not None:
            print(f"b4 mask ndim ={mask.ndim}, shape={mask.shape}")
        # dest_3d = destination
        # dest_4d = destination
        # src_3d = source
        # src_4d = source
        # if source.ndim == 4:
            # source = source[0]
            # src_3d = source[0]
        # if destination.ndim == 4:
            # destination = destination[0]
            # dest_3d = destination[0]
        # print(f"after source ndim ={source.ndim}, shape={source.shape}")
        # print(f"after destination ndim ={destination.ndim}, shape={destination.shape}")
        ## code added ends
        ######################################################################################
        if mask is None:
            mask = torch.ones_like(source)[:,:,:,0]
        
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        if mask.shape[1:3] != source.shape[1:3]:
            mask = F.interpolate(mask.permute([0, 3, 1, 2]), size=(source.shape[1], source.shape[2]), mode='bicubic')
            mask = mask.permute([0, 2, 3, 1])
        
        if mask.shape[0] > source.shape[0]:
            mask = mask[:source.shape[0]]
        elif mask.shape[0] < source.shape[0]:
            mask = torch.cat((mask, mask[-1:].repeat((source.shape[0]-mask.shape[0], 1, 1, 1))), dim=0)
        
        if destination.shape[0] > source.shape[0]:
            destination = destination[:source.shape[0]]
        elif destination.shape[0] < source.shape[0]:
            ## ##################################################################################
            ## code added by dev_aj
            # destination = dest_4d
            # source = dest_4d
            # print(f"debug1  source ndim ={source.ndim}, shape={source.shape}")
            # print(f"debug1 destination ndim ={destination.ndim}, shape={destination.shape}") 
            ## code added ends
            ######################################################################################
            destination = torch.cat((destination, destination[-1:].repeat((source.shape[0]-destination.shape[0], 1, 1, 1))), dim=0)
        
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if len(x) < destination.shape[0]:
            x = x + [x[-1]] * (destination.shape[0] - len(x))
        if len(y) < destination.shape[0]:
            y = y + [y[-1]] * (destination.shape[0] - len(y))
        
        x = [i + offset_x for i in x]
        y = [i + offset_y for i in y]

        output = []
        ## ##################################################################################
        ## code added by dev_aj        
        print("destination.shape[0] =", destination.shape[0]) 
        print("destination.shape =", destination.shape)        
        print("source.shape[2] =", source.shape[2])
        print("source.shape =", source.shape)
        ## code added ends
        ######################################################################################
        for i in range(destination.shape[0]):
            
            d = destination[i].clone()
            s = source[i]
            m = mask[i]

            if x[i]+source.shape[2] > destination.shape[2]:
                print(f"dest ndim ={destination.ndim}, shape={destination.shape}")
                print(f"s ndim ={s.ndim}, shape={s.shape}")
                print(f"m ndim ={m.ndim}, shape={m.shape}")
                s = s[:, :destination.shape[2]-x[i], :] #[height, width, channels]
                m = s[:, :destination.shape[2]-x[i], :] #[height, width, channels]
                # s = s[:, :, :destination.shape[2]-x[i], :] # commented bcoz of IndexError: too many indices for tensor of dimension 3
                # m = m[:, :, :destination.shape[2]-x[i], :] # commented bcoz of IndexError
                print(f"s.shape[2] > dest.shape[2] s.shape={s.shape}, m.shape=(m.shape)")
            if y[i]+source.shape[1] > destination.shape[1]:
                print(f"dest ndim ={destination.ndim}, shape={destination.shape}")
                print(f"s ndim ={s.ndim}, shape={s.shape}")
                print(f"m ndim ={m.ndim}, shape={m.shape}")
                
                s = s[:destination.shape[1]-y[i], :, :]
                m = m[:destination.shape[1]-y[i], :, :]     
                # s = s[:, :destination.shape[1]-y[i], :, :] # commented bcoz of IndexError:
                # m = m[:destination.shape[1]-y[i], :, :]   # commented bcoz of IndexError:             
                print(f"s.shape[1] > dest.shape[1] s.shape={s.shape}, m.shape=(m.shape)")
            
            print(f"output s.shape={s.shape}, m.shape={m.shape}, d.shape={d.shape}")
            #output.append(s * m + d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] * (1 - m))
            s_component = s * m
            d_component = d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] * (1 - m)
            d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] = s_component + d_component
            output.append(d)
        
        output = torch.stack(output)

        # apply the source to the destination at XY position using the mask
        #for i in range(destination.shape[0]):
        #    output[i, y[i]:y[i]+source.shape[1], x[i]:x[i]+source.shape[2], :] = source * mask + destination[i, y[i]:y[i]+source.shape[1], x[i]:x[i]+source.shape[2], :] * (1 - mask)

        #for x_, y_ in zip(x, y):
        #    output[:, y_:y_+source.shape[1], x_:x_+source.shape[2], :] = source * mask + destination[:, y_:y_+source.shape[1], x_:x_+source.shape[2], :] * (1 - mask)

        #output[:, y:y+source.shape[1], x:x+source.shape[2], :] = source * mask + destination[:, y:y+source.shape[1], x:x+source.shape[2], :] * (1 - mask)
        #output = destination * (1 - mask) + source * mask

        return (output,)

## #########################################################
##      COMMON HELPER FUNCTIONS START by dev_aj
## #########################################################
def np_print_image(image, skip_all_zero_rows=False):
    np.set_printoptions(threshold=np.inf)  # Print the entire array, no truncation
    # Threshold the array to get a binary (0/1) array
    binary_arr = (image > 0).astype(int)  # or use any threshold you want

    arr2d = binary_arr.squeeze()  # Now shape is (713, 512)

    # Print each row as a string of 1s and 0s
    for row in arr2d:
        
        if skip_all_zero_rows and row.any():  # True if any value in the row is nonzero
            print(''.join(str(int(x)) for x in row))
        else:
            print(''.join(str(int(x)) for x in row))


# For Image

def prepare_for_opencv_image(tensor):
    # Remove batch dimension if present
    # [1, 767, 581, 3] [batch, height, width, channel]
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    shape = tensor.shape

    np_img = None
    # (C, H, W) -> (H, W, C)
    if tensor.ndim == 3 and shape[0] in [1, 3] and shape[1]>3 and shape[2]>3 :  # likely (C, H, W)
        np_img = tensor.permute(1, 2, 0).numpy()
    # (H, W, C) already
    elif tensor.ndim == 3 and shape[2] in [1, 3] and shape[0]>3 and shape[1]>3 :  # likely (H, W, C)
        np_img = tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {shape}")

    return np_img
    
def convert_to_opencv_image(img):
    cv_img = None
    if isinstance(img, np.ndarray):
        print("NumPy array (likely OpenCV or PyTorch tensor)")
    elif isinstance(img, PILImage.Image):
        print("PIL Image")
    elif type(img).__module__.startswith('cv2'):
        print("OpenCV image")
    elif isinstance(img, torch.Tensor):
        print("PyTorch Tensor image")
        cv_img = convert_torch_image_to_opencv(img)           
    else:
        print("Unknown type:", type(img))
    return cv_img
    
def convert_torch_image_to_opencv(torch_img):
    
    # Assume 'torch_img' is your PyTorch image tensor with shape (C, H, W)
    tensor = torch_img.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients
    
    # if tensor.ndim == 4:
        # # if input torch_img has shape [1, 767, 581, 3] [batch, height, width, channel]
        # np_img = tensor.squeeze(0).numpy()     # Step 1 & 3: remove batch, to numpy -> shape [767, 581, 3]
    # else:
        # np_img = tensor.numpy()         # Convert to numpy array
        
    np_img = prepare_for_opencv_image(tensor)


    # If tensor is normalized to [0, 1], scale to [0, 255]
    np_img = (np_img * 255).astype(np.uint8)

    # Convert from RGB to BGR for OpenCV
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    
    return cv_img

def convert_opencv_image_to_torch(image):

    # Step 2: Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 3: Convert to float32 and normalize to [0, 1] if needed
    image = image.astype(np.float32) / 255.0  # Remove this line if you want to keep uint8

    # Step 4: Convert to torch tensor
    tensor = torch.from_numpy(image)  # shape: [H, W, C]

    # Step 5: Add batch dimension
    tensor = tensor.unsqueeze(0)  # shape: [1, H, W, C]
            
    return tensor

# For MASK
def prepare_torch_mask_for_opencv(tensor_mask):
    """
    Takes a PyTorch mask tensor and returns a NumPy array of shape (H, W).
    Always picks the first element along batch/channel axes if present.
    Converts to uint8 for OpenCV if needed.
    """
    
    # Acceptable shapes: [H, W], [1, H, W], [B, H, W], [H, W, C], [B, H, W, C]
    mask_np = tensor_mask.numpy()
    # Remove batch dimension if present
    if mask_np.ndim == 3:
        # [B, H, W] or [H, W, C]
        if mask_np.shape[0] == 1 and mask_np.shape[1] >3:
            # [1, H, W] to [H, W]
            mask_np = mask_np[0]    # is equivalent to  mask_np[0,:,:]
        elif mask_np.shape[2] in [1,3]:
            # [H, W, 1] or [H, W, 3] to [H, W]
            mask_np = mask_np[:,:,0]
    elif mask_np.ndim == 4:
        # [B, H, W, C] to [H, W, C]
        # [B, C, H, W] to [H, W, C]
        mask_np = mask_np[0] # removed Batch

        if mask_np.shape[2] in [1,3] and mask_np.shape[0]>3 and mask_np.shape[1]>3:
            # [H, W, 1] or [H, W, 3] to [H, W]
            mask_np = mask_np[:,:,0]
        elif mask_np.shape[0] in [1,3] and mask_np.shape[1]>3 and mask_np.shape[2]>3:
            # [C, H, W] to [H, W, C]
            mask_np = np.transpose(mask_np, (1, 2, 0))  # New shape [H, W, C]
            #  [H, W, C] to  [H, W]
            mask_np = mask_np[:,:,0] # added on 29-01-2026
    elif mask_np.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {shape}")
    # By now mask_np should be (H, W)

    # slicing and transposing creates non-contiguous arrays, which causes below issues :
    # 1. C/C++ libraries (OpenCV, CUDA, TensorRT) crash or give garbage with non-contiguous arrays
    # 2. 10-100x slower operations due to poor cache performance
    # 3. GPU transfers fail (cuDNN, PyTorch expect C-contiguous)

    # So convert to contiguous block of memory  
    # mask_np is stored in a contiguous block of memory in C order (row-major order).
    mask_np = np.ascontiguousarray(mask_np)
    
    # Convert to uint8 for OpenCV if not already
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)
    return mask_np

def convert_to_opencv_mask(mask):
    cv_mask = None
    if isinstance(mask, np.ndarray):
        print("NumPy array (likely OpenCV or PyTorch tensor)")
    elif isinstance(mask, PILImage.Image):
        print("PIL Mask")
    elif type(mask).__module__.startswith('cv2'):
        print("OpenCV Mask")
    elif isinstance(mask, torch.Tensor):
        print("PyTorch Tensor Mask")
        cv_mask = convert_torch_mask_to_opencv(mask)           
    else:
        print("Unknown type:", type(mask))
    return cv_mask
    
def convert_torch_mask_to_opencv(torch_mask):        
    # Assume 'torch_mask' is your PyTorch image tensor with shape (C, H, W)
    tensor = torch_mask.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients          
    np_mask = prepare_torch_mask_for_opencv(tensor)

    # If tensor is normalized to [0, 1], scale to [0, 255]
    # np_mask = (np_mask * 255).astype(np.uint8)
    
    return np_mask

def convert_opencv_mask_to_torch(mask):

    # 1. Convert to PyTorch tensor
    mask_torch = torch.from_numpy(mask)  # shape: [H, W]

    # 2. Add batch dimension to make shape [1, H, W]
    mask_torch_batched = mask_torch.unsqueeze(0)  # shape: [B, H, W], here B=1

    return mask_torch_batched

## #########################################################
##      COMMON HELPER FUNCTIONS END by dev_aj
## #########################################################

class WhiteNoiseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dest": ("MASK",),
                "mask": ("MASK",), # mask contains the area where white noise should be generated.                
            },
            "optional": {
                "coverage": ("FLOAT", { "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, }),           
            }
        }

    RETURN_TYPES = ("MASK","MASK",)
    RETURN_NAMES = ("WHITE NOISE","DEST WITH NOISE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


    def prepare_for_opencv(self, tensor):
        # Remove batch dimension if present
        # [1, 767, 581, 3] [batch, height, width, channel]
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        shape = tensor.shape

        np_img = None
        # (C, H, W) -> (H, W, C)
        if tensor.ndim == 3 and shape[0] in [1, 3] and shape[1]>3 and shape[2]>3 :  # likely (C, H, W)
            np_img = tensor.permute(1, 2, 0).numpy()
        # (H, W, C) already
        elif tensor.ndim == 3 and shape[2] in [1, 3] and shape[0]>3 and shape[1]>3 :  # likely (H, W, C)
            np_img = tensor.numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {shape}")

        return np_img
        
    def convert_to_opencv_image(self, img):
        print(f'img ndim={img.ndim}  shape={img.shape}')
        cv_img = None
        if isinstance(img, np.ndarray):
            print("NumPy array (likely OpenCV or PyTorch tensor)")
        elif isinstance(img, PILImage.Image):
            print("PIL Image")
        elif type(img).__module__.startswith('cv2'):
            print("OpenCV image")
        elif isinstance(img, torch.Tensor):
            print("PyTorch Tensor image")
            cv_img = self.convert_torch_image_to_opencv(img)           
        else:
            print("Unknown type:", type(img))
        return cv_img
       
    def convert_torch_image_to_opencv(self, torch_img):
 
        print("########################################################################################")       
        print(f'convert_torch_image_to_opencv() torch_img ndim={torch_img.ndim}  shape={torch_img.shape}')
        
        # Assume 'torch_img' is your PyTorch image tensor with shape (C, H, W)
        tensor = torch_img.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients
        
        np_img = self.prepare_for_opencv(tensor)
 
        print(f'convert_torch_image_to_opencv() np_img ndim={np_img.ndim}  shape={np_img.shape}')
        
        # If tensor is normalized to [0, 1], scale to [0, 255]
        np_img = (np_img * 255).astype(np.uint8)


            
        if np_img.ndim ==3 and np_img.shape[2] == 3:
            # Convert from RGB to BGR for OpenCV
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            print("COLOR_RGB2BGR")
            
        return np_img

    def convert_opencv_image_to_torch(self, image):

        print(f'opencv_image_to_torch() image ndim={image.ndim}  shape={image.shape}')
        
        if image.ndim ==3 and image.shape[2] == 3:
            # Step 2: Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 3: Convert to float32 and normalize to [0, 1] if needed
        image = image.astype(np.float32) / 255.0  # Remove this line if you want to keep uint8

        # Step 4: Convert to torch tensor
        tensor = torch.from_numpy(image)  # shape: [H, W, C]

        if image.ndim ==3 and image.shape[2] == 3:
            # Step 5: Add batch dimension
            tensor = tensor.unsqueeze(0)  # result shape: [1, H, W, C]
        elif image.ndim ==3 and image.shape[2] == 1:
            tensor = tensor.squeeze(-1)   #  to remove the last dimension, result shape: [H, W]
                
        return tensor
    
    def execute(self, dest, mask, coverage=0.1):
        dest = self.convert_to_opencv_image(dest)
        if dest is None:
            raise ValueError("dest couldn't be converted to opencv")
        
        mask = self.convert_to_opencv_image(mask)
        if mask is None:
            raise ValueError("mask couldn't be converted to opencv")

        # Resize mask to match dest shape if needed, using bicubic interpolation            
        if mask.shape[:2] != dest.shape[:2]:
            mask = cv2.resize(mask, (dest.shape[1], dest.shape[0]), interpolation=cv2.INTER_CUBIC)
            print(f"after resize, mask.shape={mask.shape}")
            
        
        # Ensure mask is binary (0 or 255)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        white_noise = np.zeros((dest.shape[:2]), dtype=np.uint8)
        print(f"dest.shape={dest.shape}")
        print(f"mask.shape={mask.shape}")

        # Get indices where mask is 255
        mask_indices = np.where(mask == 255)
        num_mask_pixels = len(mask_indices[0])
        num_noise_pixels = int(num_mask_pixels * coverage)
        print(f"num_mask_pixels={num_mask_pixels}")
        print(f"num_noise_pixels={num_noise_pixels}")
        
        # Randomly select pixels in the mask to apply noise
        selected_indices = np.random.choice(num_mask_pixels, num_noise_pixels, replace=False)
        y_coords = mask_indices[0][selected_indices]
        x_coords = mask_indices[1][selected_indices]
                
        # Apply white noise to the output image on randomly selected points
        white_noise[y_coords, x_coords] = 255

        if dest.ndim ==3 and dest.shape[2] == 1:
            dest = dest.squeeze(-1)
        
        # Convert masks to boolean (True where 255, False where 0)
        bool_mask1 = dest.astype(bool)
        bool_mask2 = white_noise.astype(bool)

        # Merge masks with logical OR
        merged_bool_mask = bool_mask1 | bool_mask2

        # If needed, convert back to 0/255 mask
        merged_mask = merged_bool_mask.astype(np.uint8) * 255
        
        white_noise_torch = self.convert_opencv_image_to_torch(white_noise)
        dest_merged_with_white_noise_torch = self.convert_opencv_image_to_torch(merged_mask)
        
        print(f"white_noise_torch.shape={white_noise_torch.shape}")
        print(f"dest_merged_with_white_noise_torch.shape={dest_merged_with_white_noise_torch.shape}")
        
        return (white_noise_torch, dest_merged_with_white_noise_torch,)

class FaceAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",), # src img contains face which needs to be aligned to dst img face.
                "dest": ("IMAGE",), # dst img contains face which is the target size and orientation to which src img face must align to.
            },
            "optional": {
                "show_facial_landmark_indications": ("BOOLEAN", { "default": False }),
                "use_5_landmark": ("BOOLEAN", { "default": True }),
                "scale_based_on_target_dist": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MAT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MAT","width", "height" )
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def euclidean_distance(self, landmark_points, p1_idx, p2_idx):
        p1 = landmark_points[p1_idx]
        p2 = landmark_points[p2_idx]
        print(f"p1={p1}")
        print(f"p2={p2}")
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        print(f"Euclidean distance: {distance}")
        return distance

    def scale_transform(self, image, p1, p2, target_dist):
        p1 = np.array(p1, dtype=np.float32)  # First point
        p2 = np.array(p2, dtype=np.float32)  # Second point

        # 1. Compute current distance and scaling factor
        current_dist = np.linalg.norm(p2 - p1)
        scale = target_dist / current_dist

        # 2. Compute the center between the points (to scale about the midpoint)
        center = (p1 + p2) / 2

        # 3. Build the scaling transformation matrix about the center
        # Translate center to origin, scale, then translate back
        M_translate1 = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]]], dtype=np.float32)
        M_scale = np.array([[scale, 0, 0],
                            [0, scale, 0]], dtype=np.float32)
        M_translate2 = np.array([[1, 0, center[0]],
                                 [0, 1, center[1]]], dtype=np.float32)

        # Combine transformations: T2 * S * T1
        M_temp = np.vstack([M_scale, [0, 0, 1]]) @ np.vstack([M_translate1, [0, 0, 1]])
        M_final = np.vstack([M_translate2, [0, 0, 1]]) @ M_temp
        M_affine = M_final[:2, :]

        # 4. Apply the affine transformation
        output_shape = (image.shape[1], image.shape[0])  # (width, height)
        scaled_image = cv2.warpAffine(image, M_affine, output_shape, flags=cv2.INTER_LINEAR)

        return scaled_image

    def prepare_for_opencv(self, tensor):
        # Remove batch dimension if present
        # [1, 767, 581, 3] [batch, height, width, channel]
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        shape = tensor.shape

        np_img = None
        # (C, H, W) -> (H, W, C)
        if tensor.ndim == 3 and shape[0] in [1, 3] and shape[1]>3 and shape[2]>3 :  # likely (C, H, W)
            np_img = tensor.permute(1, 2, 0).numpy()
        # (H, W, C) already
        elif tensor.ndim == 3 and shape[2] in [1, 3] and shape[0]>3 and shape[1]>3 :  # likely (H, W, C)
            np_img = tensor.numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {shape}")

        return np_img
        
    def convert_to_opencv_image(self, img):
        print(f'img ndim={img.ndim}  shape={img.shape}')
        cv_img = None
        if isinstance(img, np.ndarray):
            print("NumPy array (likely OpenCV or PyTorch tensor)")
        elif isinstance(img, PILImage.Image):
            print("PIL Image")
        elif type(img).__module__.startswith('cv2'):
            print("OpenCV image")
        elif isinstance(img, torch.Tensor):
            print("PyTorch Tensor image")
            cv_img = self.convert_torch_image_to_opencv(img)           
        else:
            print("Unknown type:", type(img))
        return cv_img
       
    def convert_torch_image_to_opencv(self, torch_img):
        
        print(f'convert_torch_image_to_opencv() torch_img ndim={torch_img.ndim}  shape={torch_img.shape}')
        
        # Assume 'torch_img' is your PyTorch image tensor with shape (C, H, W)
        tensor = torch_img.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients
        
        # if tensor.ndim == 4:
            # # if input torch_img has shape [1, 767, 581, 3] [batch, height, width, channel]
            # np_img = tensor.squeeze(0).numpy()     # Step 1 & 3: remove batch, to numpy -> shape [767, 581, 3]
        # else:
            # np_img = tensor.numpy()         # Convert to numpy array
            
        np_img = self.prepare_for_opencv(tensor)
 
        print("########################################################################################")
        print(f'convert_torch_image_to_opencv() np_img ndim={np_img.ndim}  shape={np_img.shape}')

        # If tensor is normalized to [0, 1], scale to [0, 255]
        np_img = (np_img * 255).astype(np.uint8)

        # Convert from RGB to BGR for OpenCV
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        return cv_img

    def convert_opencv_image_to_torch(self, image):

        print(f'opencv_image_to_torch() image ndim={image.ndim}  shape={image.shape}')
        # Step 2: Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 3: Convert to float32 and normalize to [0, 1] if needed
        image = image.astype(np.float32) / 255.0  # Remove this line if you want to keep uint8

        # Step 4: Convert to torch tensor
        tensor = torch.from_numpy(image)  # shape: [H, W, C]

        # Step 5: Add batch dimension
        tensor = tensor.unsqueeze(0)  # shape: [1, H, W, C]
                
        return tensor

    # Helper: Get 5-point landmarks (eyes, nose base, mouth corners)
    def get_landmarks(self, image, detector, predictor):
        dets = detector(image, 1)
        if len(dets) == 0:
            raise ValueError("No face detected.")
        shape = predictor(image, dets[0])
        # Use 5-point model; for 68-point model, adjust indices as needed
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
        return landmarks
        

    def indicate_landmark(self, face_image, landmark_points, mark_color=(0,0,255), mark_size=4):
        """
        Draws landmark points on a copy of the face image and displays it.

        Args:
            face_image (numpy.ndarray): The input image (as loaded by cv2).
            landmark_points (list or np.ndarray): List/array of (x, y) tuples or shape (5,2).
            mark_color (tuple): BGR color for the points (default: red).
            mark_size (int): Radius of the point in pixels (default: 4).
        """
        # Make a copy so the original image is not modified
        img_copy = face_image.copy()
        for idx, (x, y) in enumerate(landmark_points):
            # Draw filled circle at each landmark
            if idx in [1,15]:
                green = (0,255,0)
                cv2.circle(img_copy, (int(x), int(y)), mark_size, green, thickness=-1)  # -1 fills the circle[1][5][6]
            else:
                cv2.circle(img_copy, (int(x), int(y)), mark_size, mark_color, thickness=-1)  # -1 fills the circle[1][5][6]
        return img_copy

    def get_local_filepath(self, url, dirname, local_file_name=None):
        # Determine the Local File Name
        if not local_file_name:
            parsed_url = urlparse(url)
            file_name_with_bz2_ext = os.path.basename(parsed_url.path)
            local_file_name, ext = os.path.splitext(file_name_with_bz2_ext)

        # Check for an Existing File via get_full_path
        destination = folder_paths.get_full_path(dirname, local_file_name)
        if destination:
            print(f"using existing landmark model: {destination}")
            return destination

        # Ensure the Target Directory Exists
        folder = os.path.join(folder_paths.models_dir, dirname)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Download the File if Needed
        destination = os.path.join(folder, local_file_name)
        if not os.path.exists(destination):
            parsed_url = urlparse(url)
            file_name_with_bz2_ext = os.path.basename(parsed_url.path)
            destination_bz2_ext = os.path.join(folder, file_name_with_bz2_ext)
            print(f"downloading {url} to {destination_bz2_ext}")
            download_url_to_file(url, destination_bz2_ext) # Download file at the given URL to a local path
            # This will create 'landmarks.dat' and remove the .bz2 file
            subprocess.run(['bzip2', '-d', destination_bz2_ext], check=True)
        return destination

    def execute(self, src, dest, show_facial_landmark_indications=False, use_5_landmark=True, scale_based_on_target_dist=False):
        
        src_img = self.convert_to_opencv_image(src)
        if src_img is None:
            raise ValueError("src_img couldn't be converted to opencv")

        dst_img = self.convert_to_opencv_image(dest)
        if dst_img is None:
            raise ValueError("dst_img couldn't be converted to opencv")
 

        # Convert to grayscale for detection
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

        # Load dlib models
        detector = dlib.get_frontal_face_detector()
        dlib_face_landmark_dir_name = "dlib"
        dlib_face_landmark_models_list = {
            "68-point face landmark model": {
                "download_url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            },
            "5-point face landmark model": {
                "download_url": "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
            },
        }        
        if use_5_landmark:
            model = "5-point face landmark model"
        else:
            model = "68-point face landmark model"
        landmark_filepath = self.get_local_filepath(dlib_face_landmark_models_list[model]["download_url"], dlib_face_landmark_dir_name,)
        predictor = dlib.shape_predictor(landmark_filepath)  # Download from dlib model zoo

        # Get landmarks
        src_landmarks = self.get_landmarks(src_gray, detector, predictor)
        src_points = np.float32(src_landmarks)
        print(f'src_points=\n{src_points}')

        dst_landmarks = self.get_landmarks(dst_gray, detector, predictor)

        # Select corresponding points (eyes and nose, for example)
        # For 5-point model: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        # src_points = np.float32(src_landmarks)
        dst_points = np.float32(dst_landmarks)
        # print(f'src_points=\n{src_points}')
        print('#############################')
        print(f'dst_points=\n{dst_points}')

        # Compute affine transform
        M = cv2.estimateAffinePartial2D(src_points, dst_points)[0]

        # Warp source face to match destination face geometry
        aligned_src = cv2.warpAffine(src_img, M, (dst_img.shape[1], dst_img.shape[0]))
        width,height = dst_img.shape[1], dst_img.shape[0]


        aligned_src_landmarks = self.get_landmarks(aligned_src, detector, predictor)
        print('#############################')
        print(f'aligned_src_landmarks=\n{aligned_src_landmarks}')



        align_src_torch = None
      
        # Show result
        if show_facial_landmark_indications:
            # src_indicate_img = indicate_landmark(src_img, src_landmarks, mark_color=(0,0,255), mark_size=4)
            alignsrc_ind_img = self.indicate_landmark(aligned_src, aligned_src_landmarks, mark_color=(0,0,255), mark_size=4)
            # dst_indicate_img = indicate_landmark(dst_img, dst_landmarks, mark_color=(0,0,255), mark_size=4) 
            # align_src_torch = self.convert_opencv_image_to_torch(alignsrc_ind_img)
        else:
            alignsrc_ind_img = aligned_src
            # align_src_torch = self.convert_opencv_image_to_torch(aligned_src)

        if scale_based_on_target_dist and  not use_5_landmark:
            # Scale Image according to target distance
            # Work only for 68 point face landmark 
            p1_idx, p2_idx = 1,15 # [Face outermost point near left ear,  Face outermost point near right ear]
            dst_distance = self.euclidean_distance(dst_landmarks, p1_idx, p2_idx)
            
            # Scale alignsrc_ind_img based on line between points p1 and p2 to reach atleast dst_distance
            alignsrc_scaled_img = self.scale_transform(alignsrc_ind_img, aligned_src_landmarks[p1_idx], aligned_src_landmarks[p2_idx], target_dist=dst_distance)
            alignsrc_ind_img = alignsrc_scaled_img
            
        align_src_torch = self.convert_opencv_image_to_torch(alignsrc_ind_img)
        
        return(align_src_torch,M,width,height)

# FaceAlign used computer vision techniques like HOG for detecting the face. It has low detection for face that was titled or not front facing.
# In FaceAlignExternalDetector, the Bounding Box of the face is inputted as a parameter, which may be already detected using SAM or other techniques.
# Note: bbx of source and destination face must be provided.
class FaceAlignExternalDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",), # src img contains face which needs to be aligned to dst img face.
                "dest": ("IMAGE",), # dst img contains face which is the target size and orientation to which src img face must align to.
                "src_bbox": ("BBOX",), # bbx of source face
                "dest_bbox": ("BBOX",), # bbx of dest face
            },
            "optional": {
                "show_facial_landmark_indications": ("BOOLEAN", { "default": False }),
                "use_5_landmark": ("BOOLEAN", { "default": True }),
                "scale_based_on_target_dist": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MAT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MAT","width", "height" )
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def euclidean_distance(self, landmark_points, p1_idx, p2_idx):
        p1 = landmark_points[p1_idx]
        p2 = landmark_points[p2_idx]
        print(f"p1={p1}")
        print(f"p2={p2}")
        distance = np.linalg.norm(np.array(p1) - np.array(p2))
        print(f"Euclidean distance: {distance}")
        return distance

    def scale_transform(self, image, p1, p2, target_dist):
        p1 = np.array(p1, dtype=np.float32)  # First point
        p2 = np.array(p2, dtype=np.float32)  # Second point

        # 1. Compute current distance and scaling factor
        current_dist = np.linalg.norm(p2 - p1)
        scale = target_dist / current_dist

        # 2. Compute the center between the points (to scale about the midpoint)
        center = (p1 + p2) / 2

        # 3. Build the scaling transformation matrix about the center
        # Translate center to origin, scale, then translate back
        M_translate1 = np.array([[1, 0, -center[0]],
                                 [0, 1, -center[1]]], dtype=np.float32)
        M_scale = np.array([[scale, 0, 0],
                            [0, scale, 0]], dtype=np.float32)
        M_translate2 = np.array([[1, 0, center[0]],
                                 [0, 1, center[1]]], dtype=np.float32)

        # Combine transformations: T2 * S * T1
        M_temp = np.vstack([M_scale, [0, 0, 1]]) @ np.vstack([M_translate1, [0, 0, 1]]) # @ operator performs matrix multiplication between two arrays
        M_final = np.vstack([M_translate2, [0, 0, 1]]) @ M_temp
        M_affine = M_final[:2, :]

        # 4. Apply the affine transformation
        output_shape = (image.shape[1], image.shape[0])  # (width, height)
        scaled_image = cv2.warpAffine(image, M_affine, output_shape, flags=cv2.INTER_LINEAR)

        return scaled_image

    def prepare_for_opencv(self, tensor):
        # Remove batch dimension if present
        # [1, 767, 581, 3] [batch, height, width, channel]
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        shape = tensor.shape

        np_img = None
        # (C, H, W) -> (H, W, C)
        if tensor.ndim == 3 and shape[0] in [1, 3] and shape[1]>3 and shape[2]>3 :  # likely (C, H, W)
            np_img = tensor.permute(1, 2, 0).numpy()
        # (H, W, C) already
        elif tensor.ndim == 3 and shape[2] in [1, 3] and shape[0]>3 and shape[1]>3 :  # likely (H, W, C)
            np_img = tensor.numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {shape}")

        return np_img
        
    def convert_to_opencv_image(self, img):
        print(f'img ndim={img.ndim}  shape={img.shape}')
        cv_img = None
        if isinstance(img, np.ndarray):
            print("NumPy array (likely OpenCV or PyTorch tensor)")
        elif isinstance(img, PILImage.Image):
            print("PIL Image")
        elif type(img).__module__.startswith('cv2'):
            print("OpenCV image")
        elif isinstance(img, torch.Tensor):
            print("PyTorch Tensor image")
            cv_img = self.convert_torch_image_to_opencv(img)           
        else:
            print("Unknown type:", type(img))
        return cv_img
       
    def convert_torch_image_to_opencv(self, torch_img):
        
        print(f'convert_torch_image_to_opencv() torch_img ndim={torch_img.ndim}  shape={torch_img.shape}')
        
        # Assume 'torch_img' is your PyTorch image tensor with shape (C, H, W)
        tensor = torch_img.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients
        
        # if tensor.ndim == 4:
            # # if input torch_img has shape [1, 767, 581, 3] [batch, height, width, channel]
            # np_img = tensor.squeeze(0).numpy()     # Step 1 & 3: remove batch, to numpy -> shape [767, 581, 3]
        # else:
            # np_img = tensor.numpy()         # Convert to numpy array
            
        np_img = self.prepare_for_opencv(tensor)
 
        print("########################################################################################")
        print(f'convert_torch_image_to_opencv() np_img ndim={np_img.ndim}  shape={np_img.shape}')

        # If tensor is normalized to [0, 1], scale to [0, 255]
        np_img = (np_img * 255).astype(np.uint8)

        # Convert from RGB to BGR for OpenCV
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        return cv_img

    def convert_opencv_image_to_torch(self, image):

        print(f'opencv_image_to_torch() image ndim={image.ndim}  shape={image.shape}')
        # Step 2: Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 3: Convert to float32 and normalize to [0, 1] if needed
        image = image.astype(np.float32) / 255.0  # Remove this line if you want to keep uint8

        # Step 4: Convert to torch tensor
        tensor = torch.from_numpy(image)  # shape: [H, W, C]

        # Step 5: Add batch dimension
        tensor = tensor.unsqueeze(0)  # shape: [1, H, W, C]
                
        return tensor

    # Helper: Get 5-point landmarks (eyes, nose base, mouth corners)
    def get_landmarks(self, image, detector, predictor):
        dets = detector(image, 1)
        if len(dets) == 0:
            raise ValueError("No face detected.")
        shape = predictor(image, dets[0])
        # Use 5-point model; for 68-point model, adjust indices as needed
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
        return landmarks
        
    # Helper: Get landmarks by using face detection bbx from outside
    def get_landmarks_bypass_detector(self, image, dets, predictor):
        if len(dets) == 0:
            raise ValueError("No face detected.")
        shape = predictor(image, dets[0])
        # Use 5-point model; for 68-point model, adjust indices as needed
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])
        return landmarks

    def indicate_landmark(self, face_image, landmark_points, mark_color=(0,0,255), mark_size=4):
        """
        Draws landmark points on a copy of the face image and displays it.

        Args:
            face_image (numpy.ndarray): The input image (as loaded by cv2).
            landmark_points (list or np.ndarray): List/array of (x, y) tuples or shape (5,2).
            mark_color (tuple): BGR color for the points (default: red).
            mark_size (int): Radius of the point in pixels (default: 4).
        """
        # Make a copy so the original image is not modified
        img_copy = face_image.copy()
        for idx, (x, y) in enumerate(landmark_points):
            # Draw filled circle at each landmark
            if idx in [1,15]:
                green = (0,255,0)
                cv2.circle(img_copy, (int(x), int(y)), mark_size, green, thickness=-1)  # -1 fills the circle[1][5][6]
            else:
                cv2.circle(img_copy, (int(x), int(y)), mark_size, mark_color, thickness=-1)  # -1 fills the circle[1][5][6]
        return img_copy

    def get_local_filepath(self, url, dirname, local_file_name=None):
        # Determine the Local File Name
        if not local_file_name:
            parsed_url = urlparse(url)
            file_name_with_bz2_ext = os.path.basename(parsed_url.path)
            local_file_name, ext = os.path.splitext(file_name_with_bz2_ext)

        # Check for an Existing File via get_full_path
        destination = folder_paths.get_full_path(dirname, local_file_name)
        if destination:
            print(f"using existing landmark model: {destination}")
            return destination

        # Ensure the Target Directory Exists
        folder = os.path.join(folder_paths.models_dir, dirname)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Download the File if Needed
        destination = os.path.join(folder, local_file_name)
        if not os.path.exists(destination):
            parsed_url = urlparse(url)
            file_name_with_bz2_ext = os.path.basename(parsed_url.path)
            destination_bz2_ext = os.path.join(folder, file_name_with_bz2_ext)
            print(f"downloading {url} to {destination_bz2_ext}")
            download_url_to_file(url, destination_bz2_ext) # Download file at the given URL to a local path
            # This will create 'landmarks.dat' and remove the .bz2 file
            subprocess.run(['bzip2', '-d', destination_bz2_ext], check=True)
        return destination

    def if_valid_bbox(self, bbox):
        # If bbox is a list containing one tuple, unpack it
        if isinstance(bbox, list):
            if len(bbox) == 1 and isinstance(bbox[0], tuple):
                bb = bbox[0]
            else:
                # Unexpected list length or content
                raise ValueError(f"Expected list with one tuple for bbox input, got: {bbox}")
        elif isinstance(bbox, tuple):
            bb = bbox
        else:
            raise TypeError(f"Expected bbox input to be tuple or list of one tuple, got: {type(bbox)}")

        if len(bb) < 4:
            raise ValueError(f"BBox tuple must have at least 4 elements, got: {bb}")

        left, top, width, height = bb[:4]
        return bb[:4]


    def execute(self, src, dest, src_bbox, dest_bbox, show_facial_landmark_indications=False, use_5_landmark=True, scale_based_on_target_dist=False):
        
        src_img = self.convert_to_opencv_image(src)
        if src_img is None:
            raise ValueError("src_img couldn't be converted to opencv")

        dst_img = self.convert_to_opencv_image(dest)
        if dst_img is None:
            raise ValueError("dst_img couldn't be converted to opencv")
 

        if src_bbox is None:
            raise ValueError("src_bbox must be provided")

        src_x, src_y, src_w, src_h = self.if_valid_bbox(src_bbox)
        dst_x, dst_y, dst_w, dst_h = self.if_valid_bbox(dest_bbox)

        src_rect1 = dlib.rectangle(src_x, src_y, src_x + src_w, src_y + src_h)
        src_dets = dlib.rectangles([src_rect1])
        dst_rect1 = dlib.rectangle(dst_x, dst_y, dst_x + dst_w, dst_y + dst_h)
        dst_dets = dlib.rectangles([dst_rect1])
        
        # Convert to grayscale for detection
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

        # Load dlib models
        dlib_face_landmark_dir_name = "dlib"
        dlib_face_landmark_models_list = {
            "68-point face landmark model": {
                "download_url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            },
            "5-point face landmark model": {
                "download_url": "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
            },
        }        
        if use_5_landmark:
            model = "5-point face landmark model"
        else:
            model = "68-point face landmark model"
        landmark_filepath = self.get_local_filepath(dlib_face_landmark_models_list[model]["download_url"], dlib_face_landmark_dir_name,)
        predictor = dlib.shape_predictor(landmark_filepath)  # Download from dlib model zoo

        # Get landmarks
        src_landmarks = self.get_landmarks_bypass_detector(src_gray, src_dets, predictor)
        src_points = np.float32(src_landmarks)
        print(f'src_points=\n{src_points}')

        dst_landmarks = self.get_landmarks_bypass_detector(dst_gray, dst_dets, predictor)

        # Select corresponding points (eyes and nose, for example)
        # For 5-point model: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        # src_points = np.float32(src_landmarks)
        dst_points = np.float32(dst_landmarks)
        # print(f'src_points=\n{src_points}')
        print('#############################')
        print(f'dst_points=\n{dst_points}')

        # Compute affine transform
        M = cv2.estimateAffinePartial2D(src_points, dst_points)[0]

        # Warp source face to match destination face geometry
        aligned_src = cv2.warpAffine(src_img, M, (dst_img.shape[1], dst_img.shape[0]))
        width,height = dst_img.shape[1], dst_img.shape[0]

        detector = dlib.get_frontal_face_detector() # can break since HOG detector is used instead of external
        # Also have to consider CNN based dlib detector, if this breaks. Downside of CNN is it requires GPU and higher RAM and higher compute
        aligned_src_landmarks = self.get_landmarks(aligned_src, detector, predictor)
        print('#############################')
        print(f'aligned_src_landmarks=\n{aligned_src_landmarks}')



        align_src_torch = None
      
        # Show result
        if show_facial_landmark_indications:
            # src_indicate_img = indicate_landmark(src_img, src_landmarks, mark_color=(0,0,255), mark_size=4)
            alignsrc_ind_img = self.indicate_landmark(aligned_src, aligned_src_landmarks, mark_color=(0,0,255), mark_size=4)
            # dst_indicate_img = indicate_landmark(dst_img, dst_landmarks, mark_color=(0,0,255), mark_size=4)  
            # align_src_torch = self.convert_opencv_image_to_torch(alignsrc_ind_img)
        else:
            alignsrc_ind_img = aligned_src
            # align_src_torch = self.convert_opencv_image_to_torch(aligned_src)

        if scale_based_on_target_dist and  not use_5_landmark:
            # Scale Image according to target distance
            # Work only for 68 point face landmark 
            p1_idx, p2_idx = 1,15 # [Face outermost point near left ear,  Face outermost point near right ear]
            dst_distance = self.euclidean_distance(dst_landmarks, p1_idx, p2_idx)
            
            # Scale alignsrc_ind_img based on line between points p1 and p2 to reach atleast dst_distance
            alignsrc_scaled_img = self.scale_transform(alignsrc_ind_img, aligned_src_landmarks[p1_idx], aligned_src_landmarks[p2_idx], target_dist=dst_distance)
            alignsrc_ind_img = alignsrc_scaled_img
            
        align_src_torch = self.convert_opencv_image_to_torch(alignsrc_ind_img)
        
        return(align_src_torch,M,width,height)


# Warp Transform the Mask
# This is to warp affine transform the mask
class WarpTransformMask:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",), # mask which is to be transformed.
                "MAT": ("MAT",), # Matrix Transformation that is applied on mask
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


    # def prepare_torch_mask_for_opencv(self, tensor_mask):
    #     """
    #     Takes a PyTorch mask tensor and returns a NumPy array of shape (H, W).
    #     Always picks the first element along batch/channel axes if present.
    #     Converts to uint8 for OpenCV if needed.
    #     """
        
    #     # Acceptable shapes: [H, W], [1, H, W], [B, H, W], [H, W, C], [B, H, W, C]
    #     mask_np = tensor_mask.numpy()
        
    #     # Remove batch dimension if present
    #     if mask_np.ndim == 3:
    #         # [B, H, W] or [H, W, C]
    #         if mask_np.shape[0] == 1 and mask_np.shape[1] >3:
    #             # [1, H, W] to [H, W]
    #             mask_np = mask_np[0]    # is equivalent to  mask_np[0,:,:]
    #         elif mask_np.shape[2] in [1,3]:
    #             # [H, W, 1] or [H, W, 3] to [H, W]
    #             mask_np = mask_np[:,:,0]
    #     elif mask_np.ndim == 4:
    #         # [B, H, W, C] to [H, W, C]
    #         # [B, C, H, W] to [H, W, C]
    #         mask_np = mask_np[0] # removed Batch
    #         if mask_np.shape[2] in [1,3] and mask_np.shape[0]>3 and mask_np.shape[1]>3:
    #             # [H, W, 1] or [H, W, 3] to [H, W]
    #             mask_np = mask_np[:,:,0]
    #         elif mask_np.shape[0] in [1,3] and mask_np.shape[1]>3 and mask_np.shape[2]>3:
    #             # [C, H, W] to [H, W, C]
    #             mask_np = np.transpose(mask_np, (1, 2, 0))  # New shape [H, W, C]
    #     elif mask_np.ndim == 2:
    #         pass
    #     else:
    #         raise ValueError(f"Unexpected tensor shape: {shape}")
    #     # By now mask_np should be (H, W)
    #     # mask_np is stored in a contiguous block of memory in C order (row-major order).
    #     mask_np = np.ascontiguousarray(mask_np)
        
    #     # Convert to uint8 for OpenCV if not already
    #     if mask_np.dtype != np.uint8:
    #         mask_np = mask_np.astype(np.uint8)
    #     return mask_np
    
    # def convert_to_opencv_mask(self, mask):
    #     print(f'WarpTransformMask mask ndim={mask.ndim}  shape={mask.shape}')
    #     cv_mask = None
    #     if isinstance(mask, np.ndarray):
    #         print("NumPy array (likely OpenCV or PyTorch tensor)")
    #     elif isinstance(mask, PILImage.Image):
    #         print("PIL Mask")
    #     elif type(mask).__module__.startswith('cv2'):
    #         print("OpenCV Mask")
    #     elif isinstance(mask, torch.Tensor):
    #         print("PyTorch Tensor Mask")
    #         cv_mask = self.convert_torch_mask_to_opencv(mask)           
    #     else:
    #         print("Unknown type:", type(mask))
    #     return cv_mask
       
    # def convert_torch_mask_to_opencv(self, torch_mask):
        
    #     print(f'WarpTransformMask convert_torch_mask_to_opencv() torch_mask ndim={torch_mask.ndim}  shape={torch_mask.shape}')
        
    #     # Assume 'torch_mask' is your PyTorch image tensor with shape (C, H, W)
    #     tensor = torch_mask.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients          
    #     np_mask = self.prepare_torch_mask_for_opencv(tensor)
 
    #     print("########################################################################################")
    #     print(f'convert_torch_mask_to_opencv() np_mask ndim={np_mask.ndim}  shape={np_mask.shape}')

    #     # If tensor is normalized to [0, 1], scale to [0, 255]
    #     # np_mask = (np_mask * 255).astype(np.uint8)
        
    #     return np_mask

    # def convert_opencv_mask_to_torch(self, mask):

    #     print(f'convert_opencv_mask_to_torch() mask ndim={mask.ndim}  shape={mask.shape}')
        
    # # 1. Convert to PyTorch tensor
    # mask_torch = torch.from_numpy(mask)  # shape: [H, W]

    # # 2. Add batch dimension to make shape [1, H, W]
    # mask_torch_batched = mask_torch.unsqueeze(0)  # shape: [B, H, W], here B=1

    # return mask_torch_batched

    def execute(self, mask, MAT, width, height):       
        # mask_cv = self.convert_to_opencv_mask(mask)
        mask_cv = convert_to_opencv_mask(mask)
        if mask_cv is None:
            raise ValueError("mask_cv couldn't be converted to opencv")

        # Apply the same transformation to the mask_cv
        # Convert the mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if mask_cv.ndim == 3 and mask_cv.shape[2] == 1:
            mask_cv = mask_cv[:, :, 0]
        aligned_mask = cv2.warpAffine(mask_cv, MAT, (width, height), flags=cv2.INTER_NEAREST)

            
        # aligned_mask_torch = self.convert_opencv_mask_to_torch(aligned_mask)
        aligned_mask_torch = convert_opencv_mask_to_torch(aligned_mask)
        
        return(aligned_mask_torch,)


# Find the blob that is near to the ROI, Also merge all blobs that are close to ROI. 
# Limit the bottom of Merged blob so that its doesn't extend beyond top of non merged blob
class BlobNearROI:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",), # mask
                "ROI_bbox": ("BBOX",),
                "min_blob_area": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION*MAX_RESOLUTION, "step": 1, }),
                "max_blolb_area": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION*MAX_RESOLUTION, "step": 1, }),
                "near_dist": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
            },
            "optional": {
                "padding_top": ("INT", { "default": 10, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),         
            }
        }

    # RETURN_TYPES = ("MASK", "INT", "INT", "INT", "INT",)
    # RETURN_NAMES = ("nearBlob mask", "left", "top", "width", "height",)
    RETURN_TYPES = ("MASK", "BBOX",)
    RETURN_NAMES = ("nearBlob_mask", "nearBlob_bbox",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


    def create_filtered_mask(self, binary_mask_shape, roi, near_blobs):
        """
        Create a binary mask containing only blobs that are:
        - intersecting or within ROI
        - part of near_blobs list (usually merged nearBlob blobs)
        
        Parameters:
        - binary_mask_shape: shape of original binary mask (height, width)
        - roi: (x, y, w, h) bounding box
        - near_blobs: list of blob dicts (with 'contour' keys) considered nearBlob
        
        Returns:
        - filtered_mask: binary mask highlighting only the appropriate blobs
        """
        filtered_mask = np.zeros(binary_mask_shape, dtype=np.uint8)
        rx, ry, rw, rh = roi
        roi_rect = (rx, ry, rx + rw, ry + rh)

        
        # Draw only contours meeting criteria into mask
        for blob in near_blobs:
            # Fill the blob contours
            # cv2.drawContours(filtered_mask, [blob['contour']], -1, color=255, thickness=cv2.FILLED)   # preferred mask for torch is [0.0-1.0]
            cv2.drawContours(filtered_mask, [blob['contour']], -1, color=1, thickness=cv2.FILLED)

        # np.set_printoptions(threshold=np.inf)
        # print("filtered_mask")
        # print(filtered_mask[145:275,145:275])
        # np.set_printoptions(threshold=1000)
        
        return filtered_mask

    def merge_blob_near_roi(self, binary_mask, roi, d, min_area, max_area, padding=10):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def is_intersect_or_close(bbox, roi, d):
            x, y, w, h = bbox
            rx, ry, rw, rh = roi
            ax1, ay1, ax2, ay2 = x, y, x + w, y + h
            bx1, by1, bx2, by2 = rx - d, ry - d, rx + rw + d, ry + rh + d
            return (ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1)

        # Filter blobs by area
        filtered_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                bbox = cv2.boundingRect(cnt)
                filtered_blobs.append({'contour': cnt, 'bbox': bbox, 'area': area})

        # Near blobs (intersect or close to ROI)
        near_blobs = [b for b in filtered_blobs if is_intersect_or_close(b['bbox'], roi, d)]

        if not near_blobs:
            print("No blobs found near ROI within given area range.")
            return None, None

        near_bboxes = {tuple(blob['bbox']) for blob in near_blobs}
        non_near_blobs = [blob for blob in filtered_blobs if tuple(blob['bbox']) not in near_bboxes]


        # Merge near blobs contours
        all_points = np.vstack([b['contour'] for b in near_blobs])
        x, y, w, h = cv2.boundingRect(all_points)
        near_blob_bottom = y + h

        rx, ry, rw, rh = roi
        roi_bottom = ry + rh
        expanded_roi_rect = (rx - d, ry - d, rx + rw + d, ry + rh + d)

        clip_candidates = []
        for b in non_near_blobs:
            bx, by, bw, bh = b['bbox']
            bottom = by + bh
            ax1, ay1, ax2, ay2 = bx, by, bx + bw, by + bh
            # Check intersection with expanded ROI rectangle
            rx1, ry1, rx2, ry2 = expanded_roi_rect
            intersects_expanded_roi = (ax1 < rx2 and ax2 > rx1 and ay1 < ry2 and ay2 > ry1)
            if (bottom > near_blob_bottom) and (not intersects_expanded_roi):
                clip_candidates.append(by)  # top of blob as clipping candidate

        if clip_candidates and near_blob_bottom > roi_bottom:
            candidate_clip = min(clip_candidates) - padding
            # Ensure clipping is not above ROI bottom
            clip_limit = max(roi_bottom, candidate_clip)
            # Calculate new height clipped from bottom
            # new_height = clip_limit - y # old
            # ensure bounding box bottom is never extended beyond original bottom
            clipped_bottom = min(near_blob_bottom, clip_limit)
            new_height = clipped_bottom - y        
            if new_height < 0:
                new_height = 0
            print(f"Clipping nearBlob bottom from {near_blob_bottom} to {clip_limit}")
            # h = new_height
            h = max(0, new_height)

        # Only Merged Blob and blobs within ROI
        merged_blob_mask = self.create_filtered_mask(binary_mask.shape, roi, near_blobs) 

        return merged_blob_mask, (x, y, w, h) # BoundingBox

    def execute(self, mask, ROI_bbox, min_blob_area, max_blolb_area, near_dist, padding_top=10 ):
        print("START%%%%%%%%%%%%%% BlobNearROI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%") 
        # binary_mask_cv = self.convert_to_opencv_mask(mask)
        binary_mask_cv = convert_to_opencv_mask(mask)
        if binary_mask_cv is None:
            raise ValueError("binary_mask_cv couldn't be converted to opencv")

        # Convert the mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if binary_mask_cv.ndim == 3 and binary_mask_cv.shape[2] == 1:
            binary_mask_cv = binary_mask_cv[:, :, 0]

        nearBlob, nearBlobBBx = self.merge_blob_near_roi(binary_mask_cv, ROI_bbox, near_dist, min_blob_area, max_blolb_area, padding_top)
        left, top, width, height = nearBlobBBx
        bbox_list = []
        bbox_list.append((left, top, width, height))
        print("nearBlobBBx=",nearBlobBBx)
        print(" left, top, width, height =",left, top, width, height)
        print("bbox_list=",bbox_list)
        # nearBlob_mask_torch = self.convert_opencv_mask_to_torch(nearBlob)
        nearBlob_mask_torch = convert_opencv_mask_to_torch(nearBlob)

        print("%%%%%%%%%%%%%% BlobNearROI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  END") 
        # return nearBlob_mask_torch, *nearBlobBBx
        return nearBlob_mask_torch, bbox_list

"""
To display only the blobs whose bounding boxes are fully inside the ROI or intersect it (partially overlap),
 and to merge them into a single bounding box


.Only blobs inside or intersecting with the ROI are used.
.Their bounding boxes are merged into one.
.Non-matching blobs are ignored.
.Add to the final selection all blobs whose bounding boxes are fully contained in the merged bounding box
  even if they were not part of the initial ROI/ROI-intersecting set.
.Clip the merged bounding box so it does not exceed a certain clipping_distance, from all sides of the expanded ROI 
 
.Update the list of final contours to include only those fully inside this clipped bounding box

"""

class BlobWithinROI:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",), # mask
                "ROI_bbox": ("BBOX",),
            },
            "optional": {
                "expand_roi_dist": ("INT", { "default": 10, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "clipping_dist": ("INT", { "default": 50, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),       
            }
        }

    RETURN_TYPES = ("MASK", "BBOX",)
    RETURN_NAMES = ("mergedBlob_mask", "mergedBlob_bbox",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def is_inside_or_intersects(self, bbox, roi):
        # Checks if bbox is fully inside or intersects ROI rectangle
        x, y, w, h = bbox
        rx, ry, rw, rh = roi
        ax1, ay1, ax2, ay2 = x, y, x + w, y + h
        bx1, by1, bx2, by2 = rx, ry, rx + rw, ry + rh
        intersects = (ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1)
        fully_inside = (ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2)
        return intersects or fully_inside

    def is_fully_inside(self, bbox, roi):
        # Checks if bbox is fully inside ROI rectangle
        x, y, w, h = bbox
        rx, ry, rw, rh = roi
        ax1, ay1, ax2, ay2 = x, y, x + w, y + h
        bx1, by1, bx2, by2 = rx, ry, rx + rw, ry + rh
        return (ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2)

    def merge_bounding_boxes(self, bboxes):
        if not bboxes:
            return None
        x1s, y1s, x2s, y2s = zip(*[(x, y, x + w, y + h) for (x, y, w, h) in bboxes])
        xmin, ymin = min(x1s), min(y1s)
        xmax, ymax = max(x2s), max(y2s)
        return (xmin, ymin, xmax - xmin, ymax - ymin)

    def clip_bbox_to_roi(self, bbox, clipping_roi):
        """Clip bounding box coordinates so that bbox stays within clipping_roi."""
        x, y, w, h = bbox
        cx, cy, cw, ch = clipping_roi
        x_clipped = max(x, cx)
        y_clipped = max(y, cy)
        x2_clipped = min(x + w, cx + cw)
        y2_clipped = min(y + h, cy + ch)
        w_clipped = max(0, x2_clipped - x_clipped)
        h_clipped = max(0, y2_clipped - y_clipped)
        return (x_clipped, y_clipped, w_clipped, h_clipped)

    def clip_contour_to_roi(self, contour, clipping_roi):
        """Clip contour points so they fit inside clipping_roi."""
        cx, cy, cw, ch = clipping_roi
        x2 = cx + cw
        y2 = cy + ch
        clipped_points = []
        for point in contour:
            px, py = point[0]
            px_clipped = np.clip(px, cx, x2)
            py_clipped = np.clip(py, cy, y2)
            clipped_points.append([[px_clipped, py_clipped]])
        return np.array(clipped_points, dtype=np.int32)

    def execute(self, mask, ROI_bbox, expand_roi_dist=10, clipping_dist=50):
        print("START%%%%%%%%%%%%%% BlobWithinROI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%") 
        # binary_mask_cv = self.convert_to_opencv_mask(mask)
        binary_mask_cv = convert_to_opencv_mask(mask)
        if binary_mask_cv is None:
            raise ValueError("binary_mask_cv couldn't be converted to opencv")

        # Convert the mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if binary_mask_cv.ndim == 3 and binary_mask_cv.shape[2] == 1:
            binary_mask_cv = binary_mask_cv[:, :, 0]

        # Expand ROI by expand_roi_dist
        height, width = binary_mask_cv.shape
        rx, ry, rw, rh = ROI_bbox
        print("ROI_bbox=",ROI_bbox)
        ex_rx = max(rx - expand_roi_dist, 0)
        ex_ry = max(ry - expand_roi_dist, 0)
        ex_rw = min(rw + 2 * expand_roi_dist, width - ex_rx)
        ex_rh = min(rh + 2 * expand_roi_dist, height - ex_ry)
        expanded_roi = (ex_rx, ex_ry, ex_rw, ex_rh)

        # Find contours/blobs
        contours, _ = cv2.findContours(binary_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select blobs in or intersecting expanded ROI
        selected_bboxes = []
        for cnt in contours:
            bbox = cv2.boundingRect(cnt)
            if self.is_inside_or_intersects(bbox, expanded_roi):
                selected_bboxes.append(bbox)

        # Merge bounding boxes of selected blobs
        merged_bbox = self.merge_bounding_boxes(selected_bboxes)

        # # Add all blobs that are fully inside the merged bbox
        final_contours = []
        if merged_bbox is not None:
            for cnt in contours:
                bbox = cv2.boundingRect(cnt)
                if self.is_fully_inside(bbox, merged_bbox):
                    final_contours.append(cnt)

        # Define clipping rectangle by expanding expanded ROI by clipping_dist on all sides
        clip_x1 = ex_rx - clipping_dist
        clip_y1 = ex_ry - clipping_dist
        clip_x2 = ex_rx + ex_rw + clipping_dist
        clip_y2 = ex_ry + ex_rh + clipping_dist

        # Ensure clipping rectangle is within image bounds
        clip_x1 = max(0, clip_x1)
        clip_y1 = max(0, clip_y1)
        clip_x2 = min(width, clip_x2)
        clip_y2 = min(height, clip_y2)

        clipping_rect = (clip_x1, clip_y1, max(0, clip_x2 - clip_x1), max(0, clip_y2 - clip_y1))

        # Clip merged bounding box to the clipping rectangle
        if merged_bbox is not None:
            merged_bbox = self.clip_bbox_to_roi(merged_bbox, clipping_rect)

        filtered_mask = np.zeros(binary_mask_cv.shape, dtype=np.uint8)

        # Clip contours to the clipping rectangle
        clipped_final_contours = []
        for cnt in final_contours:
            clipped_contour = self.clip_contour_to_roi(cnt, clipping_rect)
            clipped_final_contours.append(clipped_contour)
            cv2.drawContours(filtered_mask, [clipped_contour], -1, color=1, thickness=cv2.FILLED)
            # Or
            # ## Only needed if clipped contours will be empty i.e 0 width or height.
            # ## Drawing or further processing empty contours can cause errors, waste resources, or lead to artifacts in the output.
            # bbox = cv2.boundingRect(cnt)
            # # Clip bbox to clipping rect to decide if it remains visible at all
            # clipped_bbox = self.clip_bbox_to_roi(bbox, clipping_rect)
            #     if clipped_bbox[2] > 0 and clipped_bbox[3] > 0:
            #     # Clip contour points individually
            #     clipped_contour = self.clip_contour_to_roi(cnt, clipping_rect)
            #     clipped_final_contours.append(clipped_contour)

        print("merged_bbox=",merged_bbox)
        left, top, width, height = merged_bbox
        bbox_list = []
        bbox_list.append((left, top, width, height))
        print("BBx=",merged_bbox)
        print(" left, top, width, height =",left, top, width, height)
        print("bbox_list=",bbox_list)
        # mergedBlob_mask = self.convert_opencv_mask_to_torch(filtered_mask)
        mergedBlob_mask = convert_opencv_mask_to_torch(filtered_mask)

        print("%%%%%%%%%%%%%% BlobWithinROI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  END") 
        return (mergedBlob_mask, bbox_list,)





from skimage.feature import local_binary_pattern
class RefineNeckSegment:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "neck_mask": ("MASK",), # mask
                "image": ("IMAGE",), # head+neck
            },
            "optional": {
                "use_color": ("BOOLEAN", { "default": True }),
                "use_texture": ("BOOLEAN", { "default": False }),
                "threshold": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "neck_top_avoid_dist": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "scale_dress_region": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_neck",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    # # For Image

    # def prepare_for_opencv_image(self, tensor):
    #     # Remove batch dimension if present
    #     # [1, 767, 581, 3] [batch, height, width, channel]
    #     if tensor.ndim == 4:
    #         tensor = tensor.squeeze(0)
    #     shape = tensor.shape

    #     np_img = None
    #     # (C, H, W) -> (H, W, C)
    #     if tensor.ndim == 3 and shape[0] in [1, 3] and shape[1]>3 and shape[2]>3 :  # likely (C, H, W)
    #         np_img = tensor.permute(1, 2, 0).numpy()
    #     # (H, W, C) already
    #     elif tensor.ndim == 3 and shape[2] in [1, 3] and shape[0]>3 and shape[1]>3 :  # likely (H, W, C)
    #         np_img = tensor.numpy()
    #     else:
    #         raise ValueError(f"Unexpected tensor shape: {shape}")

    #     return np_img
        
    # def convert_to_opencv_image(self, img):
    #     cv_img = None
    #     if isinstance(img, np.ndarray):
    #         print("NumPy array (likely OpenCV or PyTorch tensor)")
    #     elif isinstance(img, PILImage.Image):
    #         print("PIL Image")
    #     elif type(img).__module__.startswith('cv2'):
    #         print("OpenCV image")
    #     elif isinstance(img, torch.Tensor):
    #         print("PyTorch Tensor image")
    #         cv_img = self.convert_torch_image_to_opencv(img)           
    #     else:
    #         print("Unknown type:", type(img))
    #     return cv_img
       
    # def convert_torch_image_to_opencv(self, torch_img):
        
    #     # Assume 'torch_img' is your PyTorch image tensor with shape (C, H, W)
    #     tensor = torch_img.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients
        
    #     # if tensor.ndim == 4:
    #         # # if input torch_img has shape [1, 767, 581, 3] [batch, height, width, channel]
    #         # np_img = tensor.squeeze(0).numpy()     # Step 1 & 3: remove batch, to numpy -> shape [767, 581, 3]
    #     # else:
    #         # np_img = tensor.numpy()         # Convert to numpy array
            
    #     np_img = self.prepare_for_opencv_image(tensor)
 

    #     # If tensor is normalized to [0, 1], scale to [0, 255]
    #     np_img = (np_img * 255).astype(np.uint8)

    #     # Convert from RGB to BGR for OpenCV
    #     cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
    #     return cv_img

    # def convert_opencv_image_to_torch(self, image):

    #     # Step 2: Convert BGR to RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Step 3: Convert to float32 and normalize to [0, 1] if needed
    #     image = image.astype(np.float32) / 255.0  # Remove this line if you want to keep uint8

    #     # Step 4: Convert to torch tensor
    #     tensor = torch.from_numpy(image)  # shape: [H, W, C]

    #     # Step 5: Add batch dimension
    #     tensor = tensor.unsqueeze(0)  # shape: [1, H, W, C]
                
    #     return tensor


    def refine_to_get_clean_neck_mask(self, neck_mask, img, use_color, use_lbp, threshold_value, neck_top_avoid_dist, dilate_dress_region_iterations):

        # How to Detect Texture Features to Improve Dress Detection?
        # Color difference alone may fail if dress and skin have similar colors.
        # Texture analysis helps distinguish smooth skin from textured clothes.
        # Common texture features and methods:
        # a) Local Binary Patterns (LBP)
        #     LBP captures local texture patterns by thresholding neighborhood pixels.
        #     Clothes generally have more texture variation than smooth skin
        # b) Gabor Filters
        #     Gabor filters can pick up repetitive patterns/texture orientations common in fabric.
        #     Apply filters at multiple scales and orientations, then measure the filter response magnitude.

        # Parameters for LBP
        LBP_RADIUS = 1
        LBP_POINTS = 8 * LBP_RADIUS
        LBP_METHOD = 'uniform'  # rotation-invariant uniform patterns

        # Flags to control feature use
        # use_color: Enable this flag to distinguish between neck and dress by using color
        # use_lbp:: Enable this flag to distinguish between neck and dress by using texture

        # img: Your Head+Neck image, shape [H, W, 3], can be BGR
        # neck_mask: Binary mask (uint8), where neck pixels are 1


        if img is None:
            raise ValueError(f"Image must be provided")

        if neck_mask is None:
            raise ValueError(f"Neck mask must be provided")

        # Convert mask to binary: 1 for neck pixels, 0 elsewhere
        # _, neck_mask = cv2.threshold(neck_mask, 127, 1, cv2.THRESH_BINARY)

        print("Neck Mask")
        np.set_printoptions(threshold=np.inf)
        print(neck_mask.shape)
        print(neck_mask[100:216,100:216])
        np.set_printoptions(threshold=1000)

        # Why convert to Lab color space
        # 1.Lighting Invariance: Since the L channel captures lightness and a/b channels capture color, 
        #   variations in lighting (which affect only the L channel) have less impact on color-based comparisons. 
        #   Shadows and highlights change the lightness (L) channel of Lab space 
        #   but have relatively smaller effects on the chromaticity (a/b) channels
        #   This separation allows you to compare only the a/b channels, making skin and clothing comparison less sensitive to shadows or highlights
        # 2.Device Independence: Lab is perceptually uniform and device-independent, 
        #   meaning color differences correspond more closely to human perception across different cameras or conditions
        # 3.Enhanced Discrimination: Better skin segmentation performance compared to RGB or YCbCr, especially under poor or strong lighting condition

        # Convert image to Lab color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Erode neck mask to get inner safe region for skin color sampling
        eroded_neck = cv2.erode(neck_mask, np.ones((5, 5), np.uint8), iterations=1)

        # Compute average a/b for neck skin (in eroded region)
        skin_lab_pixels = img_lab[eroded_neck == 1]
        mean_a = np.mean(skin_lab_pixels[:, 1]) if use_color else None
        mean_b = np.mean(skin_lab_pixels[:, 2]) if use_color else None

        # Find edge pixels of neck mask
        edge_mask = cv2.dilate(neck_mask, np.ones((3, 3), np.uint8)) - neck_mask
        edge_indices = np.where(edge_mask == 1)

        # Exclude top of neck region (avoid_top)
        ys, xs = np.where(neck_mask == 1)
        print("xs=",xs)
        print("ys=",ys)
        top_neck = np.min(ys)
        avoid_top_limit = top_neck + neck_top_avoid_dist  # vertical pixel row to exclude below top boundary

        # Exclude edge pixels in the avoid_top region
        # Filter to include only pixels with y > avoid_top_limit
        valid_edge_mask = (edge_indices[0] > avoid_top_limit)
        filtered_edge_indices = (edge_indices[0][valid_edge_mask], edge_indices[1][valid_edge_mask])

        if use_color:
            edge_lab_colors = img_lab[filtered_edge_indices]

            # Compute color distance in a/b only (ignore L)
            ab_dist = np.linalg.norm(edge_lab_colors[:, 1:3] - [mean_a, mean_b], axis=1)

            # Normalize color dist to [0,1]
            ab_dist_norm = (ab_dist - np.min(ab_dist)) / (np.ptp(ab_dist) + 1e-8)
        else:
            ab_dist_norm = None

        if use_lbp:
            # Convert to grayscale for LBP calculation
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate LBP image of entire grayscale image
            lbp_image = local_binary_pattern(gray_img, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)

            # Extract LBP values at filtered edge pixels
            edge_lbp_values = lbp_image[filtered_edge_indices]

            # Normalize LBP to [0,1]
            lbp_norm = (edge_lbp_values - np.min(edge_lbp_values)) / (np.ptp(edge_lbp_values) + 1e-8)
        else:
            lbp_norm = None

        def update_threshold(val):
            k_default = 1.0  # default multiplier for std deviation

            dress_edge = np.zeros_like(neck_mask, dtype=np.uint8)

            # Decide combined_feature based on flags
            if use_color and use_lbp:
                # Slider controls texture weight (0-100 -> 0.0-1.0)
                texture_weight = val / 100.0
                color_weight = 1.0 - texture_weight
                # Combine normalized features with weighted sum
                combined_feature = color_weight * ab_dist_norm + texture_weight * lbp_norm

                # Find Adaptive Threshold on combined feature:
                mean_feat = np.mean(combined_feature)
                std_feat = np.std(combined_feature)
                T = mean_feat + k_default * std_feat

                # Mark pixels exceeding threshold as dress edge
                dress_pixels = combined_feature > T

                print(f"Using combined features: color_weight={color_weight:.2f}, texture_weight={texture_weight:.2f}")
                print(f"Adaptive threshold T = {T:.4f}, mean = {mean_feat:.4f}, std = {std_feat:.4f}")

            elif use_color:
                # Slider controls threshold multiplier k to scale std dev in adaptive threshold
                k = max(val / 10.0, 0.1)  # map slider 0-100 to k=0.1-10, lower limit to avoid zero

                # Find Adaptive Threshold on color distance
                mean_feat = np.mean(ab_dist_norm)
                std_feat = np.std(ab_dist_norm)
                T = mean_feat + k * std_feat

                # Mark pixels exceeding threshold as dress edge
                dress_pixels = ab_dist_norm > T

                print(f"Using color feature only")
                print(f"Threshold multiplier k = {k:.2f}, Adaptive threshold T = {T:.4f}, mean = {mean_feat:.4f}, std = {std_feat:.4f}")

                
            elif use_lbp:
                # Slider controls threshold multiplier k for LBP only
                k = max(val / 10.0, 0.1)

                # Find Adaptive Threshold on lbp feature
                mean_feat = np.mean(lbp_norm)
                std_feat = np.std(lbp_norm)
                T = mean_feat + k * std_feat

                # Mark pixels exceeding threshold as dress edge
                dress_pixels = lbp_norm > T

                print(f"Using LBP texture feature only")
                print(f"Threshold multiplier k = {k:.2f}, Adaptive threshold T = {T:.4f}, mean = {mean_feat:.4f}, std = {std_feat:.4f}")

            else:
                raise ValueError("At least one feature flag (use_color or use_texture) must be True")



            # Look for dress pixels only on the edge
            dress_edge[filtered_edge_indices[0][dress_pixels], filtered_edge_indices[1][dress_pixels]] = 1

            # Dilate dress mask and restrict to neck mask area
            dress_mask = cv2.dilate(dress_edge, np.ones((3, 3), np.uint8), iterations=dilate_dress_region_iterations)
            dress_mask = cv2.bitwise_and(dress_mask, neck_mask)

            # Remove dress from neck mask
            clean_neck_mask = cv2.bitwise_and(neck_mask, cv2.bitwise_not(dress_mask))

            return clean_neck_mask


        return update_threshold(threshold_value)

    def execute(self, neck_mask, image, use_color, use_texture, threshold, neck_top_avoid_dist, scale_dress_region ):
        # neck_mask_cv = self.convert_to_opencv_mask(neck_mask)
        neck_mask_cv = convert_to_opencv_mask(neck_mask)
        if neck_mask_cv is None:
            raise ValueError("neck_mask_cv couldn't be converted to opencv")
  
        # src_img = self.convert_to_opencv_image(image)
        src_img = convert_to_opencv_image(image)
        if src_img is None:
            raise ValueError("image couldn't be converted to opencv")

        # Convert the neck_mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if neck_mask_cv.ndim == 3 and neck_mask_cv.shape[2] == 1:
            neck_mask_cv = neck_mask_cv[:, :, 0]

        refinedNeck_mask = self.refine_to_get_clean_neck_mask(neck_mask_cv, src_img, use_color, use_texture, threshold, neck_top_avoid_dist, scale_dress_region)

        # refinedNeck_mask_torch = self.convert_opencv_mask_to_torch(refinedNeck_mask)
        refinedNeck_mask_torch = convert_opencv_mask_to_torch(refinedNeck_mask)

        return (refinedNeck_mask_torch, )

class MaximalRectangleInsideBlob:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",), # mask
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("left", "top", "width", "height", "right", "bottom",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def largest_centered_rectangle(self, mask):
        """
        Find the largest rectangle inside a binary mask (blob), 
        centered around the blob centroid, by expanding outwards until hitting zeros.

        Args:
            mask (np.ndarray): 2D binary array where 1 represents blob pixels.

        Returns:
            tuple: (left, top, width, height) of the largest centered rectangle.
                Returns None if the mask is empty.
        """
        h, w = mask.shape
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None  # Empty blob
        
        # Find centroid (mid point)
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        
        # Initialize rectangle boundaries at the centroid
        left = cx
        right = cx
        top = cy
        bottom = cy

        while True:
            left_expanded = right_expanded = top_expanded = bottom_expanded = False

            # Try expand left
            if left > 0 and np.all(mask[top:bottom+1, left-1] == 1):
                left -= 1
                left_expanded = True

            # Try expand right
            if right < w - 1 and np.all(mask[top:bottom+1, right+1] == 1):
                right += 1
                right_expanded = True

            # Try expand top
            if top > 0 and np.all(mask[top-1, left:right+1] == 1):
                top -= 1
                top_expanded = True
            
            # Try expand bottom
            if bottom < h - 1 and np.all(mask[bottom+1, left:right+1] == 1):
                bottom += 1
                bottom_expanded = True

            # Stop if no expansion in any direction
            if not (left_expanded or right_expanded or top_expanded or bottom_expanded):
                break

        width = right - left + 1
        height = bottom - top + 1
        return (left, top, width, height, right, bottom)

    def execute(self, mask ): 
        # binary_mask_cv = self.convert_to_opencv_mask(mask)
        binary_mask_cv = convert_to_opencv_mask(mask)
        if binary_mask_cv is None:
            raise ValueError("binary_mask_cv couldn't be converted to opencv")

        # Convert the mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if binary_mask_cv.ndim == 3 and binary_mask_cv.shape[2] == 1:
            binary_mask_cv = binary_mask_cv[:, :, 0]

        left, top, width, height, right, bottom = self.largest_centered_rectangle(binary_mask_cv)
        print("MaximalRectangleInsideBlob left, top, width, height, right, bottom  =",left, top, width, height, right, bottom )
        return left, top, width, height, right, bottom 

import mediapipe as mp
# from mediapipe.tasks import vision
from mediapipe.tasks.python import vision

class MediapipeImageSegmenter:

    mediapipe_model_list = {
        "Multi-class Selfie": {
            "categories": ["background", "hair", "body_skin", "face_skin", "clothes","others"],
            "model_url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        },
        "Selfie": {
            "categories": ["background", "person"],
            "model_url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        },
        "Hair": {
            "categories": ["background", "hair"],
            "model_url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
        },
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (list(MediapipeImageSegmenter.mediapipe_model_list.keys()),),
            },
            "optional": {
                "post_processing": ("BOOLEAN", { "default": True }),
                # "threshold": ("INT", { "default": 150, "min": 0, "max": 255, "step": 1, }),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK",)
    RETURN_NAMES = ("background", "hair", "person", "body_skin", "face_skin", "clothes", "others")
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


    def torch_to_mediapipe_image(self, torch_img):
        """
        Converts a PyTorch image tensor to a MediaPipe Image.
        Args:
            torch_img: PyTorch tensor of shape [C, H, W], values range [0,1] or [0,255].
        """
        
        # 1. Move to CPU and convert to numpy
        if torch_img.device != torch.device('cpu'):
            torch_img = torch_img.cpu()
        
        np_img = torch_img.numpy()
        
        # 2. Convert to (H, W, C)
        shape = np_img.shape
        # Check if first dimension is 1 (typical batch dimension)
        if len(shape) == 4 and shape[0] == 1 and shape[3] in [1,3]:
            # Remove batch dimension, convert (1, H, W, C) -> (H, W, C)
            np_img = np.squeeze(np_img, axis=0)
        elif len(shape) == 3 and shape[0] in [1,3] and shape[1]>3 and shape[2]>3:     
            # (C, H, W) -> (H, W, C)
            np_img = np.transpose(np_img, (1, 2, 0))
        
        # 3. Scale values to [0,255] and uint8 if needed
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
            
        # 4. Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
        return mp_image

    def convert_opencv_mask_to_torch(self, mask):

        print(f'convert_opencv_mask_to_torch() mask ndim={mask.ndim}  shape={mask.shape}')
        
        # 1. Convert to PyTorch tensor
        mask_torch = torch.from_numpy(mask)  # shape: [H, W]

        # 2. Add batch dimension to make shape [1, H, W]
        mask_torch_batched = mask_torch.unsqueeze(0)  # shape: [B, H, W], here B=1

        return mask_torch_batched

    def post_processing(self, mask, threshold):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # filtered = cv2.inRange(mask, np.array([threshold]), np.array([255]))
        new_mask = cv2.erode(mask, kernel, iterations=2)
        new_mask = cv2.dilate(new_mask, kernel, iterations=2)
        new_mask = cv2.GaussianBlur(new_mask, (3, 3), 0)
        return new_mask

    def get_local_filepath(self, url, dirname, local_file_name=None):
        # Determine the Local File Name
        if not local_file_name:
            parsed_url = urlparse(url)
            local_file_name = os.path.basename(parsed_url.path)

        # Check for an Existing File via get_full_path
        destination = folder_paths.get_full_path(dirname, local_file_name)
        if destination:
            print(f"using destination model: {destination}")
            return destination

        # Ensure the Target Directory Exists
        folder = os.path.join(folder_paths.models_dir, dirname)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Download the File if Needed
        destination = os.path.join(folder, local_file_name)
        if not os.path.exists(destination):
            print(f"downloading {url} to {destination}")
            download_url_to_file(url, destination)
        return destination

    def execute(self, image, model_name, post_processing=True, threshold=0.5):


        print("MediapipeImageSegmenter, image.shape", image.shape)
        FIXED_CATEGORIES = ["background", "hair", "person", "body_skin", "face_skin", "clothes", "others"]

        model_info = MediapipeImageSegmenter.mediapipe_model_list.get(model_name)
        mp_model_url = model_info.get("model_url", None)
        
        mp_model_filepath = self.get_local_filepath(mp_model_url, "mediapipe",)
        # Create a image segmenter instance with the image mode:
        options = vision.ImageSegmenterOptions(
            # base_options=mp.tasks.BaseOptions(model_asset_path='models/selfie_multiclass_256x256.tflite'),
            base_options=mp.tasks.BaseOptions(model_asset_path=mp_model_filepath),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True)
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            # # Load the input image from a numpy array.
            # numpy_image = cv2.imread('images/ComfyUI_temp_obpzb_00002_.png')
            # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
            # Suppose `image` is a PyTorch tensor from ComfyUI node (shape: [3,H,W], float)
            mp_image = self.torch_to_mediapipe_image(image)

            height, width, channel = mp_image.numpy_view().shape

            # Segment the image
            segmented_masks = segmenter.segment(mp_image)

            model_categories = model_info.get("categories", [])
            
            # Map each model category to its index in the model's category list (no normalization)
            model_cat_to_index = {cat: i for i, cat in enumerate(model_categories)}

            outputs = []
            for fixed_cat in FIXED_CATEGORIES:
                if fixed_cat in model_cat_to_index:
                    idx = model_cat_to_index[fixed_cat]
                    
                    # Get the prediction of each category
                    cat_confidence_seg = segmented_masks.confidence_masks[idx] #float within the range [0,1]
                    # cat_mask = np.uint8(cat_confidence_seg.numpy_view().copy() * 255) # convert to uint8

                    # Create binary boolean mask: True where arr > threshold, else False
                    cat_mask_bool = cat_confidence_seg.numpy_view() > threshold

                    # Convert boolean mask to integer mask (0 or 1)
                    bin_cat_mask = cat_mask_bool.astype(np.uint8)                    
  
                    torch.set_printoptions(threshold=float('inf'))
                    print("cat_confidence_seg mask=",bin_cat_mask[:40,:30])
                    torch.set_printoptions(profile='default')                          

                    if post_processing:
                        # Post process the mask to make this model look at least a little bit better
                        bin_cat_mask = self.post_processing(bin_cat_mask,threshold)

                    cat_torch_mask = self.convert_opencv_mask_to_torch(bin_cat_mask)
                else:
                    cat_torch_mask = torch.zeros((1, height, width), dtype=torch.uint8) # [B,H,W]
                    
                outputs.append(cat_torch_mask)

            return tuple(outputs)
  
# IMAGE THRESHOLD MASK NODE by aj_dev

class Image_Threshold_Mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.004}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "image_threshold_mask"

    CATEGORY = "essentials/image manipulation"


    # Tensor to PIL
    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    # PIL to Tensor
    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def image_threshold_mask(self, image, threshold=0.5):
        images = []
        for img in image:
            images.append(self.pil2tensor(self.apply_threshold(self.tensor2pil(img), threshold)))
        return (torch.cat(images, dim=0), )

    def apply_threshold(self, input_image, threshold=0.5):
        # Convert the input image to grayscale
        grayscale_image = input_image.convert('L')

        # Apply the threshold to the grayscale image
        threshold_value = int(threshold * 255)
        thresholded_image = grayscale_image.point(
            lambda x: 255 if x >= threshold_value else 0, mode='L')

        return thresholded_image

class BBox_Padding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "top": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "right": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "bottom": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
            },
            "optional": {
                "padding_left": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "padding_top": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }), 
                "padding_right": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }), 
                "padding_bottom": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),          
            }
        }

    RETURN_TYPES = ("INT","INT","INT","INT","INT","INT",)
    RETURN_NAMES = ("left","top","right","bottom","width","height")
    FUNCTION = "execute"

    CATEGORY = "essentials/image manipulation"

    def execute(self, image, left, top, right, bottom, padding_left, padding_top, padding_right, padding_bottom):
        if image.ndim == 4:
            _, height, width, _ = image.shape
        elif image.ndim == 3:
            height, width, _ = image.shape

        # padding
        new_left = max(left-padding_left,0)
        new_top = max(top-padding_top,0)
        new_right = min(right+padding_right,width-1)
        new_bottom = min(bottom+padding_bottom,height-1)
        new_width = 1+new_right-new_left
        new_height = 1+new_bottom-new_top

        return new_left, new_top, new_right, new_bottom, new_width, new_height

class BBox_to_BBox_Parameters:    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bbox": ("BBOX",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("left", "top", "width", "height", "right", "bottom",)
    FUNCTION = "convert"
    CATEGORY = "essentials/image manipulation"

    def convert(self, bbox ): 
        if bbox is None:
            raise ValueError("bbox must be provided")

        # If bbox is a list containing one tuple, unpack it
        if isinstance(bbox, list):
            if len(bbox) == 1 and isinstance(bbox[0], tuple):
                bb = bbox[0]
            else:
                # Unexpected list length or content
                raise ValueError(f"Expected list with one tuple for bbox input, got: {bbox}")
        elif isinstance(bbox, tuple):
            bb = bbox
        else:
            raise TypeError(f"Expected bbox input to be tuple or list of one tuple, got: {type(bbox)}")

        if len(bb) < 4:
            raise ValueError(f"BBox tuple must have at least 4 elements, got: {bb}")

        left, top, width, height = bb[:4]

        if width <= 0:
            right = 0
        else:
            right = (width-1) + left

        if height <= 0:
            bottom = 0
        else:        
            bottom = (height-1) + top
        
        print("BBox_to_BBox_Parameters left, top, width, height, right, bottom  =",left, top, width, height, right, bottom )
        return left, top, width, height, right, bottom     

class BBox_Parameters_to_BBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "top": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "right": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "bottom": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "width": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "convert"

    CATEGORY = "essentials/image manipulation"

    def convert(self, left, top, right, bottom, width, height):
        # print(f"BBox_Parameters_to_BBox left={left}, top={top}, right={right}, bottom={bottom}, width={width}, height={height}")
        print("START%%%%%%%%%%%%%% BBox_Parameters_to_BBox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        if width == 0 and right == 0 and height == 0 and bottom == 0:
            bbox_list = [(0, 0, 0, 0)]
            print("bbox_list=",bbox_list)
            return bbox_list
        if width == 0 and right == 0:
            raise ValueError(f"Bbox width or right, either is required")
        if height == 0 and bottom == 0:
            raise ValueError(f"Bbox height or bottom, either is required")
        
        if width == 0:
            width = 1 + right - left
         
        if height == 0:
            height = 1 + bottom - top
           
        # bbox = BBox(left, top, right, bottom, width, height)
        # bbox = BBOXT(left, top, right, bottom, width, height)
        bbox_list = []
        bbox_list.append((left, top, width, height))
        print("bbox_list=",bbox_list)
        # print("BBox_Parameters_to_BBox return (left, top, width, height)=",left, top, width, height)
        print("%%%%%%%%%%%%%% BBox_Parameters_to_BBox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  END")

        return bbox_list

class Combine_BBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_right": ("INT", { "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "max_bottom": ("INT", { "min": 0, "max": MAX_RESOLUTION, "step": 1, }), 
            },
            "optional": {
                "bbox1": ("BBOX",),
                "bbox2": ("BBOX",),
                "bbox3": ("BBOX",),
                "bbox4": ("BBOX",),
                "bbox5": ("BBOX",),
                # "bbox1": ("BBOXT",),
                # "bbox2": ("BBOXT",),
                # "bbox3": ("BBOXT",),
                # "bbox4": ("BBOXT",),
                # "bbox5": ("BBOXT",),                
                "padding_left": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "padding_top": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }), 
                "padding_right": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }), 
                "padding_bottom": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),          
            }
        }

    RETURN_TYPES = ("INT","INT","INT","INT","INT","INT",)
    RETURN_NAMES = ("left","top","right","bottom","width","height")
    FUNCTION = "combine"

    CATEGORY = "essentials/image manipulation"

    def combine(self, max_right, max_bottom, bbox1=None, bbox2=None, bbox3=None, bbox4=None, bbox5=None, padding_left=0, padding_top=0, padding_right=0, padding_bottom=0):
        # bbox = BBox(left, top, right, bottom, width, height)

        # Collect all valid BBox objects from inputs — ignore None
        # bboxes = [bbx for bbx in [bbox1, bbox2, bbox3, bbox4, bbox5]  if isinstance(bbx, BBOXT)]
        # bboxes = [bbx for bbx in [bbox1, bbox2, bbox3, bbox4, bbox5] if bbx is not None]
        print("START%%%%%%%%%%%%%% Combine_BBoxes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        bboxes = []
        print("combine bbox1=",bbox1)
        print("combine bbox2=",bbox2)
        print("combine bbox3=",bbox3)

        # Create a valid bbx list
        for bbx in [bbox1, bbox2, bbox3, bbox4, bbox5]:
            if bbx is None:
                continue

            # If bbx is a list containing one tuple, unpack it
            if isinstance(bbx, list):
                if len(bbx) == 1 and isinstance(bbx[0], tuple):
                    bb = bbx[0]
                else:
                    # Unexpected list length or content
                    raise ValueError(f"Expected list with one tuple for bbox input, got: {bbx}")
            elif isinstance(bbx, tuple):
                bb = bbx
            else:
                raise TypeError(f"Expected bbox input to be tuple or list of one tuple, got: {type(bbx)}")

            if len(bb) < 4:
                raise ValueError(f"BBox tuple must have at least 4 elements, got: {bb}")

            left, top, width, height = bb[:4]
            left = 0 if left < 0 else left
            top = 0 if top < 0 else top

            # not allowing if width or height is 0
            if width <= 0 or height <= 0:
                continue

            print("combine bb=",bb)
            right = (width-1) + left 
            bottom = (height-1) + top

            bboxes.append((left, top, width, height, right, bottom))
            print(bboxes)


        # for bbx in [bbox1, bbox2, bbox3, bbox4, bbox5]:
        #     print("bbx instance = ", type(bbx) )

        if not bboxes:
            # No bbox provided
            raise ValueError("At least one bbox must be provided")

        # new_left = max(min(bbx.left for bbx in bboxes) - padding_left, 0)
        # new_top = max(min(bbx.top for bbx in bboxes) - padding_top, 0)
        # new_right = min(max(bbx.right for bbx in bboxes) + padding_right, max_right)
        # new_bottom = min(max(bbx.bottom for bbx in bboxes) + padding_bottom, max_bottom)


        new_left = max(min(bbx[0] for bbx in bboxes) - padding_left, 0)
        new_top = max(min(bbx[1] for bbx in bboxes) - padding_top, 0)
        new_right = min(max(bbx[4] for bbx in bboxes) + padding_right, max_right)
        new_bottom = min(max(bbx[5] for bbx in bboxes) + padding_bottom, max_bottom)
    
        new_width = 1 + new_right-new_left
        new_height = 1 + new_bottom-new_top

        # print(f"Combine_BBoxes new_left={new_left}, new_top={new_top}, new_right={new_right}, new_bottom={new_bottom}, new_width={new_width}, new_height={new_height}")


        print("%%%%%%%%%%%%%% Combine_BBoxes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  END")
        return new_left, new_top, new_right, new_bottom, new_width, new_height

class MaskCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "op": (["union (max)", "intersection (min)", "difference", "multiply", "multiply_alpha", "add", "greater_or_equal", "greater"],),
                "clamp_result": (["yes", "no"],),
                "round_result": (["no", "yes"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("Mask",)
    FUNCTION = "combine_masks"

    CATEGORY = "essentials/image manipulation"

    def masks2common(self, m1: torch.Tensor, m2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Ensure both masks have shape (batch, height, width)
        if len(m1.size()) == 2:
            m1 = m1.unsqueeze(0)
        if len(m2.size()) == 2:
            m2 = m2.unsqueeze(0)
        # Match batch size
        m1s, m2s = m1.size(), m2.size()
        if m1s[0] < m2s[0]:
            m1 = m1.repeat(m2s[0], 1, 1)
        elif m1s[0] > m2s[0]:
            m2 = m2.repeat(m1s[0], 1, 1)
        return m1, m2
    
    def combine_masks(self, mask1, mask2, op, clamp_result, round_result):
        mask1, mask2 = self.masks2common(mask1, mask2)

        if op == "union (max)":
            result = torch.max(mask1, mask2)
        elif op == "intersection (min)":
            result = torch.min(mask1, mask2)
        elif op == "difference":
            result = mask1 - mask2
        elif op == "multiply":
            result = mask1 * mask2
        elif op == "multiply_alpha":
            mask1 = tensor2rgba(mask1)
            mask2 = tensor2mask(mask2)
            result = torch.cat((mask1[:, :, :, :3], (mask1[:, :, :, 3] * mask2).unsqueeze(3)), dim=3)
        elif op == "add":
            result = mask1 + mask2
        elif op == "greater_or_equal":
            result = torch.where(mask1 >= mask2, 1., 0.)
        elif op == "greater":
            result = torch.where(mask1 > mask2, 1., 0.)

        if clamp_result == "yes":
            result = torch.min(torch.max(result, torch.tensor(0.)), torch.tensor(1.))
        if round_result == "yes":
            result = torch.round(result)

        print("combine_masks() result.shape=",result.shape)
        return (result,)

# Find the common and matching features or keypoints in both images 
# and then derive the top_left corner of src in dest image.
class Find_BBox_of_Src_in_Dest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",), # src img  is the image which needs to be found
                "dest": ("IMAGE",), # dst img is the image from where to find the src img
            },
            "optional": {
                "distance_threshold": ("INT", { "default": 30 }),   # lower distance better accuracy
                "min_matching_points": ("INT", { "default": 16 }),
                "output_matches": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE", "BOOLEAN")
    RETURN_NAMES = ("bbox", "matches","status")
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def prepare_for_opencv(self, tensor):
        # Remove batch dimension if present
        # [1, 767, 581, 3] [batch, height, width, channel]
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        shape = tensor.shape

        np_img = None
        # (C, H, W) -> (H, W, C)
        if tensor.ndim == 3 and shape[0] in [1, 3] and shape[1]>3 and shape[2]>3 :  # likely (C, H, W)
            np_img = tensor.permute(1, 2, 0).numpy()
        # (H, W, C) already
        elif tensor.ndim == 3 and shape[2] in [1, 3] and shape[0]>3 and shape[1]>3 :  # likely (H, W, C)
            np_img = tensor.numpy()
        else:
            raise ValueError(f"Unexpected tensor shape: {shape}")

        return np_img
        
    def convert_to_opencv_image(self, img):
        print(f'img ndim={img.ndim}  shape={img.shape}')
        cv_img = None
        if isinstance(img, np.ndarray):
            print("NumPy array (likely OpenCV or PyTorch tensor)")
        elif isinstance(img, PILImage.Image):
            print("PIL Image")
        elif type(img).__module__.startswith('cv2'):
            print("OpenCV image")
        elif isinstance(img, torch.Tensor):
            print("PyTorch Tensor image")
            cv_img = self.convert_torch_image_to_opencv(img)           
        else:
            print("Unknown type:", type(img))
        return cv_img
       
    def convert_torch_image_to_opencv(self, torch_img):
        
        print(f'convert_torch_image_to_opencv() torch_img ndim={torch_img.ndim}  shape={torch_img.shape}')
        
        # Assume 'torch_img' is your PyTorch image tensor with shape (C, H, W)
        tensor = torch_img.cpu().detach()  # Ensure tensor is on CPU and not tracking gradients
        
        # if tensor.ndim == 4:
            # # if input torch_img has shape [1, 767, 581, 3] [batch, height, width, channel]
            # np_img = tensor.squeeze(0).numpy()     # Step 1 & 3: remove batch, to numpy -> shape [767, 581, 3]
        # else:
            # np_img = tensor.numpy()         # Convert to numpy array
            
        np_img = self.prepare_for_opencv(tensor)
 
        print("########################################################################################")
        print(f'convert_torch_image_to_opencv() np_img ndim={np_img.ndim}  shape={np_img.shape}')

        # If tensor is normalized to [0, 1], scale to [0, 255]
        np_img = (np_img * 255).astype(np.uint8)

        # Convert from RGB to BGR for OpenCV
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        return cv_img

    def convert_opencv_image_to_torch(self, image):

        print(f'opencv_image_to_torch() image ndim={image.ndim}  shape={image.shape}')
        # Step 2: Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 3: Convert to float32 and normalize to [0, 1] if needed
        image = image.astype(np.float32) / 255.0  # Remove this line if you want to keep uint8

        # Step 4: Convert to torch tensor
        tensor = torch.from_numpy(image)  # shape: [H, W, C]

        # Step 5: Add batch dimension
        tensor = tensor.unsqueeze(0)  # shape: [1, H, W, C]
                
        return tensor

    def group_similar_number(self, counter, near_definition=3):

        # Step 1: Get sorted unique numbers
        sorted_numbers = sorted(counter.keys())
        print("Sorted unique:", sorted_numbers)

        # Step 2: Group where gaps <= 3 (no single count > 2 check needed since we're grouping)
        groups = []
        i = 0
        while i < len(sorted_numbers):
            start = sorted_numbers[i]
            end = start
            group_count = counter[start]
            print(f"group_count({start}):{group_count}")
            j = i + 1
            
            # Expand group while gap <= 3
            while j < len(sorted_numbers) and sorted_numbers[j] - end <= near_definition:
                end = sorted_numbers[j]
                group_count += counter[end]
                print(f"\t end({end}):{group_count}")
                j += 1
            
            groups.append((start, end, group_count))
            i = j

        print("\nStep 2 - Groups (start, end, count):")
        for g in groups:
            print(f"  {g}")

        # Step 3: Final - average of start+end as integer, with total count
        final_counter = Counter()
        for start, end, count in groups:
            avg = (start + end) // 2  # integer average
            final_counter[avg] = count

        print("final_counter:",final_counter)
        
        return final_counter

    def execute(self, src, dest, distance_threshold=30, min_matching_points=16, output_matches=False):
        
        src_img = self.convert_to_opencv_image(src)
        if src_img is None:
            raise ValueError("src_img couldn't be converted to opencv")

        dst_img = self.convert_to_opencv_image(dest)
        if dst_img is None:
            raise ValueError("dst_img couldn't be converted to opencv")
 
      
        # Convert to grayscale for detection
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) # Usually Cropped image with the person (grayscale)
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY) # Usually full background image (grayscale)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and descriptors in both images
        kps, dess = orb.detectAndCompute(dst_gray, None) # keypoints and descriptors of dest
        kpd, desd = orb.detectAndCompute(src_gray, None) # keypoints and descriptors of src

        # Initialize Brute Force Matcher and perform the matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(dess, desd) # (queryDescriptors,trainDescriptors)

        # Sort matches based on their distance (best matches first)
        matches = sorted(matches, key=lambda m: m.distance)

        # Filter out matches that has distance > 30. 
        # Also matches should have atleast 16 matching points, if not make leniency in distance
        # lesser the distance  more accurate the match
        count = 0
        good_matches = []
        for m in matches:
            if m.distance < distance_threshold:
                good_matches.append(m)
                count = count + 1
            elif m.distance >= distance_threshold:
                if count < min_matching_points:
                    good_matches.append(m)
                    count = count + 1
                else:
                    break

        print("len(good_matches)=", len(good_matches))

        src_origin_x_pts_in_dest = []
        src_origin_y_pts_in_dest = []
        for i,m in enumerate(good_matches):
            ikps = int(kps[m.queryIdx].pt[0]), int(kps[m.queryIdx].pt[1])
            ikpd = int(kpd[m.trainIdx].pt[0]), int(kpd[m.trainIdx].pt[1])
            # cv2.circle(img2_color, center=ikpd, radius=1, color=(0, i, 255), thickness=-1)
            # cv2.circle(dst_img, center=ikps, radius=1, color=(0, i, 255), thickness=-1)

            # ikpd is offset from orgin in src_gray
            # Subtract this offset from matching point in ikps to get the location of src_gray top left corner in dst_gray
            src_orgin_in_dest = (ikps[0]-ikpd[0], ikps[1]-ikpd[1])

            # Add to the list, so that we can find the actual point either using mean or mode
            src_origin_x_pts_in_dest.append(src_orgin_in_dest[0])
            src_origin_y_pts_in_dest.append(src_orgin_in_dest[1])

            print(f"  {i+1} \t src={ikpd}, dest={ikps}, left_top_in_dest={src_orgin_in_dest}, distance={m.distance}")


        bbox_list = [(0, 0, 0, 0)]
        img_matches = np.zeros(shape=[4, 4, 3], dtype=np.uint8) # sample empty img 
        img_matches_torch = self.convert_opencv_image_to_torch(img_matches)

        # Find MODE: most frequent point using Counter
        if len(src_origin_x_pts_in_dest)>3:
            x_point_counts = Counter(src_origin_x_pts_in_dest)
            y_point_counts = Counter(src_origin_y_pts_in_dest)
            print("x_point_counts=",x_point_counts)
            print("y_point_counts=",y_point_counts)

            # most_common(n)    n is how many of the most common elements to retrieve.
            print("common x=",x_point_counts.most_common(1)[0][0]) # [(1041, 5), (1043, 3), (1040, 2), (1042, 2), (1039, 1)]
            print("common y=",y_point_counts.most_common(1)[0][0]) # [(1041, 5), (1043, 3), (1040, 2), (1042, 2), (1039, 1)]

            # if most common x_point_counts  counter <=3
            if(x_point_counts.most_common(1)[0][1]<=3):
                x_point_counts = self.group_similar_number(x_point_counts, near_definition=3)
                # after grouping 
                if(x_point_counts.most_common(1)[0][1]<=3):
                    print("Couldn't find sufficient matching points")            
                    return(bbox_list, img_matches_torch, False)

            # if most common y_point_counts counter <=3
            if(y_point_counts.most_common(1)[0][1]<=3):
                y_point_counts = self.group_similar_number(y_point_counts, near_definition=3)
                # after grouping
                if(y_point_counts.most_common(1)[0][1]<=3):
                    print("Couldn't find sufficient matching points")            
                    return(bbox_list, img_matches_torch, False)
            
            # Top_left_corner or orgin of src_gray in dst_gray
            top_left = (x_point_counts.most_common(1)[0][0], y_point_counts.most_common(1)[0][0])
            h2, w2 = src_gray.shape[:2]  # src_gray height, width
            # Calculate bottom-right corner
            bottom_right = (top_left[0] + w2, top_left[1] + h2)


            if w2 == 0 and h2 == 0:
                bbox_list = [(0, 0, 0, 0)]
            
            else:            
                bbox_list=[(top_left[0], top_left[1], w2, h2)]
                     
            # Show result
            if output_matches:                       
                # Optionally, draw the matches for visualization
                img_matches = cv2.drawMatches(dst_gray, kps, src_gray, kpd, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                img_matches_torch = self.convert_opencv_image_to_torch(img_matches)
        else:
            # raise ValueError("Couldn't find sufficient matching points")
            print("Couldn't find sufficient matching points")            
            return(bbox_list, img_matches_torch, False)
        
        return(bbox_list, img_matches_torch, True)


class Load_POSE_KEYPOINT:
    @classmethod
    def INPUT_TYPES(s):
        output_dir = folder_paths.get_output_directory()
        # files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        files = []
        for f in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, f)):
                extension = f.split('.')[-1]
                if extension in ['json','txt']:
                    files.append(f)
        return {"required":
                    {"json_file": (sorted(files), {"json_upload": True})},
                }

    CATEGORY = "essentials/image manipulation"

    RETURN_TYPES = ("POSE_KEYPOINT", )
    RETURN_NAMES = ("kps",)
    FUNCTION = "load_keypoints"

    def load_keypoints(self, json_file):
        output_dir = folder_paths.get_output_directory()
        json_path = os.path.join(output_dir, json_file)
        with open(json_path, 'r') as file:
            data = json.load(file)
            return (data)


    """ Purpose : Tells ComfyUI when to re-execute the node """
    @classmethod
    def IS_CHANGED(s, json_file):
        """Returns file hash or modification time"""
        output_dir = folder_paths.get_output_directory()
        json_path = os.path.join(output_dir, json_file)
        m = hashlib.sha256()
        with open(json_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    """ Purpose : Prevents crashes by validating inputs before execution """
    @classmethod
    def VALIDATE_INPUTS(s, json_file):
        """Check if file exists, return error string if invalid"""
        output_dir = folder_paths.get_output_directory()
        json_path = os.path.join(output_dir, json_file)
        if not os.path.exists(json_path):
            return "Invalid json file: {}".format(json_file)

        return True


 

# GLOBAL Variables
# COCO 18 BODY indices
NOSE = 0
NECK = 1
R_SHOULDER = 2
L_SHOULDER = 5
R_EYE = 14
L_EYE = 15
R_EAR = 16
L_EAR = 17
R_HIP =  8
L_HIP = 11


class GetNeckSegment:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",), # openpose keypoints in json format
                "person_mask": ("MASK",), # mask of person
            },
            "optional": {
                "face_mask":  ("MASK",),  # mask of person's face 
                "clothes_mask":  ("MASK",),  # mask of person's clothes 
                "neck_polygon_mode": (["nose_neck_ortho", "hip_side_width", "torso_mid_ortho"], { "default": "torso_mid_ortho" }),
            }
        }


    RETURN_TYPES = ("MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("neck_segment", "l_nck_apx", "r_nck_apx", "nck_dirxn")
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


  

    def line_intersection(self, P1, v1, P2, v2):
        """
        Finds the intersection of two lines represented by points and direction vectors.
        
        P1, P2: Points on Line 1 and Line 2 respectively, each represented as a tuple (x, y).
        v1, v2: Direction vectors for Line 1 and Line 2 respectively, each represented as a tuple (vx, vy).
        
        Returns the intersection point (x, y) as a tuple, or None if no intersection exists.
        """
        # Convert points and direction vectors into numpy arrays for matrix operations
        P1 = np.array(P1)
        P2 = np.array(P2)
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # Set up the matrix A and vector B
        A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
        B = P2 - P1


        # Check if the determinant of matrix A is non-zero (lines are not parallel)
        det_A = np.linalg.det(A)
        if det_A == 0:
            print("The lines are parallel and do not intersect.")
            return None
        
        # Solve for t and s using matrix inversion
        t_s = np.linalg.solve(A, B)
        t = t_s[0]  # t for Line 1, s for Line 2
        
        # Find the intersection point by substituting t into the parametric equation of Line 1
        intersection_point = P1 + t * v1
        
        return tuple(intersection_point)

    def distance_btwn_points(self, p1,p2):
        vec = np.array(p1) - np.array(p2)  # Points DOWN from dip to tip    
        distance = np.linalg.norm(vec)
        return distance
        
    # Distance ALONG the traversal direction
    def path_progress(self, p1,p2,direction_unit):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.dot(p2 - p1, direction_unit)

    # When traversing A→B→C, to find which side is L.
    # Applying Cross product sign gives the oriented angle from AB to AL. 
    # Positive = L is on left, negative = L is on right 
    def side_of_line(self, A, B, L):
        # Vector AB = B-A
        AB = B-A    
        # Vector AL = L-A  
        AL = L-A    
        # Cross product: AB × AL
        cross_z = np.cross(AB, AL)
        
        # cross_z >  0: "LEFT"
        # cross_z <  0: "RIGHT"
        # cross_z == 0: "ON_LINE"
        return cross_z

    # Bresenham's Line Algorithm: This will allow us to traverse pixel by pixel
    def bresenham_line(self, p1, p2):
        """
        Bresenham's line algorithm to return the list of pixels between p1 and p2.
        p1, p2: tuples (x, y)
        Returns a list of (x, y) coordinates for pixels on the line
        """
        x1, y1 = p1
        x2, y2 = p2
        
        pixels = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            pixels.append((x1, y1))
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        return pixels

    def is_target_near_neighbours(self, contour, mask, target_values):
        h, w = mask.shape
        is_near = True

        points = contour.squeeze()
        for point in points: # (N,2)
            # print("Point:", point)
            x = int(point[0])
            y = int(point[1])

            # Skip boundary pixels
            if x <= 0 or y <= 0 or x >= w-1 or y >= h-1:
                continue

            # Get 8-connected neighbors
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:  # Skip center
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if mask[ny, nx] not in target_values:
                            is_near = False
                            break
                if is_near == False:
                    break
            if is_near == False:
                break
        return is_near

    def fill_contours_whose_neighbors_only_have_target(self, mask, contours, hierarchy, target_values=255, fill_color=255):
        h, w = mask.shape
        result_mask = mask.copy()

        # print("contours len=",len(contours))
        for i, contour in enumerate(contours):
            no_blemish = True
            h = hierarchy[i]
            #  hierarchy = [Next, Previous, First_Child, Parent]
            if h[3] == -1:  # Outer contour (no parent)

                # skip degenerates having only less than 2 point
                if(contour.shape[0]<2):
                    continue

                no_blemish = self.is_target_near_neighbours(contour, mask, target_values)
                # skip contour that has neighbours other than target
                if no_blemish == False:
                    continue
                
                # Inner hole
                child_idx = h[2]
                # if children exist
                while child_idx != -1:
                    hole_contour = contours[child_idx]

                    # only contour's that have more than 1 point
                    if(hole_contour.shape[0]>1):
                        no_blemish = self.is_target_near_neighbours(hole_contour, mask, target_values)

                        # If hole_contour has neighbours other than target, stop looking for children
                        if no_blemish == False:
                            break

                    child_idx = hierarchy[child_idx][0]  # Next sibling

                # Fill contour if it's neighbors has only target 
                if no_blemish == True:
                    # contourIdx	Parameter indicating a contour to draw. If it is negative, all the contours are drawn. 
                    cv2.drawContours(result_mask, [contour], -1, fill_color, thickness=-1)
    
        return result_mask

    def create_neck_polygon(self, pose_pixels, pose_kps, h, w):
        """Create neck polygon using orthogonal lines at nose/neck with (eye,ear,sholder)-based width"""

        global NOSE, NECK, R_SHOULDER, L_SHOULDER, R_EYE, L_EYE, R_EAR, L_EAR
        # Check confidence
        if pose_kps[NOSE, 2] < 0.5 or pose_kps[NECK, 2] < 0.5:
            return None, None, None
        
        eye_width = ear_width = sholdr_width = 0

        nose_pt = pose_pixels[NOSE].astype(int)
        neck_pt = pose_pixels[NECK].astype(int)
        
        # Eye points for width measurement (fallback if low confidence)
        if pose_kps[R_EYE, 2] > 0.3 and pose_kps[L_EYE, 2] > 0.3:
            r_eye = pose_pixels[R_EYE].astype(int)
            l_eye = pose_pixels[L_EYE].astype(int)
            eye_width = np.linalg.norm(r_eye - l_eye)

        # Ear points for width measurement (fallback if low confidence)
        if pose_kps[R_EAR, 2] > 0.3 and pose_kps[L_EAR, 2] > 0.3:
            r_ear = pose_pixels[R_EAR].astype(int)
            l_ear = pose_pixels[L_EAR].astype(int)
            ear_width = np.linalg.norm(r_ear - l_ear)

        # Shoulder points for width measurement
        if pose_kps[R_SHOULDER, 2] > 0.3 and pose_kps[L_SHOULDER, 2] > 0.3:
            r_sholdr = pose_pixels[R_SHOULDER].astype(int)
            l_sholdr = pose_pixels[L_SHOULDER].astype(int)
            sholdr_width = np.linalg.norm(r_sholdr - l_sholdr)

        neck_width = max(eye_width,ear_width,sholdr_width)

        # Neck vector (nose -> neck)
        # Example: nose at (300, 200)
        # Example: neck at (310, 250)
        # Result: [310-300, 250-200] = [10, 50]
        neck_vec = np.array(neck_pt) - np.array(nose_pt)

        # Purpose of neck vector in Neck Polygon
        # This vector defines the neck orientation. Next steps use it to:
        # Normalize → get unit length vector
        # Perpendicular → rotate 90° for width direction: [-y, x] = [-50, 10]
        # Offset → create left/right boundaries parallel to neck

        neck_length = np.linalg.norm(neck_vec)
        
        if neck_length == 0:
            return None, None, None
        
        # Unit perpendicular vector (rotate 90 degrees)
        perp_unit = np.array([-neck_vec[1], neck_vec[0]]) / neck_length
        
        # Create 4 polygon points
        half_width = neck_width / 2
        
        # Nose end points (orthogonal)
        nose_left = nose_pt + perp_unit * half_width
        nose_right = nose_pt - perp_unit * half_width
        
        # Neck end points (orthogonal)  
        neck_left = neck_pt + perp_unit * half_width
        neck_right = neck_pt - perp_unit * half_width

        # BOUNDS CHECK: Final polygon vertices
        polygon_points = [nose_left, nose_right, neck_right, neck_left]
        clamped_points = []
        for pt in polygon_points:
            clamped_pt = (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
            clamped_points.append(clamped_pt)
        
        polygon = np.array(clamped_points, dtype=np.int32) 
        
        return polygon, nose_pt, neck_pt

    def create_neck_polygon2(self, pose_pixels, pose_kps, h, w):
        """Create neck polygon with hip-oriented side lines and (eye,ear,sholder)-based width"""
        
        global NOSE, NECK, R_SHOULDER, L_SHOULDER, R_EYE, L_EYE, R_EAR, L_EAR, R_HIP, L_HIP
        
        # Check confidence
        if pose_kps[NOSE, 2] < 0.5 or pose_kps[NECK, 2] < 0.5:
            return None, None, None
        
        eye_width = ear_width = sholdr_width = hip_width = 0

        nose_pt = pose_pixels[NOSE].astype(int)
        neck_pt = pose_pixels[NECK].astype(int)
        
        # Eye points for width measurement 
        if pose_kps[R_EYE, 2] > 0.3 and pose_kps[L_EYE, 2] > 0.3:
            r_eye = pose_pixels[R_EYE].astype(int)
            l_eye = pose_pixels[L_EYE].astype(int)
            eye_width = np.linalg.norm(r_eye - l_eye)
        
        # Ear points for width measurement
        if pose_kps[R_EAR, 2] > 0.3 and pose_kps[L_EAR, 2] > 0.3:
            r_ear = pose_pixels[R_EAR].astype(int)
            l_ear = pose_pixels[L_EAR].astype(int)
            ear_width = np.linalg.norm(r_ear - l_ear)

        # Shoulder points for width measurement
        if pose_kps[R_SHOULDER, 2] > 0.3 and pose_kps[L_SHOULDER, 2] > 0.3:
            r_sholdr = pose_pixels[R_SHOULDER].astype(int)
            l_sholdr = pose_pixels[L_SHOULDER].astype(int)
            sholdr_width = np.linalg.norm(r_sholdr - l_sholdr)

        neck_width = max(eye_width,ear_width,sholdr_width)

        # Hip points for torso orientation reference
        if pose_kps[R_HIP, 2] > 0.3 and pose_kps[L_HIP, 2] > 0.3:
            r_hip = pose_pixels[R_HIP].astype(int)
            l_hip = pose_pixels[L_HIP].astype(int)
            hip_width = np.linalg.norm(r_hip - l_hip)
    
        # Hip orientation vector (neck_pt → mid_hip)
        mid_hip = (r_hip + l_hip) // 2 if hip_width > 0 else neck_pt
        torso_vec = np.array(mid_hip) - np.array(neck_pt)  # Points DOWN from neck to hips
        
        torso_length = np.linalg.norm(torso_vec)
        if torso_length == 0:
            # Fallback to neck vector if no hip data
            torso_vec = np.array(neck_pt) - np.array(nose_pt)
            torso_length = np.linalg.norm(torso_vec)
        
        # Side orientation = perpendicular to torso (hip-based)
        side_unit = np.array([-torso_vec[1], torso_vec[0]]) / torso_length  # Rotate 90° LEFT
        
        # Create 4 polygon points
        half_width = neck_width / 2
        
        # Top line: centered at nose_pt, perpendicular to torso
        nose_left = nose_pt + side_unit * half_width
        nose_right = nose_pt - side_unit * half_width
        
        # Bottom line: centered at neck_pt, perpendicular to torso  
        neck_left = neck_pt + side_unit * half_width
        neck_right = neck_pt - side_unit * half_width

        # BOUNDS CHECK: Final polygon vertices
        polygon_points = [nose_left, nose_right, neck_right, neck_left]
        clamped_points = []
        for pt in polygon_points:
            clamped_pt = (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
            clamped_points.append(clamped_pt)
        
        polygon = np.array(clamped_points, dtype=np.int32)

        return polygon, nose_pt, neck_pt

    def create_neck_polygon3(self, pose_pixels, pose_kps, h, w):
        """Create neck polygon using torso midline and shoulder width"""

        global NOSE, NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP

        # Check confidence for required points
        required_points = [NOSE, NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP]
        if any(pose_kps[i, 2] < 0.5 for i in required_points):
            return None, None, None
        
        nose_pt = pose_pixels[NOSE].astype(int)
        neck_pt = pose_pixels[NECK].astype(int)
        r_shoulder = pose_pixels[R_SHOULDER].astype(int)
        l_shoulder = pose_pixels[L_SHOULDER].astype(int)
        r_hip = pose_pixels[R_HIP].astype(int)
        l_hip = pose_pixels[L_HIP].astype(int)
        
        # STEP 1: Torso midline (neck_pt → mid_hip)
        mid_hip = ((r_hip + l_hip) // 2).astype(int)

        torso_midline_vec = np.array(mid_hip) - np.array(neck_pt)
        torso_length = np.linalg.norm(torso_midline_vec)
        torso_direction = torso_midline_vec / torso_length
        
        # STEP 2: Bottom line = shoulder width at neck_pt
        bottom_left = r_shoulder  # Right shoulder (adjust if needed based on view)
        bottom_right = l_shoulder  # Left shoulder
        
        # STEP 3: Side lines parallel to torso midline from shoulders
        # Extend side lines sufficiently long (2x torso length)
        side_length = np.linalg.norm(torso_midline_vec) * 2
        
        # Left side line: from bottom_left parallel to torso_direction
        left_side_end = (bottom_left + side_length * torso_direction).astype(int)
        
        # Right side line: from bottom_right parallel to torso_direction  
        right_side_end = (bottom_right + side_length * torso_direction).astype(int)
        
        # STEP 4: Top line orthogonal to torso midline at nose_pt
        # Perpendicular to torso_direction
        perp_torso = np.array([-torso_direction[1], torso_direction[0]])
        
        # Find intersection points: solve line-line intersections
        # print("In create_neck_polygon3(), nose_pt=",nose_pt, " bottom_left=",bottom_left, " bottom_right=" ,bottom_right)
        top_left = self.line_intersection(nose_pt, perp_torso, bottom_left, torso_direction)
        top_right = self.line_intersection(nose_pt, perp_torso, bottom_right, torso_direction)
        
        if top_left is None or top_right is None:
            return None, None, None

        # BOUNDS CHECK: Final polygon vertices
        polygon_points = [top_left, top_right, bottom_right, bottom_left]
        clamped_points = []
        for pt in polygon_points:
            clamped_pt = (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
            clamped_points.append(clamped_pt)
        
        polygon = np.array(clamped_points, dtype=np.int32)
        
        return polygon, nose_pt, neck_pt

    def create_face_polygon(self, face_kps, face_pixels, w, h):
        """ Helper to create a polygon by listing only face points """
        FACE_PARTS = {
            0: 'Jaw_R0', 8: 'Chin', 16: 'Jaw_L0',
            17: 'RBrow0', 19: 'RBrowPeak', 21: 'RBrowInner',
            22: 'LBrow0', 24: 'LBrowPeak', 26: 'LBrowInner',
            27: 'NoseT', 30: 'NoseTip', 35: 'NoseB',
            36: 'REyeOut', 39: 'REyeCen', 41: 'REyeIn',
            42: 'LEyeOut', 45: 'LEyeCen', 47: 'LEyeIn',
            48: 'ULipOutL', 54: 'ULipOutR', 59: 'LLipOutL', 53: 'LLipOutR',
            60: 'ULipInL', 64: 'ULipInR', 67: 'LLipInL', 63: 'LLipInR'
        }
        # Face Keypoints Mapping (0-69)
        # 0-16:  Jawline (right to left)
        # 17-21: Right eyebrow
        # 22-26: Left eyebrow  
        # 27-30: Nose bridge
        # 31-35: Nose tip/bottom
        # 36-41: Right eye (outer→inner)
        # 42-47: Left eye (outer→inner)
        # 48-59: Outer lips (upper→lower)
        # 60-67: Inner lips (upper→lower)

        # Ensure face_poly is a numpy array of shape (N, 2) of type np.int32

        # Face Jawline
        face_points = []
        for i in range(17):  # 0-16 jawline
            if i < len(face_pixels) and face_kps[i, 2] > 0.3:  # Confidence check
                pt = tuple(face_pixels[i].astype(int))
                face_points.append(pt)
        # Left eyebrow, Right eyebrow
        for i in range(26,16,-1):  # 26-17 eye brow in reversed order for polygon to close properly
            if i < len(face_pixels) and face_kps[i, 2] > 0.3:  # Confidence check
                pt = tuple(face_pixels[i].astype(int))
                face_points.append(pt)
        
        face_poly = np.array(face_points, dtype=np.int32)
        return face_poly

    def create_torso_polygon(self, pose_pixels, w, h, sh_buf=10, hp_buf=50):
        # Extend 10px buffer for shoulder and 50px buffer for hips

        global R_SHOULDER, L_SHOULDER, R_HIP, L_HIP

        # right shoulder, left shoulder,  left hip, right hip
        idx = [R_SHOULDER, L_SHOULDER, L_HIP, R_HIP]  # indices for the torso polygon (shoulders and hips)
        # Ensure torso_poly is a numpy array of shape (N, 2) of type np.int32
        buff_torso_pixels = []

        # Right Shoulder
        x, y = pose_pixels[R_SHOULDER]
        x = int(x)-sh_buf if int(x)-sh_buf > 0 else 1
        y = int(y)-sh_buf if int(y)-sh_buf > 0 else 1 
        buff_torso_pixels.append([x,y])

        # Left Shoulder
        x, y = pose_pixels[L_SHOULDER]
        x = int(x)+sh_buf if int(x)+sh_buf < w else w-1
        y = int(y)-sh_buf if int(y)-sh_buf > 0 else 1 
        buff_torso_pixels.append([x,y])

        # Left  Hip
        x, y = pose_pixels[L_HIP]
        x = int(x)+hp_buf if int(x)+hp_buf < w else w-1
        y = int(y)+hp_buf if int(y)+hp_buf < h else h-1
        buff_torso_pixels.append([x,y])

        # Right Hip
        x, y = pose_pixels[R_HIP]
        x = int(x)-hp_buf if int(x)-hp_buf > 0 else 1
        y = int(y)+hp_buf if int(y)+hp_buf < h else h-1
        buff_torso_pixels.append([x,y])
        
        
        torso_poly = np.array(buff_torso_pixels, dtype=np.int32)
        return torso_poly

    def polygon_from_mask(self, mask, epsilon_percent=0.01):
        polygon = None

        # Find contours 
        # RETR_EXTERNAL for outer boundary only
        # CHAIN_APPROX_SIMPLE reduces the contour's vertices to minimum required
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (main object)
        if contours:
            # each contour is of shape (N,1,2) , N: Number of points,  1: Single channel (like grayscale image), 2: x,y coordinates per point
            # key applies cv2.contourArea() on each contour in the list.
            largest_contour = max(contours, key=cv2.contourArea) # Returns the contour with largest area

            # Approximate to fewer vertices (epsilon=1% of perimeter)
            epsilon = epsilon_percent * cv2.arcLength(largest_contour, True)

            # approxPolyDP(	curved_input, epsilon, closed)
            # closed: If true, the approximated curve is closed (its first and last vertices are connected)
            # approxPolyDP() is to simplify contours by removing points while controlling deviation with epsilon. 
            # Epsilon defines the maximum allowed distance between any original contour point and the simplified polygon's edges.
            polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon = polygon.reshape(-1, 2) # reshape to [N,2]

        return polygon

    def hand_segment_mask(self, person_mask,
                            pose_kps, pose_pixels,
                            lhand_kps, lhand_pixels,
                            rhand_kps, rhand_pixels,
                            torso_poly, face_poly):

        global NOSE, NECK, R_SHOULDER, L_SHOULDER
    
        h, w = person_mask.shape

        # Find Hand mask
        hand_mask = np.zeros_like(person_mask)
        person_cpy = person_mask.copy() # where ever person is present, it's value = 255
        bgrd_colr = 0
        left_colr = 10
        rght_colr = 20
        lft2_colr = 30
        rht2_colr = 40
        fill_colr = 50
        face_colr = 60
        tors_colr = 70
        pers_colr = 255

        if torso_poly is not None and len(torso_poly) > 3:
            cv2.fillPoly(person_cpy, pts=[torso_poly], color=(tors_colr))
            
        
        if face_poly is not None and len(face_poly) > 3:
            cv2.fillPoly(person_cpy, pts=[face_poly], color=(face_colr))


        # Define paths through the skeleton (predefined paths)
        hand_paths = {
            # Right hand: shoulder→wrist→fingertip
            'right_arm': [(R_SHOULDER, 3), (3, 4) ],  # Right arm from shoulder to wrist 
            'right_pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],    # Wrist→MCP→Tip
            'right_ring': [(0, 13), (13, 14), (14, 15), (15, 16)],     # Wrist→MCP→Tip
            'right_middle': [(0, 9), (9, 10), (10, 11), (11, 12)],     # Wrist→MCP→Tip
            'right_index': [(0, 5), (5, 6), (6, 7), (7, 8)],           # Wrist→MCP→Tip
            'right_thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],           # Wrist→MCP→Tip
            
            # Left hand: shoulder→wrist→fingertip
            'left_arm': [(L_SHOULDER, 6), (6, 7) ],  # Left arm from shoulder to wrist 
            'left_pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],     # Wrist→MCP→Tip
            'left_ring': [(0, 13), (13, 14), (14, 15), (15, 16)],      # Wrist→MCP→Tip
            'left_middle': [(0, 9), (9, 10), (10, 11), (11, 12)],      # Wrist→MCP→Tip
            'left_index': [(0, 5), (5, 6), (6, 7), (7, 8)],            # Wrist→MCP→Tip
            'left_thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],            # Wrist→MCP→Tip
        }

        side_sign = 1 # for right side +1 and for left side -1
        eps = 0.01

        neck_pt = center = None
        # Neck point or Nose or center of the image is considered as Body's Center 
        # to know whether we are moving towards the body or away from it
        if pose_kps[NECK, 2] >= 0.3:
            tmp_pt = pose_pixels[NECK].astype(int)
            if(tmp_pt[0]>eps and tmp_pt[1]>eps):
                center = neck_pt = tmp_pt
        elif pose_kps[NOSE, 2] >= 0.3:
            tmp_pt = pose_pixels[NOSE].astype(int)
            if(tmp_pt[0]>eps and tmp_pt[1]>eps):
                center = nose_pt = tmp_pt
        else:
            center = np.array([w//2,h//2],dtype=np.int32)

        # Find the width of fingers
        finger_width = w
        arm_width = 0    
        dip_pt = pip_pt = None
        for path_name, path in hand_paths.items():
            distal_length = middle_length = proximal_length = 0
            # skip not fingers path
            if path_name in ["right_arm","left_arm"]:
                continue
            elif path_name in ['right_pinky', 'right_ring','right_middle', 'right_index',  'right_thumb']:
                pixels = rhand_pixels
            elif path_name in ['left_pinky', 'left_ring','left_middle', 'left_index',  'left_thumb']:
                pixels = lhand_pixels

            # 4th path of finger is DIP to Tip
            dip, tip = path[3][0], path[3][1]
            tmp1_pt = pixels[dip].astype(int)
            tmp2_pt = pixels[tip].astype(int)
            if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and tmp2_pt[0]>eps and tmp2_pt[1]>eps):
                dip_pt = tmp1_pt
                tip_pt = tmp2_pt
                distal_length = self.distance_btwn_points(tip_pt,dip_pt)
                if(distal_length>1):
                    finger_width = min(finger_width, distal_length)
            
            # 3rd path of finger is PIP to DIP
            pip = path[2][0]
            tmp1_pt = pixels[pip].astype(int)
            if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and dip_pt is not None):
                pip_pt = tmp1_pt
                middle_length = self.distance_btwn_points(dip_pt,pip_pt)

            # 2nd  path of finger is MCP to PIP
            mcp = path[1][0]
            tmp1_pt = pixels[mcp].astype(int)
            if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and pip_pt is not None):
                mcp_pt = tmp1_pt
                proximal_length = self.distance_btwn_points(pip_pt,mcp_pt)
            
            if distal_length > 0 and middle_length > 0 and proximal_length > 0:
                arm_width = max(arm_width, (distal_length + middle_length + proximal_length))


        # find the width of half of torso
        half_torso_width = w
        tmp1_pt = pose_pixels[R_SHOULDER].astype(int)
        tmp2_pt = pose_pixels[L_SHOULDER].astype(int)
        if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and tmp2_pt[0]>eps and tmp2_pt[1]>eps):
            rshd_pt = tmp1_pt
            lshd_pt = tmp2_pt
            if neck_pt is not None:
                half_torso_width = min(half_torso_width,  self.distance_btwn_points(rshd_pt,neck_pt))
                half_torso_width = min(half_torso_width,  self.distance_btwn_points(lshd_pt,neck_pt))
            else:
                half_torso_width = self.distance_btwn_points(rshd_pt,lshd_pt)//2

        # hard set if could not find width of arm, finger or torso
        if arm_width < 1:
            arm_width = 90
        if finger_width >= w:
            finger_width = 18
        if half_torso_width >= w:
            half_torso_width = 120

        good_arm_width = good_arm_width_left = good_arm_width_rght = w
        

        # Primary Traverse each predefined path
        for path_name, path in hand_paths.items():
            if path_name in ['right_pinky', 'right_ring','right_middle', 'right_index',  'right_thumb']:
                pixels = rhand_pixels
                kps = rhand_kps
                side_sign = 1
                min_width = finger_width
                max_width = finger_width*2
            elif path_name in ['right_arm']:
                pixels = pose_pixels
                kps = pose_kps
                side_sign = 1
                min_width = arm_width
                max_width = arm_width
            elif path_name in ['left_pinky', 'left_ring','left_middle', 'left_index',  'left_thumb']:
                pixels = lhand_pixels
                kps = lhand_kps
                side_sign = -1
                min_width = finger_width
                max_width = finger_width*2
            elif path_name in ['left_arm']:
                pixels = pose_pixels
                kps = pose_kps
                side_sign = -1
                min_width = arm_width
                max_width = arm_width
            else:
                continue


            for i in range(len(path)):
                s, e = path[i][0], path[i][1]
                if kps[s,2] > 0.3 and kps[e,2] > 0.3:
                    p1 = pixels[s].astype(int)
                    p2 = pixels[e].astype(int)
                    x1 = p1[0]
                    y1 = p1[1]
                    x2 = p2[0]
                    y2 = p2[1]
                    

                    if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                        # Calculate direction vector between consecutive skeleton points
                        edge_vector = np.array(p1) - np.array(p2)
                        magnitude = np.linalg.norm(edge_vector)

                        # Normalize direction vector
                        direction_unit = edge_vector / magnitude

                        # Perpendicular direction (normalized)
                        perpendicular_unit = np.array([-direction_unit[1], direction_unit[0]])

                        # Traverse pixel by pixel along the line from p1 to p2
                        line_pts = self.bresenham_line(p1,p2)

                        # Find the side sign based on where the point is wrt neck pt 
                        side_sign = 1 if self.side_of_line(p1, p2, center) >= 0 else -1

                        # Determine finger length based on how far it is from neck
                        distance_from_center = self.distance_btwn_points(p1, center)
                        # if not close to center, then let finger width be maximum
                        if(distance_from_center > half_torso_width):
                            allowed_width = max_width
                        else:
                            allowed_width = min_width

                        # Traverse the edge
                        current_point_count = 0
                        min_left_cnt = w
                        min_rght_cnt = w
                        good_arm_width_cnt = 0
                        for current_point in line_pts:

                            left_traversal_brokn_by_out = False
                            rght_traversal_brokn_by_out = False

                            # LEFT traversal
                            left_count = 0
                            while True:
                                left_pt = current_point + side_sign * perpendicular_unit * (left_count+1)
                                x, y = left_pt
                                lx, ly = int(x), int(y)

                                if lx < 0 or ly < 0 or lx >= w or ly >= h:
                                    break
                                if person_mask[ly, lx] == 0:
                                    left_traversal_brokn_by_out = True
                                    break

                                # Check if inside face polygon (if collision, stop traversal)
                                # A 10px buffer
                                if face_poly is not None and cv2.pointPolygonTest(face_poly, (lx,ly), measureDist= True) + 10 >= 0:
                                    break
                                
                                # width gets too long
                                if left_count > allowed_width:
                                    break

                                left_count += 1

                                # Check if inside torso polygon , skip to next point
                                # pointPolygonTest returns whethers the point is inside (1), or outside (-1), or on the edge (0) of the polygon
                                if torso_poly is not None and cv2.pointPolygonTest(torso_poly, (lx,ly), False) >= 0:
                                    continue
                                # No need to overdraw when already traversed and marked
                                if person_cpy[ly, lx] == left_colr or person_cpy[ly, lx] == rght_colr:
                                    continue

                                cv2.circle(person_cpy, (lx, ly), 3, (left_colr), -1)  # Filled circle 
                                

                            # RIGHT traversal
                            right_count = 0
                            while True:
                                right_pt = current_point - side_sign * perpendicular_unit * (right_count+1)
                                x, y = right_pt
                                rx, ry = int(x), int(y)

                                if rx < 0 or ry < 0 or rx >= w or ry >= h:
                                    break
                                if person_mask[ry, rx] == 0:
                                    rght_traversal_brokn_by_out = True
                                    break
                                
                                # pointPolygonTest returns whethers the point is inside (1), or outside (-1), or on the edge (0) of the polygon
                                if torso_poly is not None and cv2.pointPolygonTest(torso_poly, (rx,ry), False) >= 0:
                                    break
                                # Check if inside face polygon (if collision, stop traversal)
                                # A 15px buffer
                                if face_poly is not None and  cv2.pointPolygonTest(face_poly, (rx,ry), measureDist= True) + 15 >= 0:
                                    break
                                # width gets too long
                                if right_count > allowed_width:
                                    break

                                right_count += 1

                                # No need to overdraw when already traversed and marked
                                if person_cpy[ry, rx] == left_colr or person_cpy[ry, rx] == rght_colr:
                                    continue

                                cv2.circle(person_cpy, (rx, ry), 3, (rght_colr), -1)  # Filled circle 
                            
                            # Find the arm width, if traversal to left and right of skeleton ends in outside of the person mask and 
                            # if we can get likewise continously for atleast 5 pixel points. Then we can consider that as a minimum good arm width
                            if( left_traversal_brokn_by_out and rght_traversal_brokn_by_out and
                                left_count > finger_width and right_count > finger_width and path_name in ['right_arm','left_arm']
                                ):
                                good_arm_width_cnt += 1
                                min_left_cnt = min(min_left_cnt, left_count)
                                min_rght_cnt = min(min_rght_cnt, right_count)
                            else:
                                good_arm_width_cnt = 0
                                min_left_cnt = w
                                min_rght_cnt = w

                            if good_arm_width_cnt > 5:
                                if (min_left_cnt + min_rght_cnt) <  good_arm_width:
                                    good_arm_width_left = min_left_cnt
                                    good_arm_width_rght = min_rght_cnt
                                    good_arm_width = good_arm_width_left + good_arm_width_rght



                        # Draw edges of skeleton
                        cv2.line(person_cpy, (x1, y1), (x2, y2), (left_colr), 3)


        # Secondary Traverse only arm, to fill arm_width
        if finger_width < good_arm_width < w :
            for path_name, path in hand_paths.items():
                if path_name in ["right_arm","left_arm"]:
                    pixels = pose_pixels
                    kps = pose_kps

                    for i in range(len(path)):
                        s, e = path[i][0], path[i][1]
                        if kps[s,2] > 0.3 and kps[e,2] > 0.3:
                            p1 = pixels[s].astype(int)
                            p2 = pixels[e].astype(int)
                            x1 = p1[0]
                            y1 = p1[1]
                            x2 = p2[0]
                            y2 = p2[1]                        

                            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                                # Calculate direction vector between consecutive skeleton points
                                edge_vector = np.array(p1) - np.array(p2)
                                magnitude = np.linalg.norm(edge_vector)

                                # Normalize direction vector
                                direction_unit = edge_vector / magnitude

                                # Perpendicular direction (normalized)
                                perpendicular_unit = np.array([-direction_unit[1], direction_unit[0]])

                                # Traverse pixel by pixel along the line from p1 to p2
                                line_pts = self.bresenham_line(p1,p2)

                                # Find the side sign based on where the point is wrt neck pt 
                                side_sign = 1 if self.side_of_line(p1, p2, center) >= 0 else -1

                                for current_point in line_pts:
                                    # LEFT traversal
                                    left_count = 0
                                    while True:
                                        left_pt = current_point + side_sign * perpendicular_unit * (left_count+1)
                                        x, y = left_pt
                                        lx, ly = int(x), int(y)

                                        if lx < 0 or ly < 0 or lx >= w or ly >= h:
                                            break

                                        # Check if inside face polygon (if collision, stop traversal)
                                        # A 15px buffer
                                        if face_poly is not None and cv2.pointPolygonTest(face_poly, (lx,ly), measureDist= True) + 10 >= 0:
                                            break
                                        
                                        # width gets too long
                                        if left_count > good_arm_width_left:
                                            break

                                        left_count += 1

                                        if person_mask[ly, lx] == 0:
                                            continue
                                        # No need to overdraw when already traversed and marked
                                        if person_cpy[ly, lx] == left_colr or person_cpy[ly, lx] == rght_colr:
                                            continue

                                        cv2.circle(person_cpy, (lx, ly), 2, (lft2_colr), -1)  # Filled circle 

                                    # RIGHT traversal
                                    right_count = 0
                                    while True:
                                        right_pt = current_point - side_sign * perpendicular_unit * (right_count+1)
                                        x, y = right_pt
                                        rx, ry = int(x), int(y)

                                        if rx < 0 or ry < 0 or rx >= w or ry >= h:
                                            break
                                        
                                        # Check if inside face polygon (if collision, stop traversal)
                                        # A 15px buffer
                                        if face_poly is not None and cv2.pointPolygonTest(face_poly, (rx,ry), measureDist= True) + 15 >= 0:
                                            break
                                        # width gets too long
                                        if right_count > good_arm_width_rght:
                                            break

                                        right_count += 1

                                        if person_mask[ry, rx] == 0:
                                            continue
                                        # No need to overdraw when already traversed and marked
                                        if person_cpy[ry, rx] == left_colr or person_cpy[ry, rx] == rght_colr:
                                            continue

                                        cv2.circle(person_cpy, (rx, ry), 2, (rht2_colr), -1)  # Filled circle 

        if  face_poly is not None or torso_poly is not None:

            # debug:
            print(f"person_cpy shape: {person_cpy.shape}, dtype: {person_cpy.dtype}")
            print(f"Unique values: {np.unique(person_cpy)}")  # Should be [0, 255]
            print(f"Non-zero pixels: {np.count_nonzero(person_cpy)}")
            # # Visualize
            # cv2.imshow("person_cpy", person_cpy)
            # cv2.waitKey(0)

            # 2nd pass Find contours that are still not filled
            _th, person_cpy2 = cv2.threshold(person_cpy, 127, 255, cv2.THRESH_BINARY)
            # # Outer Boundary only
            # contours, _ = cv2.findContours(person_cpy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # filled_mask =  fill_contours_with_target_neighbors(person_cpy,contours,target_values=[bgrd_colr,left_colr,rght_colr,pers_colr,lft2_colr,rht2_colr],fill_color=fill_colr)

            # debug:
            print(f"person_cpy2 shape: {person_cpy2.shape}, dtype: {person_cpy2.dtype}")
            print(f"Unique values: {np.unique(person_cpy2)}")  # Should be [0, 255]
            print(f"Non-zero pixels: {np.count_nonzero(person_cpy2)}")

            # or Outer Boundary and Inner Hole
            contours, hierarchy = cv2.findContours(person_cpy2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # debug
            print(f"Found {len(contours)} contours")
            print("hierarchy.shape=",hierarchy.shape)  # (1, N, 4)

            hierarchy = hierarchy.squeeze()
            target=[bgrd_colr,left_colr,rght_colr,pers_colr,lft2_colr,rht2_colr]
            person_cpy =  self.fill_contours_whose_neighbors_only_have_target(person_cpy,contours,hierarchy, target_values=target,fill_color=fill_colr)

        # Get only the grayscale that are in this range
        lower = np.array([left_colr], dtype=np.uint8)   # Min value
        upper = np.array([fill_colr], dtype=np.uint8)   # Max value
        hand_mask = cv2.inRange(person_cpy, lower, upper)

        # Remove non person parts from hand mask
        hand_mask = cv2.bitwise_and(hand_mask, person_mask)
        return hand_mask

    def extract_polygon_region_from_binary(self, binary_img, polygon):
        """
        Extract polygon region from binary image.
        binary_img: 255=foreground, 0=background (or vice versa)
        polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        Returns: extracted polygon region as binary image
        """
        # Convert points to required format: list of arrays
        pts = np.array(polygon, np.int32)
        
        # Create mask same size as input image
        mask = np.zeros(binary_img.shape[:2], dtype=np.uint8)
        
        # Fill polygon with white (255)
        cv2.fillPoly(mask, [pts], 255)
        
        # Apply mask to extract polygon region
        extracted = cv2.bitwise_and(binary_img, binary_img, mask=mask)
        
        return extracted

    def is_foreground(self, pixel, threshold=0):
        """
        Returns True if pixel differs from black background (0,0,0).
        """
        # Handle numpy scalars (uint8, int32, etc.)
        if np.isscalar(pixel):
            channels = [int(pixel)]
        # Handle numpy arrays
        elif hasattr(pixel, '__array__') or isinstance(pixel, np.ndarray):
            channels = pixel.tolist()
        # Handle lists/tuples
        elif isinstance(pixel, (list, tuple)):
            channels = list(pixel)
        else:
            # Native Python int/float
            channels = [int(pixel)]
        
        # Foreground if max channel exceeds threshold
        return max(channels) > threshold

    def offset_along_line(self, point, unit_vector, distance):
        """Offset point by distance along unit vector direction"""
        return (
            point[0] + distance * unit_vector[0],
            point[1] + distance * unit_vector[1]
        )

    def sort_points_by_distance(self, points, ref):
        """Sort points nearest to ref first - FASTEST method"""
        return sorted(points, key=lambda p: (p[0]-ref[0])**2 + (p[1]-ref[1])**2)

    def concave_points_of_polygon(self, polygon, step=5):
        """
        polygon: Nx2 array of points (CCW)
        step: spacing for stability (larger = smoother)
        returns: list of concave points # list of indices of concave points
        """
        n = len(polygon)
        concave = []

        for i in range(n):
            # Get the previous, current, and next points (cyclically)
            p_prev = polygon[(i - step) % n]
            p_curr = polygon[i]
            p_next = polygon[(i + step) % n]

            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            # 2D cross product (determinant) to determine the direction of the turn
            cross = np.cross(v1, v2)

            if cross < 0:  # concave for CCW contour
                concave.append(p_curr)
                # concave.append(i)

        return concave

    def find_concave_nearest_to_ref(self, polygon, ref_pt):
        concave_pts = self.concave_points_of_polygon(polygon, step=1)
        concave_pts_nearest = self.sort_points_by_distance(concave_pts, ref_pt)

        if len(concave_pts_nearest) > 0:
            return concave_pts_nearest[0]
        else:
            return None

    def smallest_bounding_box_radial(self, P, r, w, h, img_shape=None, refside="left"):
        """
        Calculate SMALLEST bounding box containing ALL possible object positions
        given point P with radial uncertainty r.
        
        Args:
            P: Tuple (px, py) - reference point
            r: Radial uncertainty (pixels)
            w, h: Object width, height
            img_shape: Optional (height, width) for image bounds
        
        Returns: (min_x, min_y, max_x, max_y) - smallest enclosing box
        """
        px, py = P
        
        # Step 1: Find extreme left_midpoint positions (circle radius r around P)
        # All possible left_midpoints form a CIRCLE of radius r around P
        
        # Step 2: For each possible left_midpoint, object extends:
        # - Left: 0px (left edge = left_midpoint_x)
        # - Right: w pixels
        # - Top: -h/2 pixels  
        # - Bottom: +h/2 pixels
        
        # Step 3: Find EXTREME positions across entire circle:
        if refside == "left":
            # LEFTMOST possible left edge: (P_x - r)
            leftmost_left_edge = px - r
            
            # RIGHTMOST possible right edge: (P_x + r) + w  
            rightmost_right_edge = px + r + w
            
            # TOPMOST possible top edge: (P_y - h/2) at highest point (P_y + r)
            topmost_top_edge = py - r - h//2
            
            # BOTTOMMOST possible bottom edge: (P_y + h/2) at lowest point (P_y - r)
            bottommost_bottom_edge = py + r + h//2
        elif refside == "right":
            # LEFTMOST possible left edge:
            leftmost_left_edge = px - r - w
            
            # RIGHTMOST possible right edge:
            rightmost_right_edge = px + r
            
            # TOPMOST possible top edge: (P_y - h/2) at highest point (P_y + r)
            topmost_top_edge = py - r - h//2
            
            # BOTTOMMOST possible bottom edge: (P_y + h/2) at lowest point (P_y - r)
            bottommost_bottom_edge = py + r + h//2

        # Smallest enclosing bounding box
        min_x = leftmost_left_edge
        min_y = topmost_top_edge
        max_x = rightmost_right_edge
        max_y = bottommost_bottom_edge
        
        # Clip to image bounds if provided
        if img_shape:
            img_h, img_w = img_shape[:2]
            min_x = max(0, min_x)
            min_x = min(img_w-1, min_x)
            min_y = max(0, min_y)
            min_y = min(img_h-1, min_y)

            max_x = min(img_w-1, max_x)
            max_x = max(0, max_x)
            max_y = min(img_h-1, max_y)
            max_y = max(0, max_y)
        
        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def clear_confident_region_and_bleed(self, binary_img, safe_bbx, max_bbx):
        """
        Remove foreground in safe_bbx AND any connected foreground reaching max_bbx
        
        Args:
            binary_img: Binary image (foreground=255, background=0)
            safe_bbx: [x1, y1, x2, y2] - safe region (ALWAYS remove foreground here)
            max_bbx: [x1, y1, x2, y2] - maximum extent to check connections
        
        Returns:
            cleaned_img: Binary image with unwanted foreground removed
        """
        cleaned_img = binary_img.copy()
        safe_x1, safe_y1, safe_x2, safe_y2 = safe_bbx
        max_x1, max_y1, max_x2, max_y2 = max_bbx

        # STEP 1: Clear safe region
        cleaned_img[safe_y1:safe_y2, safe_x1:safe_x2] = 0

        # STEP 2: Create TEMPORARY MASK of max_bbx region ONLY
        h, w = binary_img.shape
        mask = np.ones((h+2, w+2), np.uint8)  # +2 for floodFill border,  # Wherever non zero, floodfill is BLOCKED
        mask[max_y1+1:max_y2+1, max_x1+1:max_x2+1] = 0 # floodfill is only allowed in zero mask pixels, +1 in mask for perfect alignment with the image
        
        # STEP 3: Generate boundary pixels (safe_bbx edges)
        boundary_pixels = []
        
        for x in range(safe_x1, safe_x2):
            boundary_pixels.append((safe_y1, x)) # Top edge
            boundary_pixels.append((safe_y2-1, x)) # Bottom edge 
        
        
        for y in range(safe_y1+1, safe_y2-1):
            boundary_pixels.append((y, safe_x1)) # Left edge (skip corners - already added)
            boundary_pixels.append((y, safe_x2-1))  # Right edge (skip corners - already added)
        
        # Check 8-connectivity from boundary → max_bbx
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # STEP 4: Flood fill CONNECTED foreground WITHIN max_bbx ONLY
        for y, x in boundary_pixels:
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                
                # Valid neighbor in max_bbx with foreground?
                if (0 <= ny < h and 0 <= nx < w and 
                    max_x1 <= nx < max_x2 and max_y1 <= ny < max_y2 and 
                    cleaned_img[ny, nx] > 0):
                    
                    # Flood fill removes entire connected component within max_bbx due to mask
                    # cv.floodFill(	image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]	) -> 	retval, image, mask, rect
                    # cv2.floodFill(cleaned_img, mask, (nx, ny), 0, (2,2), (2,2))
                    cv2.floodFill(cleaned_img, mask, (nx, ny), 0)
                    break  # Found connection, move to next boundary pixel


        return cleaned_img


    stage_counters = {
        # Part 1/3 foreground counters
        1: 0,    
        # Part 1/3 hole counters  
        3: 0,    
        # Part 1/3 bottom FG counters
        5: 0,    
        # Part 2/4 foreground counters
        7: 0,    
        # Part 2/4 hole counters  
        '7_hole': 0,
        # Part 2/4 small hole counters (stage 8, 17)
        '8_hole': 0, 
    }
    part1_col = None
    part2_col = None

    def reset_stage_counters(self):
        """Reset ALL counters including hole variants"""
        self.stage_counters = {k: 0 for k in self.stage_counters.keys()}
        self.part1_col = None
        self.part2_col = None

    def increment_counter(self, stage, hole=False):
        """ Increments stage-specific counter and returns current value"""
        key = f"{stage}_hole" if hole else stage
        self.stage_counters[key] = self.stage_counters.get(key, 0) + 1 # Auto-create missing keys
        return self.stage_counters[key]

    # Helper functions for stage matching
    def stage_col_match(self, stage, col_idx):
        """Check if current column matches expected part column"""
        col_map = {
            1: self.part1_col, 2: self.part1_col, 3: self.part1_col, 4: self.part1_col, 5: self.part1_col,
            7: self.part2_col, 8: self.part2_col
        }
        expected_col = col_map.get(stage)
        return expected_col is not None and col_idx == expected_col

    def process_stage(self, mask, ry, rx, stage, col_idx, reached_btm):
        """Single source of truth for ALL stage transitions """
        """Master state machine - handles ALL 9 stages"""
        breakloop = False

        # Part 1/2 Detection (stages 0→6)
        if stage == 0:
            if self.is_foreground(mask[ry, rx]):
                stage = 1   # Found the foregrnd
                self.stage_counters[stage] = 0 
                self.part1_col = col_idx # part 1 should be in same column

        # if foregrnd found
        elif stage == 1 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]):
                if self.increment_counter(stage) > 4: 
                    stage = 2 # Assume detected non noise like face or neck part
            else: 
                self.reset_stage_counters()
                stage = 0 # It was noise

        # If non noise face or neck part    
        elif stage == 2 and self.stage_col_match(stage, col_idx):
            if not self.is_foreground(mask[ry, rx]): 
                stage = 3 # Found the Hole
                self.stage_counters[stage] = 0 
        
        # If Found the Hole
        elif stage == 3 and self.stage_col_match(stage, col_idx):
            if not self.is_foreground(mask[ry, rx]):
                if self.increment_counter(stage) > 4: 
                    stage = 4 # Assume detected non noise hole
            else:
                self.stage_counters[stage] = 0  # Reset when FG found too early

        # If non noise Hole
        elif stage == 4 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]): 
                stage = 5 # Found the Bottom FG
                self.stage_counters[stage] = 0 
        
        # If Found the Bottom FG
        elif stage == 5 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]):
                if self.increment_counter(stage) > 2: 
                    stage = 6   # Assume detected non noise  Bottom FG
                                # detected Part 1, so need to further traveses this column
                    breakloop = True
            else:
                self.stage_counters[stage] = 0  # reset flag, if fg found too early
        
        # Part 2/4 Detection (stages 6→9)
        elif stage == 6:
            if self.is_foreground(mask[ry, rx]):
                stage = 7 # Found the foregrnd in next column
                self.stage_counters[stage] = 0
                self.stage_counters[f"{stage}_hole"] = 0
                self.part2_col = col_idx # part 2 should be in same column
        
        # if  Part 2 foregrnd found
        elif stage == 7 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]):
                self.stage_counters[f"{stage}_hole"] = 0
                if self.increment_counter(stage) > 4: 
                    stage = 8 # Assume detected Part 2 non noise like face or neck part
                    self.stage_counters[stage] = 0
            else:
                self.stage_counters[stage] = 0 # It was noise
                if self.increment_counter(stage, hole=True) > 4: 
                    stage = 6 # Assume detected  hole
                    # Neck part without hole is not there in this column so try next column
                    breakloop = True

        # If Part 2 non noise face or neck part       
        elif stage == 8 and self.stage_col_match(stage, col_idx):
            if not self.is_foreground(mask[ry, rx]):
                # Found a Hole
                if self.increment_counter(stage, hole=True) > 1:
                    stage = 6 # Assume detected non noise hole
                    # Neck part without hole is not there in this column so try next column
                    breakloop = True
            else: 
                self.stage_counters[f"{stage}_hole"] = 0  # Reset hole counter
                if reached_btm:
                    # Found the Neck Part wo Hole end of Part2
                    stage = 9
                    breakloop = True
        
        return stage, breakloop

    def find_neck_pattern_start(self, neck_mask, neck_polygon, line_pts, col_height, col_direction_unit, top_end_pt, top_dir, btm_start_pt, btm_dir):
        """Find neck pattern start (stage 0→9) """
        top_left, top_right, bottom_right, bottom_left = neck_polygon
        top_width = abs(self.path_progress(top_left, top_right, top_dir))
        stage = 0
        self.reset_stage_counters()

        for col_idx, col_start in enumerate(line_pts):
            cur_top_width = abs(self.path_progress(col_start, top_end_pt, top_dir))
            cur_top_progress = 1-(cur_top_width / top_width)

            # the right side of neck line is expected to find before the mid point of neck width
            # If current col is more than 75% of width, too late so break
            if(cur_top_progress > 0.74):
                break

            # Find where THIS column intersects bottom line
            # print("In find_neck_pattern_start(), col_start=",col_start, " btm_start_pt=",btm_start_pt)
            intersect_pt = self.line_intersection(col_start, col_direction_unit, btm_start_pt, btm_dir)
            if intersect_pt is None:
                continue # Parallel - shouldn't happen

            # Project distances along column direction
            col_start_to_intersect = self.path_progress(col_start, intersect_pt, col_direction_unit)
            
            ## if in a column couldn't find part 2, then reset 
            if stage < 6:
                stage = 0

            botm_count = 0
            # Top to down traversal
            while True:
                row_pt = col_start + col_direction_unit * (botm_count + 1)
                rx, ry = int(row_pt[0]), int(row_pt[1])


                if rx < 0 or ry < 0 or botm_count > col_height: 
                    break
                
                if neck_polygon is not None and cv2.pointPolygonTest(neck_polygon, (rx, ry), False) >= 0:
                    # Row Progress to Btm line
                    cur_row_height = self.path_progress(col_start, row_pt, col_direction_unit)

                    ## Recommended Tolerances for Images
                    # 1e-6  = 0.000001 px  # Too precise - causes misses
                    # 0.1   = 0.1 px       # Sub-pixel
                    # 0.5   = 0.5 px       # Good default for images  
                    # 1.0   = 1 px         # Whole pixel tolerance
                    # Touched/Crossed if object reached or passed intersection point
                    img_tolerance = 0.9
                    reached_btm = False
                    if cur_row_height >= col_start_to_intersect - img_tolerance:
                        reached_btm = True

                    stage, breakloop = self.process_stage(neck_mask, ry, rx, stage, col_idx, reached_btm)


                    if stage == 9:  # Neck pattern found
                        return col_idx, col_start, row_pt
                    
                    if breakloop:
                        break
                
                botm_count += 1

        return None, None, None

    def remove_facial_part_from_region(self, neck_mask, region_mask, region_poly, pose_kps, pose_pixels, top_mid_pt, side='left'):
        """Extract and remove facial part from ONE side - REUSED for left/right"""
        global NOSE
        
        # top_left, top_right, bottom_right, bottom_left = polygon

        if side == 'left':
            # right side of left poly, top to bottom
            region_end_top = region_poly[1]
            region_end_btm = region_poly[2]
            offset_dir = -4
        else:
            # left side of right poly, top to bottom
            region_end_top = region_poly[0]
            region_end_btm = region_poly[3]
            offset_dir = 4

        apx_pt = None
        col_idx = 0
        stage = 0
        reached_btm = False
        self.reset_stage_counters()

        # region_poly, top to bottom , Find apex point
        line_pts = self.bresenham_line(region_end_top, region_end_btm)
        for i, row_pt in enumerate(line_pts):
            x, y = row_pt
            rx, ry = int(x), int(y)

            stage, breakloop = self.process_stage(region_mask, ry, rx, stage, col_idx, reached_btm)

            if stage == 3:  # Neck pattern found
                apx_pt = (rx, ry)
                break

            if breakloop:
                break

        # if apex point not found, then trying findin reflex pt of concave
        if apx_pt is None:
            # Check Nose kps confidence
            if pose_kps[NOSE, 2] < 0.5:
                ref_pt = top_mid_pt
            else:
                ref_pt = pose_pixels[NOSE].astype(int)

            # polygon_from_mask() will only consider 1 blob that has largest area
            poly_approx = self.polygon_from_mask(region_mask, epsilon_percent=0.03)
            apx_pt = self.find_concave_nearest_to_ref(poly_approx, ref_pt)
            
        # Find vectors and direction
        region_left_col_vector = np.array(region_poly[0]) - np.array(region_poly[3]) # top to btm
        region_rght_col_vector = np.array(region_poly[1]) - np.array(region_poly[2]) # top to btm
        region_btm_vector = np.array(region_poly[2]) - np.array(region_poly[3])  # left to right
        region_top_vector = np.array(region_poly[1]) - np.array(region_poly[0])  # left to right

        # Normalize direction vector
        region_left_col_ttb_dir_unit = region_left_col_vector / np.linalg.norm(region_left_col_vector)
        region_rght_col_ttb_dir_unit = region_rght_col_vector / np.linalg.norm(region_rght_col_vector)
        region_btm_ltr_dir_unit = region_btm_vector / np.linalg.norm(region_btm_vector)
        region_top_ltr_dir_unit = region_top_vector / np.linalg.norm(region_top_vector)


        # Find where THIS apex point  intersects left side of region_poly
        # print("In remove_facial_part_from_region(), apx_pt=",apx_pt, " region_poly[0]=",region_poly[0], " region_poly[1]=",region_poly[1])
        left_intersect_pt = self.line_intersection(apx_pt, -region_btm_ltr_dir_unit, region_poly[0], region_left_col_ttb_dir_unit)
    
        # Find where THIS apex point  intersects right side of region_poly
        rght_intersect_pt = self.line_intersection(apx_pt, region_btm_ltr_dir_unit, region_poly[1], region_rght_col_ttb_dir_unit)

        if left_intersect_pt is not None and rght_intersect_pt is not None:
            left_intersect_pt = tuple(map(int, np.round(left_intersect_pt)))
            rght_intersect_pt = tuple(map(int, np.round(rght_intersect_pt)))
            if side == 'left':
                region_start_btm_intersect_pt = left_intersect_pt
                region_end_btm_intersect_pt = rght_intersect_pt
            else:
                region_start_btm_intersect_pt = rght_intersect_pt
                region_end_btm_intersect_pt = left_intersect_pt

            # # offset left  by 4px if left part
            # # offset right by 4px if right part
            top_end_offset_pt = self.offset_along_line(region_end_top, region_top_ltr_dir_unit, offset_dir)
            top_end_offset_pt = (int(top_end_offset_pt[0]), int(top_end_offset_pt[1]))
            btm_end_offset_pt = self.offset_along_line(region_end_btm_intersect_pt, region_btm_ltr_dir_unit, offset_dir)
            btm_end_offset_pt = (int(btm_end_offset_pt[0]), int(btm_end_offset_pt[1]))

            remove_region_poly = region_poly.copy()
            # top_left, top_right, bottom_right, bottom_left = polygon
            if side == 'left':
                remove_region_poly[1] = top_end_offset_pt
                remove_region_poly[2] = btm_end_offset_pt
                remove_region_poly[3] = region_start_btm_intersect_pt
            else :
                remove_region_poly[0] = top_end_offset_pt
                remove_region_poly[2] = region_start_btm_intersect_pt
                remove_region_poly[3] = btm_end_offset_pt

            # Remove the remove_region_poly part from the neck_mask
            neck_mask = cv2.fillPoly(neck_mask, pts=[remove_region_poly], color=(0))
        return neck_mask, apx_pt

    def remove_remaining_facial_parts2(self, neck_mask, neck_polygon, pose_kps, pose_pixels):
        top_left, top_right, bottom_right, bottom_left = neck_polygon
        
        # Common setup (shared for both sides)
        img_h, img_w = neck_mask.shape

        lapex_pt = rapex_pt = None
        self.reset_stage_counters()

            
        tw = self.distance_btwn_points(top_left,top_right)
        bw = self.distance_btwn_points(bottom_left,bottom_right)
        if(tw > bw):
            leftmost = top_left
            rightmost = top_right
        else:
            leftmost = bottom_left
            rightmost = bottom_right
        
        lh = self.distance_btwn_points(bottom_left,top_left)
        rh = self.distance_btwn_points(bottom_right,top_right)
        if(lh > rh):
            bottommost = bottom_left
            topmost = top_left
        else:
            bottommost = bottom_right
            topmost = top_right

        

        # Direction vectors (shared)
        col_vector = np.array(bottommost) - np.array(topmost)
        h_mag = np.linalg.norm(col_vector)
        col_direction_unit = col_vector / h_mag
        col_height = int(h_mag)
        btm_vector = np.array(bottom_right) - np.array(bottom_left)
        btm_direction_unit = btm_vector / np.linalg.norm(btm_vector)
        top_vector = np.array(top_right) - np.array(top_left)
        top_direction_unit = top_vector / np.linalg.norm(top_vector)
        

        # Find LEFT neck part
        line_pts = self.bresenham_line(top_left, top_right)
        neck_left_col, neck_left_top_pt,  neck_left_btm_pt = self.find_neck_pattern_start(neck_mask, neck_polygon, line_pts, col_height, col_direction_unit, top_right, top_direction_unit, bottom_left, btm_direction_unit)
        # neck_left_col, neck_left_top_pt,  neck_left_btm_pt = None, None, None

        # Find RIGHT neck part
        line_pts = self.bresenham_line(top_right, top_left)
        neck_rght_col, neck_rght_top_pt,  neck_rght_btm_pt = self.find_neck_pattern_start(neck_mask, neck_polygon, line_pts, col_height, col_direction_unit, top_left, -top_direction_unit, bottom_right, -btm_direction_unit)
        # neck_rght_col, neck_rght_top_pt,  neck_rght_btm_pt = None, None, None
        
        # print("neck_left_col=",neck_left_col)
        # print("neck_rght_col=",neck_rght_col)
        if neck_left_col is not None and neck_rght_col is not None:
            # top_left, top_right, bottom_right, bottom_left = neck_polygon
            # Extract regions (symmetric)
            left_poly = neck_polygon.copy()
            left_poly[1] = neck_left_top_pt  
            left_poly[2] = neck_left_btm_pt
            extr_left_part = self.extract_polygon_region_from_binary(neck_mask, left_poly)
            
            right_poly = neck_polygon.copy()
            right_poly[0] = neck_rght_top_pt
            right_poly[3] = neck_rght_btm_pt
            extr_right_part = self.extract_polygon_region_from_binary(neck_mask, right_poly)
            
            ## Visualize
            # cv2.imshow("left",extr_left_part)
            # cv2.moveWindow("left", (extr_left_part.shape[1]*2)+100, 100)
            # cv2.imshow("right",extr_right_part)
            # cv2.moveWindow("right", (extr_right_part.shape[1]*4)+100, 100)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 

            top_mid_pt = ( int(top_left[0]+top_right[0]/2), int(top_left[1]+top_right[1]/2))
            # Remove facial parts - REUSE same logic for both sides
            neck_mask, lapex_pt  = self.remove_facial_part_from_region(neck_mask, extr_left_part, left_poly, pose_kps, pose_pixels, 
                                                        top_mid_pt, side='left')
            neck_mask, rapex_pt = self.remove_facial_part_from_region(neck_mask, extr_right_part, right_poly, pose_kps, pose_pixels, 
                                                        top_mid_pt, side='right')

        global R_EAR, L_EAR
        # Remove Ear
        if pose_kps[R_EAR, 2] > 0.3 and pose_kps[L_EAR, 2] > 0.3:
            r_ear = pose_pixels[R_EAR].astype(int)
            l_ear = pose_pixels[L_EAR].astype(int)

            # left ear to right ear distance
            letre = self.distance_btwn_points(l_ear,r_ear)

            min_ratio_w, min_ratio_h, min_ratio_o = [0.17, 0.35, 0.04] # min harcoded ratio of bbox_width, bbox_height, bbox_offset_radius
            max_ratio_w, max_ratio_h, max_ratio_o = [0.24, 0.42, 0.09] # max harcoded ratio of bbox_width, bbox_height, bbox_offset_radius

            r_min = int(np.round(letre * min_ratio_o)) # min offset radius 
            w_min = int(np.round(letre * min_ratio_w)) # bbox width
            h_min = int(np.round(letre * min_ratio_h)) # bbox height
            r_max = int(np.round(letre * max_ratio_o)) # max offset radius 
            w_max = int(np.round(letre * max_ratio_w)) # bbox width
            h_max = int(np.round(letre * max_ratio_h)) # bbox height

            # Right Ear
            # why l_ear in right?  bcoz right is Perspective of View from me, so right, 
            # but actually from the perspective of the image character its their left ear.

            # smallest_bounding_box_radial() gives a bounding box that covers all possible bounding box from reference_point 
            # and also considers that reference point can be offset by a radius.
            # if refside="left", then Reference_point is on leftside  and vertically on mid point of leftside
            # if refside="right", then Reference_point is on rightside  and vertically on mid point of rightside
            rbbox_small = self.smallest_bounding_box_radial(l_ear, r_min, w_min, h_min, neck_mask.shape, refside="left")
            rbbox_big = self.smallest_bounding_box_radial(l_ear, r_max, w_max, h_max, neck_mask.shape, refside="left")
            
            # remove the ear confidently in the rbbox_small region and remove all foreground connected from rbbox_small till rbbox_big (bleed region)
            neck_mask = self.clear_confident_region_and_bleed(neck_mask, rbbox_small, rbbox_big) # neck_mask wo right ear
            rbbx_sm_lx, rbbx_sm_ty, rbbx_sm_rx, rbbx_sm_by  = rbbox_small
            rbbx_bg_lx, rbbx_bg_ty, rbbx_bg_rx, rbbx_bg_by  = rbbox_big

            # Left ear
            lbbox_small = self.smallest_bounding_box_radial(r_ear, r_min, w_min, h_min, neck_mask.shape, refside="right")
            lbbox_big = self.smallest_bounding_box_radial(r_ear, r_max, w_max, h_max, neck_mask.shape, refside="right")
            
            # remove the ear confidently in the lbbox_small region and remove all foreground connected from lbbox_small till lbbox_big (bleed region)
            neck_mask = self.clear_confident_region_and_bleed(neck_mask, lbbox_small, lbbox_big) # neck_mask wo right ear

        return neck_mask, lapex_pt, rapex_pt

    def extract_neck_segment(self, kps, person_mask, face_mask = None, clothes_mask = None, neck_polygon_mode="torso_mid_ortho"):
        neck_segment = None

        # Handle both list-wrapped and direct JSON structures
        if isinstance(kps, list) and len(kps) > 0:
            kps = kps[0]
        elif not isinstance(kps, dict):
            raise ValueError(f"Error: Invalid openpose JSON structure")

        person = kps.get('people')
        if person is None:
            raise ValueError(f"people key not found in openpose JSON")

        if isinstance(person, list) and len(person) > 0:
            person = person[0]

        canvas_w = kps.get('canvas_width')
        if canvas_w is None:
            raise ValueError(f"canvas_width key not found in openpose JSON")

        canvas_h = kps.get('canvas_height')
        if canvas_h is None:
            raise ValueError(f"canvas_height key not found in openpose JSON")

        # Body Parts
        body = person.get('pose_keypoints_2d')
        if body is None:
            raise ValueError(f"pose_keypoints_2d key not found in openpose JSON")

        if person_mask is None:
            raise ValueError(f"Person mask must be provided")

        # convert mask with [0,1] to [0,255]
        person_mask = (person_mask == 1).astype(np.uint8) * 255
        # # Visualize
        # cv2.imshow("person_mask", person_mask)
        # person_mask_255 = (person_mask == 1.0).astype(np.uint8) * 255
        # cv2.imshow("person_mask_255", person_mask_255)
        # cv2.waitKey(0)

        pose_kps = np.array(body).reshape(-1, 3) #  Nx3: [x,y,confidence]
        pose_pixels = pose_kps[:, :2] * [canvas_w, canvas_h]

        # LEFT HAND Parts
        left_hand = person.get('hand_left_keypoints_2d')
        lhand_kps = lhand_pixels = None
        if left_hand is not None:
            lhand_kps = np.array(left_hand).reshape(-1, 3)  # Nx3: [x,y,confidence]
            lhand_pixels = lhand_kps[:, :2] * [canvas_w, canvas_h]  # Nx2: [[x1,y1], [x2,y2]...]

        # Right HAND Parts
        right_hand = person.get('hand_right_keypoints_2d')
        rhand_kps = rhand_pixels = None
        if right_hand is not None:
            rhand_kps = np.array(right_hand).reshape(-1, 3)  # Nx3: [x,y,confidence]
            rhand_pixels = rhand_kps[:, :2] * [canvas_w, canvas_h]  # Nx2: [[x1,y1], [x2,y2]...]


        # Face Parts
        face = person.get('face_keypoints_2d')
        face_kps = None
        face_pixels = np.zeros((70, 2))
        if face is not None:
            face_kps = np.array(face).reshape(-1, 3)  # Nx3: [x,y,confidence]
            face_pixels = face_kps[:, :2] * [canvas_w, canvas_h]  # Nx2: [[x1,y1], [x2,y2]...]
        
        h, w = person_mask.shape


        # Create neck polygon
        if neck_polygon_mode == "nose_neck_ortho":
            # Create neck polygon using orthogonal lines at nose/neck with (eye,ear,sholder)-based width
            polygon, nose_pt, neck_pt = self.create_neck_polygon(pose_pixels, pose_kps, h, w )
        elif neck_polygon_mode == "hip_side_width":
            # Create neck polygon with hip-oriented side lines and (eye,ear,sholder)-based width
            polygon, nose_pt, neck_pt = self.create_neck_polygon2(pose_pixels, pose_kps, h, w )
        elif neck_polygon_mode == "torso_mid_ortho":
            # Create neck polygon using torso midline and shoulder width
            polygon, nose_pt, neck_pt = self.create_neck_polygon3(pose_pixels, pose_kps, h, w )
        else:
            polygon = nose_pt = neck_pt = None

        if polygon is None:
            return None, None, None, None

        ## Neck Direction based on polygon
        mid_poly_top = ((polygon[0] + polygon[1]) // 2).astype(int)
        mid_poly_btm = ((polygon[2] + polygon[3]) // 2).astype(int)
        neck_mid_poly_vec = np.array(mid_poly_btm) - np.array(mid_poly_top)
        neck_mid_poly_length = np.linalg.norm(neck_mid_poly_vec)
        neck_poly_direction = neck_mid_poly_vec / neck_mid_poly_length

        # Create polygon mask
        neck_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(neck_mask, [polygon], 255)
        neck_segment = cv2.bitwise_and(person_mask, neck_mask)

        # Remove Face from Neck
        face_poly = None
        if face_mask is not None:
            # convert mask with [0,1] to [0,255]
            face_mask = (face_mask == 1).astype(np.uint8) * 255
            face_poly = self.polygon_from_mask(face_mask)
            print("    face_mask shape =", face_mask.shape, " dtype=",face_mask.dtype)
            print(" neck_segment shape =", neck_segment.shape, " dtype=",neck_segment.dtype)
            # Remove Face got from SAM2
            neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(face_mask))
            # # # Visualize
            # cv2.imshow("neck_segment_after_face", neck_segment)

        elif face_pixels is not None and len(face_pixels) > 0 and face_kps is not None:
            face_poly = self.create_face_polygon(face_kps, face_pixels, w, h)
            # Create mask from closed polygon
            face_poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_poly_mask, [face_poly], 255)    
            # Slight dilation for complete coverage
            kernel = np.ones((8, 8), np.uint8)
            face_poly_mask =  cv2.dilate(face_poly_mask, kernel, iterations=1)
            # Remove Face using openpose face points
            neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(face_poly_mask))


        # Remove Hand from Neck
        torso_poly = self.create_torso_polygon(pose_pixels,w,h)
        hand_mask = self.hand_segment_mask(person_mask, pose_kps, pose_pixels, lhand_kps, lhand_pixels, rhand_kps, rhand_pixels,  torso_poly, face_poly)

        # # Visualize
        # cv2.imshow("hand_mask", hand_mask)
        # cv2.waitKey(0)

        neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(hand_mask))

        # Remove any facial parts and ears that couldn't be removed earlier
        neck_segment, lapex_pt, rapex_pt = self.remove_remaining_facial_parts2(neck_segment, polygon, pose_kps, pose_pixels)

        # Remove Clothes from Neck
        if clothes_mask is not None:
            # convert mask with [0,1] to [0,255]
            clothes_mask = (clothes_mask == 1).astype(np.uint8) * 255

            print(" clothes_mask shape =", clothes_mask.shape, " dtype=",clothes_mask.dtype)
            print(" neck_segment shape =", neck_segment.shape, " dtype=",neck_segment.dtype)            
            # Remove Clothes got from SAM2
            neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(clothes_mask))

        # convert mask with [0,255] to [0,1] # best suited for torch mask
        neck_segment = (neck_segment > 0).astype(np.uint8)

        return neck_segment, lapex_pt, rapex_pt, neck_poly_direction


    def execute(self, kps, person_mask, face_mask = None, clothes_mask = None, neck_polygon_mode="torso_mid_ortho"):
        person_mask_cv = convert_to_opencv_mask(person_mask)
        if person_mask_cv is None:
            raise ValueError("person_mask couldn't be converted to opencv")
  
        if kps is None:
            raise ValueError("kps is required")

        # Convert the neck_mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if person_mask_cv.ndim == 3 and person_mask_cv.shape[2] in [1,3]:
            person_mask_cv = person_mask_cv[:, :, 0]

        face_mask_cv = None
        if face_mask is not None:
            face_mask_cv = convert_to_opencv_mask(face_mask)
            if face_mask_cv.ndim == 3 and face_mask_cv.shape[2] in [1,3]:
                face_mask_cv = face_mask_cv[:, :, 0]

        clothes_mask_cv = None
        if clothes_mask is not None:
            clothes_mask_cv = convert_to_opencv_mask(clothes_mask)
            if clothes_mask_cv.ndim == 3 and clothes_mask_cv.shape[2] in [1,3]:
                clothes_mask_cv = clothes_mask_cv[:, :, 0]

        # neck_mask, lapex_pt, rapex_pt = self.extract_neck_segment(kps, person_mask_cv, face_mask_cv, clothes_mask_cv, neck_polygon_mode)
        neck_mask, lapex_pt, rapex_pt, neck_dir = self.extract_neck_segment(kps, person_mask_cv, face_mask_cv, clothes_mask_cv, neck_polygon_mode)

        ## lapex_pt is None, it throws TypeError: 'NoneType' object is not subscriptable lapex_pt[0] 
        ## Handling this error
        if  lapex_pt is None:
            lapex_pt = [-1,-1]
        if  rapex_pt is None:
            rapex_pt = [-1,-1]
        print("in execute neck_dir =",neck_dir)
        neck_mask_torch = convert_opencv_mask_to_torch(neck_mask)
        l_nck_apx = json.dumps({"x": int(lapex_pt[0]), "y": int(lapex_pt[1])})
        r_nck_apx = json.dumps({"x": int(rapex_pt[0]), "y": int(rapex_pt[1])})
        nck_dirxn = json.dumps({"x": neck_dir[0], "y": neck_dir[1]})

        return (neck_mask_torch, l_nck_apx, r_nck_apx, nck_dirxn, )


class GetNeckSegment2:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",), # openpose keypoints in json format
                "person_mask": ("MASK",), # mask of person
            },
            "optional": {
                "face_mask":  ("MASK",),  # mask of person's face 
                "clothes_mask":  ("MASK",),  # mask of person's clothes 
                "neck_polygon_mode": (["nose_neck_ortho", "hip_side_width", "torso_mid_ortho"], { "default": "torso_mid_ortho" }),
                "trim_l_top":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),  # percentage to trim from top of neck mask of left side
                "trim_l_btm":  ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}), # percentage to trim from bottom of neck mask of left side
                "trim_r_top":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),  # percentage to trim from top of neck mask of right side
                "trim_r_btm":  ("FLOAT", {"default": 32.0, "min": 0.0, "max": 100.0, "step": 1.0}), # percentage to trim from bottom of neck mask of right side
                "max_ang_tol":  ("FLOAT", {"default": 30.0, "min": 0.0, "max": 360.0, "step": 1.0}), # remove segments which were deviates from neck direction
            }
        }


    RETURN_TYPES = ("MASK", "STRING",  "STRING", "IMAGE", "FLOAT")
    RETURN_NAMES = ("neck_segment", "nck_dirxn", "sidelines", "sideln_img", "confidence")
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


  

    def line_intersection(self, P1, v1, P2, v2):
        """
        Finds the intersection of two lines represented by points and direction vectors.
        
        P1, P2: Points on Line 1 and Line 2 respectively, each represented as a tuple (x, y).
        v1, v2: Direction vectors for Line 1 and Line 2 respectively, each represented as a tuple (vx, vy).
        
        Returns the intersection point (x, y) as a tuple, or None if no intersection exists.
        """
        # Convert points and direction vectors into numpy arrays for matrix operations
        P1 = np.array(P1)
        P2 = np.array(P2)
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        # Set up the matrix A and vector B
        A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
        B = P2 - P1


        # Check if the determinant of matrix A is non-zero (lines are not parallel)
        det_A = np.linalg.det(A)
        if det_A == 0:
            print("The lines are parallel and do not intersect.")
            return None
        
        # Solve for t and s using matrix inversion
        t_s = np.linalg.solve(A, B)
        t = t_s[0]  # t for Line 1, s for Line 2
        
        # Find the intersection point by substituting t into the parametric equation of Line 1
        intersection_point = P1 + t * v1
        
        return tuple(intersection_point)

    def distance_btwn_points(self, p1,p2):
        vec = np.array(p1) - np.array(p2)  # Points DOWN from dip to tip    
        distance = np.linalg.norm(vec)
        return distance
        
    # Distance ALONG the traversal direction
    def path_progress(self, p1,p2,direction_unit):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.dot(p2 - p1, direction_unit)

    # When traversing A→B→C, to find which side is L.
    # Applying Cross product sign gives the oriented angle from AB to AL. 
    # Positive = L is on left, negative = L is on right 
    def side_of_line(self, A, B, L):
        # Vector AB = B-A
        AB = B-A    
        # Vector AL = L-A  
        AL = L-A    
        # Cross product: AB × AL
        cross_z = np.cross(AB, AL)
        
        # cross_z >  0: "LEFT"
        # cross_z <  0: "RIGHT"
        # cross_z == 0: "ON_LINE"
        return cross_z

    # Bresenham's Line Algorithm: This will allow us to traverse pixel by pixel
    def bresenham_line(self, p1, p2):
        """
        Bresenham's line algorithm to return the list of pixels between p1 and p2.
        p1, p2: tuples (x, y)
        Returns a list of (x, y) coordinates for pixels on the line
        """
        x1, y1 = p1
        x2, y2 = p2
        
        pixels = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            pixels.append((x1, y1))
            
            if x1 == x2 and y1 == y2:
                break
            
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        return pixels

    def is_target_near_neighbours(self, contour, mask, target_values):
        h, w = mask.shape
        is_near = True

        points = contour.squeeze()
        for point in points: # (N,2)
            # print("Point:", point)
            x = int(point[0])
            y = int(point[1])

            # Skip boundary pixels
            if x <= 0 or y <= 0 or x >= w-1 or y >= h-1:
                continue

            # Get 8-connected neighbors
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:  # Skip center
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if mask[ny, nx] not in target_values:
                            is_near = False
                            break
                if is_near == False:
                    break
            if is_near == False:
                break
        return is_near

    def fill_contours_whose_neighbors_only_have_target(self, mask, contours, hierarchy, target_values=255, fill_color=255):
        h, w = mask.shape
        result_mask = mask.copy()

        # print("contours len=",len(contours))
        for i, contour in enumerate(contours):
            no_blemish = True
            h = hierarchy[i]
            #  hierarchy = [Next, Previous, First_Child, Parent]
            if h[3] == -1:  # Outer contour (no parent)

                # skip degenerates having only less than 2 point
                if(contour.shape[0]<2):
                    continue

                no_blemish = self.is_target_near_neighbours(contour, mask, target_values)
                # skip contour that has neighbours other than target
                if no_blemish == False:
                    continue
                
                # Inner hole
                child_idx = h[2]
                # if children exist
                while child_idx != -1:
                    hole_contour = contours[child_idx]

                    # only contour's that have more than 1 point
                    if(hole_contour.shape[0]>1):
                        no_blemish = self.is_target_near_neighbours(hole_contour, mask, target_values)

                        # If hole_contour has neighbours other than target, stop looking for children
                        if no_blemish == False:
                            break

                    child_idx = hierarchy[child_idx][0]  # Next sibling

                # Fill contour if it's neighbors has only target 
                if no_blemish == True:
                    # contourIdx	Parameter indicating a contour to draw. If it is negative, all the contours are drawn. 
                    cv2.drawContours(result_mask, [contour], -1, fill_color, thickness=-1)
    
        return result_mask

    def create_neck_polygon(self, pose_pixels, pose_kps, h, w):
        """Create neck polygon using orthogonal lines at nose/neck with (eye,ear,sholder)-based width"""

        global NOSE, NECK, R_SHOULDER, L_SHOULDER, R_EYE, L_EYE, R_EAR, L_EAR
        # Check confidence
        if pose_kps[NOSE, 2] < 0.5 or pose_kps[NECK, 2] < 0.5:
            return None, None, None
        
        eye_width = ear_width = sholdr_width = 0

        nose_pt = pose_pixels[NOSE].astype(int)
        neck_pt = pose_pixels[NECK].astype(int)
        
        # Eye points for width measurement (fallback if low confidence)
        if pose_kps[R_EYE, 2] > 0.3 and pose_kps[L_EYE, 2] > 0.3:
            r_eye = pose_pixels[R_EYE].astype(int)
            l_eye = pose_pixels[L_EYE].astype(int)
            eye_width = np.linalg.norm(r_eye - l_eye)

        # Ear points for width measurement (fallback if low confidence)
        if pose_kps[R_EAR, 2] > 0.3 and pose_kps[L_EAR, 2] > 0.3:
            r_ear = pose_pixels[R_EAR].astype(int)
            l_ear = pose_pixels[L_EAR].astype(int)
            ear_width = np.linalg.norm(r_ear - l_ear)

        # Shoulder points for width measurement
        if pose_kps[R_SHOULDER, 2] > 0.3 and pose_kps[L_SHOULDER, 2] > 0.3:
            r_sholdr = pose_pixels[R_SHOULDER].astype(int)
            l_sholdr = pose_pixels[L_SHOULDER].astype(int)
            sholdr_width = np.linalg.norm(r_sholdr - l_sholdr)

        neck_width = max(eye_width,ear_width,sholdr_width)

        # Neck vector (nose -> neck)
        # Example: nose at (300, 200)
        # Example: neck at (310, 250)
        # Result: [310-300, 250-200] = [10, 50]
        neck_vec = np.array(neck_pt) - np.array(nose_pt)

        # Purpose of neck vector in Neck Polygon
        # This vector defines the neck orientation. Next steps use it to:
        # Normalize → get unit length vector
        # Perpendicular → rotate 90° for width direction: [-y, x] = [-50, 10]
        # Offset → create left/right boundaries parallel to neck

        neck_length = np.linalg.norm(neck_vec)
        
        if neck_length == 0:
            return None, None, None
        
        # Unit perpendicular vector (rotate 90 degrees)
        perp_unit = np.array([-neck_vec[1], neck_vec[0]]) / neck_length
        
        # Create 4 polygon points
        half_width = neck_width / 2
        
        # Nose end points (orthogonal)
        nose_left = nose_pt + perp_unit * half_width
        nose_right = nose_pt - perp_unit * half_width
        
        # Neck end points (orthogonal)  
        neck_left = neck_pt + perp_unit * half_width
        neck_right = neck_pt - perp_unit * half_width

        # BOUNDS CHECK: Final polygon vertices
        polygon_points = [nose_left, nose_right, neck_right, neck_left]
        clamped_points = []
        for pt in polygon_points:
            clamped_pt = (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
            clamped_points.append(clamped_pt)
        
        polygon = np.array(clamped_points, dtype=np.int32) 
        
        return polygon, nose_pt, neck_pt

    def create_neck_polygon2(self, pose_pixels, pose_kps, h, w):
        """Create neck polygon with hip-oriented side lines and (eye,ear,sholder)-based width"""
        
        global NOSE, NECK, R_SHOULDER, L_SHOULDER, R_EYE, L_EYE, R_EAR, L_EAR, R_HIP, L_HIP
        
        # Check confidence
        if pose_kps[NOSE, 2] < 0.5 or pose_kps[NECK, 2] < 0.5:
            return None, None, None
        
        eye_width = ear_width = sholdr_width = hip_width = 0

        nose_pt = pose_pixels[NOSE].astype(int)
        neck_pt = pose_pixels[NECK].astype(int)
        
        # Eye points for width measurement 
        if pose_kps[R_EYE, 2] > 0.3 and pose_kps[L_EYE, 2] > 0.3:
            r_eye = pose_pixels[R_EYE].astype(int)
            l_eye = pose_pixels[L_EYE].astype(int)
            eye_width = np.linalg.norm(r_eye - l_eye)
        
        # Ear points for width measurement
        if pose_kps[R_EAR, 2] > 0.3 and pose_kps[L_EAR, 2] > 0.3:
            r_ear = pose_pixels[R_EAR].astype(int)
            l_ear = pose_pixels[L_EAR].astype(int)
            ear_width = np.linalg.norm(r_ear - l_ear)

        # Shoulder points for width measurement
        if pose_kps[R_SHOULDER, 2] > 0.3 and pose_kps[L_SHOULDER, 2] > 0.3:
            r_sholdr = pose_pixels[R_SHOULDER].astype(int)
            l_sholdr = pose_pixels[L_SHOULDER].astype(int)
            sholdr_width = np.linalg.norm(r_sholdr - l_sholdr)

        neck_width = max(eye_width,ear_width,sholdr_width)

        # Hip points for torso orientation reference
        if pose_kps[R_HIP, 2] > 0.3 and pose_kps[L_HIP, 2] > 0.3:
            r_hip = pose_pixels[R_HIP].astype(int)
            l_hip = pose_pixels[L_HIP].astype(int)
            hip_width = np.linalg.norm(r_hip - l_hip)
    
        # Hip orientation vector (neck_pt → mid_hip)
        mid_hip = (r_hip + l_hip) // 2 if hip_width > 0 else neck_pt
        torso_vec = np.array(mid_hip) - np.array(neck_pt)  # Points DOWN from neck to hips
        
        torso_length = np.linalg.norm(torso_vec)
        if torso_length == 0:
            # Fallback to neck vector if no hip data
            torso_vec = np.array(neck_pt) - np.array(nose_pt)
            torso_length = np.linalg.norm(torso_vec)
        
        # Side orientation = perpendicular to torso (hip-based)
        side_unit = np.array([-torso_vec[1], torso_vec[0]]) / torso_length  # Rotate 90° LEFT
        
        # Create 4 polygon points
        half_width = neck_width / 2
        
        # Top line: centered at nose_pt, perpendicular to torso
        nose_left = nose_pt + side_unit * half_width
        nose_right = nose_pt - side_unit * half_width
        
        # Bottom line: centered at neck_pt, perpendicular to torso  
        neck_left = neck_pt + side_unit * half_width
        neck_right = neck_pt - side_unit * half_width

        # BOUNDS CHECK: Final polygon vertices
        polygon_points = [nose_left, nose_right, neck_right, neck_left]
        clamped_points = []
        for pt in polygon_points:
            clamped_pt = (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
            clamped_points.append(clamped_pt)
        
        polygon = np.array(clamped_points, dtype=np.int32)

        return polygon, nose_pt, neck_pt

    def create_neck_polygon3(self, pose_pixels, pose_kps, h, w):
        """Create neck polygon using torso midline and shoulder width"""

        global NOSE, NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP

        # Check confidence for required points
        required_points = [NOSE, NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP]
        if any(pose_kps[i, 2] < 0.5 for i in required_points):
            return None, None, None
        
        nose_pt = pose_pixels[NOSE].astype(int)
        neck_pt = pose_pixels[NECK].astype(int)
        r_shoulder = pose_pixels[R_SHOULDER].astype(int)
        l_shoulder = pose_pixels[L_SHOULDER].astype(int)
        r_hip = pose_pixels[R_HIP].astype(int)
        l_hip = pose_pixels[L_HIP].astype(int)
        
        # STEP 1: Torso midline (neck_pt → mid_hip)
        mid_hip = ((r_hip + l_hip) // 2).astype(int)

        torso_midline_vec = np.array(mid_hip) - np.array(neck_pt)
        torso_length = np.linalg.norm(torso_midline_vec)
        torso_direction = torso_midline_vec / torso_length
        
        # STEP 2: Bottom line = shoulder width at neck_pt
        bottom_left = r_shoulder  # Right shoulder (adjust if needed based on view)
        bottom_right = l_shoulder  # Left shoulder
        
        # STEP 3: Side lines parallel to torso midline from shoulders
        # Extend side lines sufficiently long (2x torso length)
        side_length = np.linalg.norm(torso_midline_vec) * 2
        
        # Left side line: from bottom_left parallel to torso_direction
        left_side_end = (bottom_left + side_length * torso_direction).astype(int)
        
        # Right side line: from bottom_right parallel to torso_direction  
        right_side_end = (bottom_right + side_length * torso_direction).astype(int)
        
        # STEP 4: Top line orthogonal to torso midline at nose_pt
        # Perpendicular to torso_direction
        perp_torso = np.array([-torso_direction[1], torso_direction[0]])
        
        # Find intersection points: solve line-line intersections
        # print("In create_neck_polygon3(), nose_pt=",nose_pt, " bottom_left=",bottom_left, " bottom_right=" ,bottom_right)
        top_left = self.line_intersection(nose_pt, perp_torso, bottom_left, torso_direction)
        top_right = self.line_intersection(nose_pt, perp_torso, bottom_right, torso_direction)
        
        if top_left is None or top_right is None:
            return None, None, None

        # BOUNDS CHECK: Final polygon vertices
        polygon_points = [top_left, top_right, bottom_right, bottom_left]
        clamped_points = []
        for pt in polygon_points:
            clamped_pt = (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
            clamped_points.append(clamped_pt)
        
        polygon = np.array(clamped_points, dtype=np.int32)
        
        return polygon, nose_pt, neck_pt

    def create_face_polygon(self, face_kps, face_pixels, w, h):
        """ Helper to create a polygon by listing only face points """
        FACE_PARTS = {
            0: 'Jaw_R0', 8: 'Chin', 16: 'Jaw_L0',
            17: 'RBrow0', 19: 'RBrowPeak', 21: 'RBrowInner',
            22: 'LBrow0', 24: 'LBrowPeak', 26: 'LBrowInner',
            27: 'NoseT', 30: 'NoseTip', 35: 'NoseB',
            36: 'REyeOut', 39: 'REyeCen', 41: 'REyeIn',
            42: 'LEyeOut', 45: 'LEyeCen', 47: 'LEyeIn',
            48: 'ULipOutL', 54: 'ULipOutR', 59: 'LLipOutL', 53: 'LLipOutR',
            60: 'ULipInL', 64: 'ULipInR', 67: 'LLipInL', 63: 'LLipInR'
        }
        # Face Keypoints Mapping (0-69)
        # 0-16:  Jawline (right to left)
        # 17-21: Right eyebrow
        # 22-26: Left eyebrow  
        # 27-30: Nose bridge
        # 31-35: Nose tip/bottom
        # 36-41: Right eye (outer→inner)
        # 42-47: Left eye (outer→inner)
        # 48-59: Outer lips (upper→lower)
        # 60-67: Inner lips (upper→lower)

        # Ensure face_poly is a numpy array of shape (N, 2) of type np.int32

        # Face Jawline
        face_points = []
        for i in range(17):  # 0-16 jawline
            if i < len(face_pixels) and face_kps[i, 2] > 0.3:  # Confidence check
                pt = tuple(face_pixels[i].astype(int))
                face_points.append(pt)
        # Left eyebrow, Right eyebrow
        for i in range(26,16,-1):  # 26-17 eye brow in reversed order for polygon to close properly
            if i < len(face_pixels) and face_kps[i, 2] > 0.3:  # Confidence check
                pt = tuple(face_pixels[i].astype(int))
                face_points.append(pt)
        
        face_poly = np.array(face_points, dtype=np.int32)
        return face_poly

    def create_torso_polygon(self, pose_pixels, w, h, sh_buf=10, hp_buf=50):
        # Extend 10px buffer for shoulder and 50px buffer for hips

        global R_SHOULDER, L_SHOULDER, R_HIP, L_HIP

        # right shoulder, left shoulder,  left hip, right hip
        idx = [R_SHOULDER, L_SHOULDER, L_HIP, R_HIP]  # indices for the torso polygon (shoulders and hips)
        # Ensure torso_poly is a numpy array of shape (N, 2) of type np.int32
        buff_torso_pixels = []

        # Right Shoulder
        x, y = pose_pixels[R_SHOULDER]
        x = int(x)-sh_buf if int(x)-sh_buf > 0 else 1
        y = int(y)-sh_buf if int(y)-sh_buf > 0 else 1 
        buff_torso_pixels.append([x,y])

        # Left Shoulder
        x, y = pose_pixels[L_SHOULDER]
        x = int(x)+sh_buf if int(x)+sh_buf < w else w-1
        y = int(y)-sh_buf if int(y)-sh_buf > 0 else 1 
        buff_torso_pixels.append([x,y])

        # Left  Hip
        x, y = pose_pixels[L_HIP]
        x = int(x)+hp_buf if int(x)+hp_buf < w else w-1
        y = int(y)+hp_buf if int(y)+hp_buf < h else h-1
        buff_torso_pixels.append([x,y])

        # Right Hip
        x, y = pose_pixels[R_HIP]
        x = int(x)-hp_buf if int(x)-hp_buf > 0 else 1
        y = int(y)+hp_buf if int(y)+hp_buf < h else h-1
        buff_torso_pixels.append([x,y])
        
        
        torso_poly = np.array(buff_torso_pixels, dtype=np.int32)
        return torso_poly

    def polygon_from_mask(self, mask, epsilon_percent=0.01):
        polygon = None

        # Find contours 
        # RETR_EXTERNAL for outer boundary only
        # CHAIN_APPROX_SIMPLE reduces the contour's vertices to minimum required
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (main object)
        if contours:
            # each contour is of shape (N,1,2) , N: Number of points,  1: Single channel (like grayscale image), 2: x,y coordinates per point
            # key applies cv2.contourArea() on each contour in the list.
            largest_contour = max(contours, key=cv2.contourArea) # Returns the contour with largest area

            # Approximate to fewer vertices (epsilon=1% of perimeter)
            epsilon = epsilon_percent * cv2.arcLength(largest_contour, True)

            # approxPolyDP(	curved_input, epsilon, closed)
            # closed: If true, the approximated curve is closed (its first and last vertices are connected)
            # approxPolyDP() is to simplify contours by removing points while controlling deviation with epsilon. 
            # Epsilon defines the maximum allowed distance between any original contour point and the simplified polygon's edges.
            polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon = polygon.reshape(-1, 2) # reshape to [N,2]

        return polygon

    def hand_segment_mask(self, person_mask,
                            pose_kps, pose_pixels,
                            lhand_kps, lhand_pixels,
                            rhand_kps, rhand_pixels,
                            torso_poly, face_poly):

        global NOSE, NECK, R_SHOULDER, L_SHOULDER
    
        h, w = person_mask.shape

        # Find Hand mask
        hand_mask = np.zeros_like(person_mask)
        person_cpy = person_mask.copy() # where ever person is present, it's value = 255
        bgrd_colr = 0
        left_colr = 10
        rght_colr = 20
        lft2_colr = 30
        rht2_colr = 40
        fill_colr = 50
        face_colr = 60
        tors_colr = 70
        pers_colr = 255

        if torso_poly is not None and len(torso_poly) > 3:
            cv2.fillPoly(person_cpy, pts=[torso_poly], color=(tors_colr))
            
        
        if face_poly is not None and len(face_poly) > 3:
            cv2.fillPoly(person_cpy, pts=[face_poly], color=(face_colr))


        # Define paths through the skeleton (predefined paths)
        hand_paths = {
            # Right hand: shoulder→wrist→fingertip
            'right_arm': [(R_SHOULDER, 3), (3, 4) ],  # Right arm from shoulder to wrist 
            'right_pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],    # Wrist→MCP→Tip
            'right_ring': [(0, 13), (13, 14), (14, 15), (15, 16)],     # Wrist→MCP→Tip
            'right_middle': [(0, 9), (9, 10), (10, 11), (11, 12)],     # Wrist→MCP→Tip
            'right_index': [(0, 5), (5, 6), (6, 7), (7, 8)],           # Wrist→MCP→Tip
            'right_thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],           # Wrist→MCP→Tip
            
            # Left hand: shoulder→wrist→fingertip
            'left_arm': [(L_SHOULDER, 6), (6, 7) ],  # Left arm from shoulder to wrist 
            'left_pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],     # Wrist→MCP→Tip
            'left_ring': [(0, 13), (13, 14), (14, 15), (15, 16)],      # Wrist→MCP→Tip
            'left_middle': [(0, 9), (9, 10), (10, 11), (11, 12)],      # Wrist→MCP→Tip
            'left_index': [(0, 5), (5, 6), (6, 7), (7, 8)],            # Wrist→MCP→Tip
            'left_thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],            # Wrist→MCP→Tip
        }

        side_sign = 1 # for right side +1 and for left side -1
        eps = 0.01

        neck_pt = center = None
        # Neck point or Nose or center of the image is considered as Body's Center 
        # to know whether we are moving towards the body or away from it
        if pose_kps[NECK, 2] >= 0.3:
            tmp_pt = pose_pixels[NECK].astype(int)
            if(tmp_pt[0]>eps and tmp_pt[1]>eps):
                center = neck_pt = tmp_pt
        elif pose_kps[NOSE, 2] >= 0.3:
            tmp_pt = pose_pixels[NOSE].astype(int)
            if(tmp_pt[0]>eps and tmp_pt[1]>eps):
                center = nose_pt = tmp_pt
        else:
            center = np.array([w//2,h//2],dtype=np.int32)

        # Find the width of fingers
        finger_width = w
        arm_width = 0    
        dip_pt = pip_pt = None
        for path_name, path in hand_paths.items():
            distal_length = middle_length = proximal_length = 0
            # skip not fingers path
            if path_name in ["right_arm","left_arm"]:
                continue
            elif path_name in ['right_pinky', 'right_ring','right_middle', 'right_index',  'right_thumb']:
                pixels = rhand_pixels
            elif path_name in ['left_pinky', 'left_ring','left_middle', 'left_index',  'left_thumb']:
                pixels = lhand_pixels

            # 4th path of finger is DIP to Tip
            dip, tip = path[3][0], path[3][1]
            tmp1_pt = pixels[dip].astype(int)
            tmp2_pt = pixels[tip].astype(int)
            if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and tmp2_pt[0]>eps and tmp2_pt[1]>eps):
                dip_pt = tmp1_pt
                tip_pt = tmp2_pt
                distal_length = self.distance_btwn_points(tip_pt,dip_pt)
                if(distal_length>1):
                    finger_width = min(finger_width, distal_length)
            
            # 3rd path of finger is PIP to DIP
            pip = path[2][0]
            tmp1_pt = pixels[pip].astype(int)
            if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and dip_pt is not None):
                pip_pt = tmp1_pt
                middle_length = self.distance_btwn_points(dip_pt,pip_pt)

            # 2nd  path of finger is MCP to PIP
            mcp = path[1][0]
            tmp1_pt = pixels[mcp].astype(int)
            if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and pip_pt is not None):
                mcp_pt = tmp1_pt
                proximal_length = self.distance_btwn_points(pip_pt,mcp_pt)
            
            if distal_length > 0 and middle_length > 0 and proximal_length > 0:
                arm_width = max(arm_width, (distal_length + middle_length + proximal_length))


        # find the width of half of torso
        half_torso_width = w
        tmp1_pt = pose_pixels[R_SHOULDER].astype(int)
        tmp2_pt = pose_pixels[L_SHOULDER].astype(int)
        if(tmp1_pt[0]>eps and tmp1_pt[1]>eps and tmp2_pt[0]>eps and tmp2_pt[1]>eps):
            rshd_pt = tmp1_pt
            lshd_pt = tmp2_pt
            if neck_pt is not None:
                half_torso_width = min(half_torso_width,  self.distance_btwn_points(rshd_pt,neck_pt))
                half_torso_width = min(half_torso_width,  self.distance_btwn_points(lshd_pt,neck_pt))
            else:
                half_torso_width = self.distance_btwn_points(rshd_pt,lshd_pt)//2

        # hard set if could not find width of arm, finger or torso
        if arm_width < 1:
            arm_width = 90
        if finger_width >= w:
            finger_width = 18
        if half_torso_width >= w:
            half_torso_width = 120

        good_arm_width = good_arm_width_left = good_arm_width_rght = w
        

        # Primary Traverse each predefined path
        for path_name, path in hand_paths.items():
            if path_name in ['right_pinky', 'right_ring','right_middle', 'right_index',  'right_thumb']:
                pixels = rhand_pixels
                kps = rhand_kps
                side_sign = 1
                min_width = finger_width
                max_width = finger_width*2
            elif path_name in ['right_arm']:
                pixels = pose_pixels
                kps = pose_kps
                side_sign = 1
                min_width = arm_width
                max_width = arm_width
            elif path_name in ['left_pinky', 'left_ring','left_middle', 'left_index',  'left_thumb']:
                pixels = lhand_pixels
                kps = lhand_kps
                side_sign = -1
                min_width = finger_width
                max_width = finger_width*2
            elif path_name in ['left_arm']:
                pixels = pose_pixels
                kps = pose_kps
                side_sign = -1
                min_width = arm_width
                max_width = arm_width
            else:
                continue


            for i in range(len(path)):
                s, e = path[i][0], path[i][1]
                if kps[s,2] > 0.3 and kps[e,2] > 0.3:
                    p1 = pixels[s].astype(int)
                    p2 = pixels[e].astype(int)
                    x1 = p1[0]
                    y1 = p1[1]
                    x2 = p2[0]
                    y2 = p2[1]
                    

                    if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                        # Calculate direction vector between consecutive skeleton points
                        edge_vector = np.array(p1) - np.array(p2)
                        magnitude = np.linalg.norm(edge_vector)

                        # Normalize direction vector
                        direction_unit = edge_vector / magnitude

                        # Perpendicular direction (normalized)
                        perpendicular_unit = np.array([-direction_unit[1], direction_unit[0]])

                        # Traverse pixel by pixel along the line from p1 to p2
                        line_pts = self.bresenham_line(p1,p2)

                        # Find the side sign based on where the point is wrt neck pt 
                        side_sign = 1 if self.side_of_line(p1, p2, center) >= 0 else -1

                        # Determine finger length based on how far it is from neck
                        distance_from_center = self.distance_btwn_points(p1, center)
                        # if not close to center, then let finger width be maximum
                        if(distance_from_center > half_torso_width):
                            allowed_width = max_width
                        else:
                            allowed_width = min_width

                        # Traverse the edge
                        current_point_count = 0
                        min_left_cnt = w
                        min_rght_cnt = w
                        good_arm_width_cnt = 0
                        for current_point in line_pts:

                            left_traversal_brokn_by_out = False
                            rght_traversal_brokn_by_out = False

                            # LEFT traversal
                            left_count = 0
                            while True:
                                left_pt = current_point + side_sign * perpendicular_unit * (left_count+1)
                                x, y = left_pt
                                lx, ly = int(x), int(y)

                                if lx < 0 or ly < 0 or lx >= w or ly >= h:
                                    break
                                if person_mask[ly, lx] == 0:
                                    left_traversal_brokn_by_out = True
                                    break

                                # Check if inside face polygon (if collision, stop traversal)
                                # A 10px buffer
                                if face_poly is not None and cv2.pointPolygonTest(face_poly, (lx,ly), measureDist= True) + 10 >= 0:
                                    break
                                
                                # width gets too long
                                if left_count > allowed_width:
                                    break

                                left_count += 1

                                # Check if inside torso polygon , skip to next point
                                # pointPolygonTest returns whethers the point is inside (1), or outside (-1), or on the edge (0) of the polygon
                                if torso_poly is not None and cv2.pointPolygonTest(torso_poly, (lx,ly), False) >= 0:
                                    continue
                                # No need to overdraw when already traversed and marked
                                if person_cpy[ly, lx] == left_colr or person_cpy[ly, lx] == rght_colr:
                                    continue

                                cv2.circle(person_cpy, (lx, ly), 3, (left_colr), -1)  # Filled circle 
                                

                            # RIGHT traversal
                            right_count = 0
                            while True:
                                right_pt = current_point - side_sign * perpendicular_unit * (right_count+1)
                                x, y = right_pt
                                rx, ry = int(x), int(y)

                                if rx < 0 or ry < 0 or rx >= w or ry >= h:
                                    break
                                if person_mask[ry, rx] == 0:
                                    rght_traversal_brokn_by_out = True
                                    break
                                
                                # pointPolygonTest returns whethers the point is inside (1), or outside (-1), or on the edge (0) of the polygon
                                if torso_poly is not None and cv2.pointPolygonTest(torso_poly, (rx,ry), False) >= 0:
                                    break
                                # Check if inside face polygon (if collision, stop traversal)
                                # A 15px buffer
                                if face_poly is not None and  cv2.pointPolygonTest(face_poly, (rx,ry), measureDist= True) + 15 >= 0:
                                    break
                                # width gets too long
                                if right_count > allowed_width:
                                    break

                                right_count += 1

                                # No need to overdraw when already traversed and marked
                                if person_cpy[ry, rx] == left_colr or person_cpy[ry, rx] == rght_colr:
                                    continue

                                cv2.circle(person_cpy, (rx, ry), 3, (rght_colr), -1)  # Filled circle 
                            
                            # Find the arm width, if traversal to left and right of skeleton ends in outside of the person mask and 
                            # if we can get likewise continously for atleast 5 pixel points. Then we can consider that as a minimum good arm width
                            if( left_traversal_brokn_by_out and rght_traversal_brokn_by_out and
                                left_count > finger_width and right_count > finger_width and path_name in ['right_arm','left_arm']
                                ):
                                good_arm_width_cnt += 1
                                min_left_cnt = min(min_left_cnt, left_count)
                                min_rght_cnt = min(min_rght_cnt, right_count)
                            else:
                                good_arm_width_cnt = 0
                                min_left_cnt = w
                                min_rght_cnt = w

                            if good_arm_width_cnt > 5:
                                if (min_left_cnt + min_rght_cnt) <  good_arm_width:
                                    good_arm_width_left = min_left_cnt
                                    good_arm_width_rght = min_rght_cnt
                                    good_arm_width = good_arm_width_left + good_arm_width_rght



                        # Draw edges of skeleton
                        cv2.line(person_cpy, (x1, y1), (x2, y2), (left_colr), 3)


        # Secondary Traverse only arm, to fill arm_width
        if finger_width < good_arm_width < w :
            for path_name, path in hand_paths.items():
                if path_name in ["right_arm","left_arm"]:
                    pixels = pose_pixels
                    kps = pose_kps

                    for i in range(len(path)):
                        s, e = path[i][0], path[i][1]
                        if kps[s,2] > 0.3 and kps[e,2] > 0.3:
                            p1 = pixels[s].astype(int)
                            p2 = pixels[e].astype(int)
                            x1 = p1[0]
                            y1 = p1[1]
                            x2 = p2[0]
                            y2 = p2[1]                        

                            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                                # Calculate direction vector between consecutive skeleton points
                                edge_vector = np.array(p1) - np.array(p2)
                                magnitude = np.linalg.norm(edge_vector)

                                # Normalize direction vector
                                direction_unit = edge_vector / magnitude

                                # Perpendicular direction (normalized)
                                perpendicular_unit = np.array([-direction_unit[1], direction_unit[0]])

                                # Traverse pixel by pixel along the line from p1 to p2
                                line_pts = self.bresenham_line(p1,p2)

                                # Find the side sign based on where the point is wrt neck pt 
                                side_sign = 1 if self.side_of_line(p1, p2, center) >= 0 else -1

                                for current_point in line_pts:
                                    # LEFT traversal
                                    left_count = 0
                                    while True:
                                        left_pt = current_point + side_sign * perpendicular_unit * (left_count+1)
                                        x, y = left_pt
                                        lx, ly = int(x), int(y)

                                        if lx < 0 or ly < 0 or lx >= w or ly >= h:
                                            break

                                        # Check if inside face polygon (if collision, stop traversal)
                                        # A 15px buffer
                                        if face_poly is not None and cv2.pointPolygonTest(face_poly, (lx,ly), measureDist= True) + 10 >= 0:
                                            break
                                        
                                        # width gets too long
                                        if left_count > good_arm_width_left:
                                            break

                                        left_count += 1

                                        if person_mask[ly, lx] == 0:
                                            continue
                                        # No need to overdraw when already traversed and marked
                                        if person_cpy[ly, lx] == left_colr or person_cpy[ly, lx] == rght_colr:
                                            continue

                                        cv2.circle(person_cpy, (lx, ly), 2, (lft2_colr), -1)  # Filled circle 

                                    # RIGHT traversal
                                    right_count = 0
                                    while True:
                                        right_pt = current_point - side_sign * perpendicular_unit * (right_count+1)
                                        x, y = right_pt
                                        rx, ry = int(x), int(y)

                                        if rx < 0 or ry < 0 or rx >= w or ry >= h:
                                            break
                                        
                                        # Check if inside face polygon (if collision, stop traversal)
                                        # A 15px buffer
                                        if face_poly is not None and cv2.pointPolygonTest(face_poly, (rx,ry), measureDist= True) + 15 >= 0:
                                            break
                                        # width gets too long
                                        if right_count > good_arm_width_rght:
                                            break

                                        right_count += 1

                                        if person_mask[ry, rx] == 0:
                                            continue
                                        # No need to overdraw when already traversed and marked
                                        if person_cpy[ry, rx] == left_colr or person_cpy[ry, rx] == rght_colr:
                                            continue

                                        cv2.circle(person_cpy, (rx, ry), 2, (rht2_colr), -1)  # Filled circle 

        if  face_poly is not None or torso_poly is not None:

            # debug:
            print(f"person_cpy shape: {person_cpy.shape}, dtype: {person_cpy.dtype}")
            print(f"Unique values: {np.unique(person_cpy)}")  # Should be [0, 255]
            print(f"Non-zero pixels: {np.count_nonzero(person_cpy)}")
            # # Visualize
            # cv2.imshow("person_cpy", person_cpy)
            # cv2.waitKey(0)

            # 2nd pass Find contours that are still not filled
            _th, person_cpy2 = cv2.threshold(person_cpy, 127, 255, cv2.THRESH_BINARY)
            # # Outer Boundary only
            # contours, _ = cv2.findContours(person_cpy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # filled_mask =  fill_contours_with_target_neighbors(person_cpy,contours,target_values=[bgrd_colr,left_colr,rght_colr,pers_colr,lft2_colr,rht2_colr],fill_color=fill_colr)

            # debug:
            print(f"person_cpy2 shape: {person_cpy2.shape}, dtype: {person_cpy2.dtype}")
            print(f"Unique values: {np.unique(person_cpy2)}")  # Should be [0, 255]
            print(f"Non-zero pixels: {np.count_nonzero(person_cpy2)}")

            # or Outer Boundary and Inner Hole
            contours, hierarchy = cv2.findContours(person_cpy2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # debug
            print(f"Found {len(contours)} contours")
            print("hierarchy.shape=",hierarchy.shape)  # (1, N, 4)

            hierarchy = hierarchy.squeeze()
            target=[bgrd_colr,left_colr,rght_colr,pers_colr,lft2_colr,rht2_colr]
            person_cpy =  self.fill_contours_whose_neighbors_only_have_target(person_cpy,contours,hierarchy, target_values=target,fill_color=fill_colr)

        # Get only the grayscale that are in this range
        lower = np.array([left_colr], dtype=np.uint8)   # Min value
        upper = np.array([fill_colr], dtype=np.uint8)   # Max value
        hand_mask = cv2.inRange(person_cpy, lower, upper)

        # Remove non person parts from hand mask
        hand_mask = cv2.bitwise_and(hand_mask, person_mask)
        return hand_mask

    def extract_polygon_region_from_binary(self, binary_img, polygon):
        """
        Extract polygon region from binary image.
        binary_img: 255=foreground, 0=background (or vice versa)
        polygon: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        Returns: extracted polygon region as binary image
        """
        # Convert points to required format: list of arrays
        pts = np.array(polygon, np.int32)
        
        # Create mask same size as input image
        mask = np.zeros(binary_img.shape[:2], dtype=np.uint8)
        
        # Fill polygon with white (255)
        cv2.fillPoly(mask, [pts], 255)
        
        # Apply mask to extract polygon region
        extracted = cv2.bitwise_and(binary_img, binary_img, mask=mask)
        
        return extracted

    def is_foreground(self, pixel, threshold=0):
        """
        Returns True if pixel differs from black background (0,0,0).
        """
        # Handle numpy scalars (uint8, int32, etc.)
        if np.isscalar(pixel):
            channels = [int(pixel)]
        # Handle numpy arrays
        elif hasattr(pixel, '__array__') or isinstance(pixel, np.ndarray):
            channels = pixel.tolist()
        # Handle lists/tuples
        elif isinstance(pixel, (list, tuple)):
            channels = list(pixel)
        else:
            # Native Python int/float
            channels = [int(pixel)]
        
        # Foreground if max channel exceeds threshold
        return max(channels) > threshold

    def offset_along_line(self, point, unit_vector, distance):
        """Offset point by distance along unit vector direction"""
        return (
            point[0] + distance * unit_vector[0],
            point[1] + distance * unit_vector[1]
        )

    def sort_points_by_distance(self, points, ref):
        """Sort points nearest to ref first - FASTEST method"""
        return sorted(points, key=lambda p: (p[0]-ref[0])**2 + (p[1]-ref[1])**2)

    def concave_points_of_polygon(self, polygon, step=5):
        """
        polygon: Nx2 array of points (CCW)
        step: spacing for stability (larger = smoother)
        returns: list of concave points # list of indices of concave points
        """
        n = len(polygon)
        concave = []

        for i in range(n):
            # Get the previous, current, and next points (cyclically)
            p_prev = polygon[(i - step) % n]
            p_curr = polygon[i]
            p_next = polygon[(i + step) % n]

            # Calculate vectors
            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            # 2D cross product (determinant) to determine the direction of the turn
            cross = np.cross(v1, v2)

            if cross < 0:  # concave for CCW contour
                concave.append(p_curr)
                # concave.append(i)

        return concave

    def find_concave_nearest_to_ref(self, polygon, ref_pt):
        concave_pts = self.concave_points_of_polygon(polygon, step=1)
        concave_pts_nearest = self.sort_points_by_distance(concave_pts, ref_pt)

        if len(concave_pts_nearest) > 0:
            return concave_pts_nearest[0]
        else:
            return None

    def smallest_bounding_box_radial(self, P, r, w, h, img_shape=None, refside="left"):
        """
        Calculate SMALLEST bounding box containing ALL possible object positions
        given point P with radial uncertainty r.
        
        Args:
            P: Tuple (px, py) - reference point
            r: Radial uncertainty (pixels)
            w, h: Object width, height
            img_shape: Optional (height, width) for image bounds
        
        Returns: (min_x, min_y, max_x, max_y) - smallest enclosing box
        """
        px, py = P
        
        # Step 1: Find extreme left_midpoint positions (circle radius r around P)
        # All possible left_midpoints form a CIRCLE of radius r around P
        
        # Step 2: For each possible left_midpoint, object extends:
        # - Left: 0px (left edge = left_midpoint_x)
        # - Right: w pixels
        # - Top: -h/2 pixels  
        # - Bottom: +h/2 pixels
        
        # Step 3: Find EXTREME positions across entire circle:
        if refside == "left":
            # LEFTMOST possible left edge: (P_x - r)
            leftmost_left_edge = px - r
            
            # RIGHTMOST possible right edge: (P_x + r) + w  
            rightmost_right_edge = px + r + w
            
            # TOPMOST possible top edge: (P_y - h/2) at highest point (P_y + r)
            topmost_top_edge = py - r - h//2
            
            # BOTTOMMOST possible bottom edge: (P_y + h/2) at lowest point (P_y - r)
            bottommost_bottom_edge = py + r + h//2
        elif refside == "right":
            # LEFTMOST possible left edge:
            leftmost_left_edge = px - r - w
            
            # RIGHTMOST possible right edge:
            rightmost_right_edge = px + r
            
            # TOPMOST possible top edge: (P_y - h/2) at highest point (P_y + r)
            topmost_top_edge = py - r - h//2
            
            # BOTTOMMOST possible bottom edge: (P_y + h/2) at lowest point (P_y - r)
            bottommost_bottom_edge = py + r + h//2

        # Smallest enclosing bounding box
        min_x = leftmost_left_edge
        min_y = topmost_top_edge
        max_x = rightmost_right_edge
        max_y = bottommost_bottom_edge
        
        # Clip to image bounds if provided
        if img_shape:
            img_h, img_w = img_shape[:2]
            min_x = max(0, min_x)
            min_x = min(img_w-1, min_x)
            min_y = max(0, min_y)
            min_y = min(img_h-1, min_y)

            max_x = min(img_w-1, max_x)
            max_x = max(0, max_x)
            max_y = min(img_h-1, max_y)
            max_y = max(0, max_y)
        
        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def clear_confident_region_and_bleed(self, binary_img, safe_bbx, max_bbx):
        """
        Remove foreground in safe_bbx AND any connected foreground reaching max_bbx
        
        Args:
            binary_img: Binary image (foreground=255, background=0)
            safe_bbx: [x1, y1, x2, y2] - safe region (ALWAYS remove foreground here)
            max_bbx: [x1, y1, x2, y2] - maximum extent to check connections
        
        Returns:
            cleaned_img: Binary image with unwanted foreground removed
        """
        cleaned_img = binary_img.copy()
        safe_x1, safe_y1, safe_x2, safe_y2 = safe_bbx
        max_x1, max_y1, max_x2, max_y2 = max_bbx

        # STEP 1: Clear safe region
        cleaned_img[safe_y1:safe_y2, safe_x1:safe_x2] = 0

        # STEP 2: Create TEMPORARY MASK of max_bbx region ONLY
        h, w = binary_img.shape
        mask = np.ones((h+2, w+2), np.uint8)  # +2 for floodFill border,  # Wherever non zero, floodfill is BLOCKED
        mask[max_y1+1:max_y2+1, max_x1+1:max_x2+1] = 0 # floodfill is only allowed in zero mask pixels, +1 in mask for perfect alignment with the image
        
        # STEP 3: Generate boundary pixels (safe_bbx edges)
        boundary_pixels = []
        
        for x in range(safe_x1, safe_x2):
            boundary_pixels.append((safe_y1, x)) # Top edge
            boundary_pixels.append((safe_y2-1, x)) # Bottom edge 
        
        
        for y in range(safe_y1+1, safe_y2-1):
            boundary_pixels.append((y, safe_x1)) # Left edge (skip corners - already added)
            boundary_pixels.append((y, safe_x2-1))  # Right edge (skip corners - already added)
        
        # Check 8-connectivity from boundary → max_bbx
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # STEP 4: Flood fill CONNECTED foreground WITHIN max_bbx ONLY
        for y, x in boundary_pixels:
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                
                # Valid neighbor in max_bbx with foreground?
                if (0 <= ny < h and 0 <= nx < w and 
                    max_x1 <= nx < max_x2 and max_y1 <= ny < max_y2 and 
                    cleaned_img[ny, nx] > 0):
                    
                    # Flood fill removes entire connected component within max_bbx due to mask
                    # cv.floodFill(	image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]]	) -> 	retval, image, mask, rect
                    # cv2.floodFill(cleaned_img, mask, (nx, ny), 0, (2,2), (2,2))
                    cv2.floodFill(cleaned_img, mask, (nx, ny), 0)
                    break  # Found connection, move to next boundary pixel


        return cleaned_img


    stage_counters = {
        # Part 1/3 foreground counters
        1: 0,    
        # Part 1/3 hole counters  
        3: 0,    
        # Part 1/3 bottom FG counters
        5: 0,    
        # Part 2/4 foreground counters
        7: 0,    
        # Part 2/4 hole counters  
        '7_hole': 0,
        # Part 2/4 small hole counters (stage 8, 17)
        '8_hole': 0, 
    }
    part1_col = None
    part2_col = None

    def reset_stage_counters(self):
        """Reset ALL counters including hole variants"""
        self.stage_counters = {k: 0 for k in self.stage_counters.keys()}
        self.part1_col = None
        self.part2_col = None

    def increment_counter(self, stage, hole=False):
        """ Increments stage-specific counter and returns current value"""
        key = f"{stage}_hole" if hole else stage
        self.stage_counters[key] = self.stage_counters.get(key, 0) + 1 # Auto-create missing keys
        return self.stage_counters[key]

    # Helper functions for stage matching
    def stage_col_match(self, stage, col_idx):
        """Check if current column matches expected part column"""
        col_map = {
            1: self.part1_col, 2: self.part1_col, 3: self.part1_col, 4: self.part1_col, 5: self.part1_col,
            7: self.part2_col, 8: self.part2_col
        }
        expected_col = col_map.get(stage)
        return expected_col is not None and col_idx == expected_col

    def process_stage(self, mask, ry, rx, stage, col_idx, reached_btm):
        """Single source of truth for ALL stage transitions """
        """Master state machine - handles ALL 9 stages"""
        breakloop = False

        # Part 1/2 Detection (stages 0→6)
        if stage == 0:
            if self.is_foreground(mask[ry, rx]):
                stage = 1   # Found the foregrnd
                self.stage_counters[stage] = 0 
                self.part1_col = col_idx # part 1 should be in same column

        # if foregrnd found
        elif stage == 1 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]):
                if self.increment_counter(stage) > 4: 
                    stage = 2 # Assume detected non noise like face or neck part
            else: 
                self.reset_stage_counters()
                stage = 0 # It was noise

        # If non noise face or neck part    
        elif stage == 2 and self.stage_col_match(stage, col_idx):
            if not self.is_foreground(mask[ry, rx]): 
                stage = 3 # Found the Hole
                self.stage_counters[stage] = 0 
        
        # If Found the Hole
        elif stage == 3 and self.stage_col_match(stage, col_idx):
            if not self.is_foreground(mask[ry, rx]):
                if self.increment_counter(stage) > 4: 
                    stage = 4 # Assume detected non noise hole
            else:
                self.stage_counters[stage] = 0  # Reset when FG found too early

        # If non noise Hole
        elif stage == 4 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]): 
                stage = 5 # Found the Bottom FG
                self.stage_counters[stage] = 0 
        
        # If Found the Bottom FG
        elif stage == 5 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]):
                if self.increment_counter(stage) > 2: 
                    stage = 6   # Assume detected non noise  Bottom FG
                                # detected Part 1, so need to further traveses this column
                    breakloop = True
            else:
                self.stage_counters[stage] = 0  # reset flag, if fg found too early
        
        # Part 2/4 Detection (stages 6→9)
        elif stage == 6:
            if self.is_foreground(mask[ry, rx]):
                stage = 7 # Found the foregrnd in next column
                self.stage_counters[stage] = 0
                self.stage_counters[f"{stage}_hole"] = 0
                self.part2_col = col_idx # part 2 should be in same column
        
        # if  Part 2 foregrnd found
        elif stage == 7 and self.stage_col_match(stage, col_idx):
            if self.is_foreground(mask[ry, rx]):
                self.stage_counters[f"{stage}_hole"] = 0
                if self.increment_counter(stage) > 4: 
                    stage = 8 # Assume detected Part 2 non noise like face or neck part
                    self.stage_counters[stage] = 0
            else:
                self.stage_counters[stage] = 0 # It was noise
                if self.increment_counter(stage, hole=True) > 4: 
                    stage = 6 # Assume detected  hole
                    # Neck part without hole is not there in this column so try next column
                    breakloop = True

        # If Part 2 non noise face or neck part       
        elif stage == 8 and self.stage_col_match(stage, col_idx):
            if not self.is_foreground(mask[ry, rx]):
                # Found a Hole
                if self.increment_counter(stage, hole=True) > 1:
                    stage = 6 # Assume detected non noise hole
                    # Neck part without hole is not there in this column so try next column
                    breakloop = True
            else: 
                self.stage_counters[f"{stage}_hole"] = 0  # Reset hole counter
                if reached_btm:
                    # Found the Neck Part wo Hole end of Part2
                    stage = 9
                    breakloop = True
        
        return stage, breakloop

    def find_neck_pattern_start(self, neck_mask, neck_polygon, line_pts, col_height, col_direction_unit, top_end_pt, top_dir, btm_start_pt, btm_dir):
        """Find neck pattern start (stage 0→9) """
        top_left, top_right, bottom_right, bottom_left = neck_polygon
        top_width = abs(self.path_progress(top_left, top_right, top_dir))
        stage = 0
        self.reset_stage_counters()

        for col_idx, col_start in enumerate(line_pts):
            cur_top_width = abs(self.path_progress(col_start, top_end_pt, top_dir))
            cur_top_progress = 1-(cur_top_width / top_width)

            # the right side of neck line is expected to find before the mid point of neck width
            # If current col is more than 75% of width, too late so break
            if(cur_top_progress > 0.74):
                break

            # Find where THIS column intersects bottom line
            # print("In find_neck_pattern_start(), col_start=",col_start, " btm_start_pt=",btm_start_pt)
            intersect_pt = self.line_intersection(col_start, col_direction_unit, btm_start_pt, btm_dir)
            if intersect_pt is None:
                continue # Parallel - shouldn't happen

            # Project distances along column direction
            col_start_to_intersect = self.path_progress(col_start, intersect_pt, col_direction_unit)
            
            ## if in a column couldn't find part 2, then reset 
            if stage < 6:
                stage = 0

            botm_count = 0
            # Top to down traversal
            while True:
                row_pt = col_start + col_direction_unit * (botm_count + 1)
                rx, ry = int(row_pt[0]), int(row_pt[1])


                if rx < 0 or ry < 0 or botm_count > col_height: 
                    break
                
                if neck_polygon is not None and cv2.pointPolygonTest(neck_polygon, (rx, ry), False) >= 0:
                    # Row Progress to Btm line
                    cur_row_height = self.path_progress(col_start, row_pt, col_direction_unit)

                    ## Recommended Tolerances for Images
                    # 1e-6  = 0.000001 px  # Too precise - causes misses
                    # 0.1   = 0.1 px       # Sub-pixel
                    # 0.5   = 0.5 px       # Good default for images  
                    # 1.0   = 1 px         # Whole pixel tolerance
                    # Touched/Crossed if object reached or passed intersection point
                    img_tolerance = 0.9
                    reached_btm = False
                    if cur_row_height >= col_start_to_intersect - img_tolerance:
                        reached_btm = True

                    stage, breakloop = self.process_stage(neck_mask, ry, rx, stage, col_idx, reached_btm)


                    if stage == 9:  # Neck pattern found
                        return col_idx, col_start, row_pt
                    
                    if breakloop:
                        break
                
                botm_count += 1

        return None, None, None

    def remove_facial_part_from_region(self, neck_mask, region_mask, region_poly, pose_kps, pose_pixels, top_mid_pt, side='left'):
        """Extract and remove facial part from ONE side - REUSED for left/right"""
        global NOSE
        
        # top_left, top_right, bottom_right, bottom_left = polygon

        if side == 'left':
            # right side of left poly, top to bottom
            region_end_top = region_poly[1]
            region_end_btm = region_poly[2]
            offset_dir = -4
        else:
            # left side of right poly, top to bottom
            region_end_top = region_poly[0]
            region_end_btm = region_poly[3]
            offset_dir = 4

        apx_pt = None
        col_idx = 0
        stage = 0
        reached_btm = False
        self.reset_stage_counters()

        # region_poly, top to bottom , Find apex point
        line_pts = self.bresenham_line(region_end_top, region_end_btm)
        for i, row_pt in enumerate(line_pts):
            x, y = row_pt
            rx, ry = int(x), int(y)

            stage, breakloop = self.process_stage(region_mask, ry, rx, stage, col_idx, reached_btm)

            if stage == 3:  # Neck pattern found
                apx_pt = (rx, ry)
                break

            if breakloop:
                break

        # if apex point not found, then trying findin reflex pt of concave
        if apx_pt is None:
            # Check Nose kps confidence
            if pose_kps[NOSE, 2] < 0.5:
                ref_pt = top_mid_pt
            else:
                ref_pt = pose_pixels[NOSE].astype(int)

            # polygon_from_mask() will only consider 1 blob that has largest area
            poly_approx = self.polygon_from_mask(region_mask, epsilon_percent=0.03)
            apx_pt = self.find_concave_nearest_to_ref(poly_approx, ref_pt)

        print("apx_pt=",apx_pt)
            
        # Find vectors and direction
        region_left_col_vector = np.array(region_poly[0]) - np.array(region_poly[3]) # top to btm
        region_rght_col_vector = np.array(region_poly[1]) - np.array(region_poly[2]) # top to btm
        region_btm_vector = np.array(region_poly[2]) - np.array(region_poly[3])  # left to right
        region_top_vector = np.array(region_poly[1]) - np.array(region_poly[0])  # left to right

        # Normalize direction vector
        region_left_col_ttb_dir_unit = region_left_col_vector / np.linalg.norm(region_left_col_vector)
        region_rght_col_ttb_dir_unit = region_rght_col_vector / np.linalg.norm(region_rght_col_vector)
        region_btm_ltr_dir_unit = region_btm_vector / np.linalg.norm(region_btm_vector)
        region_top_ltr_dir_unit = region_top_vector / np.linalg.norm(region_top_vector)

        left_intersect_pt = rght_intersect_pt = None

        if apx_pt is not None:
            # Find where THIS apex point  intersects left side of region_poly
            # print("In remove_facial_part_from_region(), apx_pt=",apx_pt, " region_poly[0]=",region_poly[0], " region_poly[1]=",region_poly[1])
            left_intersect_pt = self.line_intersection(apx_pt, -region_btm_ltr_dir_unit, region_poly[0], region_left_col_ttb_dir_unit)
        
            # Find where THIS apex point  intersects right side of region_poly
            rght_intersect_pt = self.line_intersection(apx_pt, region_btm_ltr_dir_unit, region_poly[1], region_rght_col_ttb_dir_unit)

        if left_intersect_pt is not None and rght_intersect_pt is not None:
            left_intersect_pt = tuple(map(int, np.round(left_intersect_pt)))
            rght_intersect_pt = tuple(map(int, np.round(rght_intersect_pt)))
            if side == 'left':
                region_start_btm_intersect_pt = left_intersect_pt
                region_end_btm_intersect_pt = rght_intersect_pt
            else:
                region_start_btm_intersect_pt = rght_intersect_pt
                region_end_btm_intersect_pt = left_intersect_pt

            # # offset left  by 4px if left part
            # # offset right by 4px if right part
            top_end_offset_pt = self.offset_along_line(region_end_top, region_top_ltr_dir_unit, offset_dir)
            top_end_offset_pt = (int(top_end_offset_pt[0]), int(top_end_offset_pt[1]))
            btm_end_offset_pt = self.offset_along_line(region_end_btm_intersect_pt, region_btm_ltr_dir_unit, offset_dir)
            btm_end_offset_pt = (int(btm_end_offset_pt[0]), int(btm_end_offset_pt[1]))

            remove_region_poly = region_poly.copy()
            # top_left, top_right, bottom_right, bottom_left = polygon
            if side == 'left':
                remove_region_poly[1] = top_end_offset_pt
                remove_region_poly[2] = btm_end_offset_pt
                remove_region_poly[3] = region_start_btm_intersect_pt
            else :
                remove_region_poly[0] = top_end_offset_pt
                remove_region_poly[2] = region_start_btm_intersect_pt
                remove_region_poly[3] = btm_end_offset_pt

            # Remove the remove_region_poly part from the neck_mask
            neck_mask = cv2.fillPoly(neck_mask, pts=[remove_region_poly], color=(0))
        return neck_mask, apx_pt

    def remove_remaining_facial_parts2(self, neck_mask, neck_polygon, pose_kps, pose_pixels):
        top_left, top_right, bottom_right, bottom_left = neck_polygon
        
        # Common setup (shared for both sides)
        img_h, img_w = neck_mask.shape

        lapex_pt = rapex_pt = None
        self.reset_stage_counters()

            
        tw = self.distance_btwn_points(top_left,top_right)
        bw = self.distance_btwn_points(bottom_left,bottom_right)
        if(tw > bw):
            leftmost = top_left
            rightmost = top_right
        else:
            leftmost = bottom_left
            rightmost = bottom_right
        
        lh = self.distance_btwn_points(bottom_left,top_left)
        rh = self.distance_btwn_points(bottom_right,top_right)
        if(lh > rh):
            bottommost = bottom_left
            topmost = top_left
        else:
            bottommost = bottom_right
            topmost = top_right

        

        # Direction vectors (shared)
        col_vector = np.array(bottommost) - np.array(topmost)
        h_mag = np.linalg.norm(col_vector)
        col_direction_unit = col_vector / h_mag
        col_height = int(h_mag)
        btm_vector = np.array(bottom_right) - np.array(bottom_left)
        btm_direction_unit = btm_vector / np.linalg.norm(btm_vector)
        top_vector = np.array(top_right) - np.array(top_left)
        top_direction_unit = top_vector / np.linalg.norm(top_vector)
        

        # Find LEFT neck part
        line_pts = self.bresenham_line(top_left, top_right)
        neck_left_col, neck_left_top_pt,  neck_left_btm_pt = self.find_neck_pattern_start(neck_mask, neck_polygon, line_pts, col_height, col_direction_unit, top_right, top_direction_unit, bottom_left, btm_direction_unit)
        # neck_left_col, neck_left_top_pt,  neck_left_btm_pt = None, None, None

        # Find RIGHT neck part
        line_pts = self.bresenham_line(top_right, top_left)
        neck_rght_col, neck_rght_top_pt,  neck_rght_btm_pt = self.find_neck_pattern_start(neck_mask, neck_polygon, line_pts, col_height, col_direction_unit, top_left, -top_direction_unit, bottom_right, -btm_direction_unit)
        # neck_rght_col, neck_rght_top_pt,  neck_rght_btm_pt = None, None, None
        
        # print("neck_left_col=",neck_left_col)
        # print("neck_rght_col=",neck_rght_col)
        if neck_left_col is not None and neck_rght_col is not None:
            # top_left, top_right, bottom_right, bottom_left = neck_polygon
            # Extract regions (symmetric)
            left_poly = neck_polygon.copy()
            left_poly[1] = neck_left_top_pt  
            left_poly[2] = neck_left_btm_pt
            extr_left_part = self.extract_polygon_region_from_binary(neck_mask, left_poly)
            
            right_poly = neck_polygon.copy()
            right_poly[0] = neck_rght_top_pt
            right_poly[3] = neck_rght_btm_pt
            extr_right_part = self.extract_polygon_region_from_binary(neck_mask, right_poly)
            
            ## Visualize
            # cv2.imshow("left",extr_left_part)
            # cv2.moveWindow("left", (extr_left_part.shape[1]*2)+100, 100)
            # cv2.imshow("right",extr_right_part)
            # cv2.moveWindow("right", (extr_right_part.shape[1]*4)+100, 100)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 

            top_mid_pt = ( int(top_left[0]+top_right[0]/2), int(top_left[1]+top_right[1]/2))
            # Remove facial parts - REUSE same logic for both sides
            neck_mask, lapex_pt  = self.remove_facial_part_from_region(neck_mask, extr_left_part, left_poly, pose_kps, pose_pixels, 
                                                        top_mid_pt, side='left')
            neck_mask, rapex_pt = self.remove_facial_part_from_region(neck_mask, extr_right_part, right_poly, pose_kps, pose_pixels, 
                                                        top_mid_pt, side='right')

        global R_EAR, L_EAR
        # Remove Ear
        if pose_kps[R_EAR, 2] > 0.3 and pose_kps[L_EAR, 2] > 0.3:
            r_ear = pose_pixels[R_EAR].astype(int)
            l_ear = pose_pixels[L_EAR].astype(int)

            # left ear to right ear distance
            letre = self.distance_btwn_points(l_ear,r_ear)

            min_ratio_w, min_ratio_h, min_ratio_o = [0.17, 0.35, 0.04] # min harcoded ratio of bbox_width, bbox_height, bbox_offset_radius
            max_ratio_w, max_ratio_h, max_ratio_o = [0.24, 0.42, 0.09] # max harcoded ratio of bbox_width, bbox_height, bbox_offset_radius

            r_min = int(np.round(letre * min_ratio_o)) # min offset radius 
            w_min = int(np.round(letre * min_ratio_w)) # bbox width
            h_min = int(np.round(letre * min_ratio_h)) # bbox height
            r_max = int(np.round(letre * max_ratio_o)) # max offset radius 
            w_max = int(np.round(letre * max_ratio_w)) # bbox width
            h_max = int(np.round(letre * max_ratio_h)) # bbox height

            # Right Ear
            # why l_ear in right?  bcoz right is Perspective of View from me, so right, 
            # but actually from the perspective of the image character its their left ear.

            # smallest_bounding_box_radial() gives a bounding box that covers all possible bounding box from reference_point 
            # and also considers that reference point can be offset by a radius.
            # if refside="left", then Reference_point is on leftside  and vertically on mid point of leftside
            # if refside="right", then Reference_point is on rightside  and vertically on mid point of rightside
            rbbox_small = self.smallest_bounding_box_radial(l_ear, r_min, w_min, h_min, neck_mask.shape, refside="left")
            rbbox_big = self.smallest_bounding_box_radial(l_ear, r_max, w_max, h_max, neck_mask.shape, refside="left")
            
            # remove the ear confidently in the rbbox_small region and remove all foreground connected from rbbox_small till rbbox_big (bleed region)
            neck_mask = self.clear_confident_region_and_bleed(neck_mask, rbbox_small, rbbox_big) # neck_mask wo right ear
            rbbx_sm_lx, rbbx_sm_ty, rbbx_sm_rx, rbbx_sm_by  = rbbox_small
            rbbx_bg_lx, rbbx_bg_ty, rbbx_bg_rx, rbbx_bg_by  = rbbox_big

            # Left ear
            lbbox_small = self.smallest_bounding_box_radial(r_ear, r_min, w_min, h_min, neck_mask.shape, refside="right")
            lbbox_big = self.smallest_bounding_box_radial(r_ear, r_max, w_max, h_max, neck_mask.shape, refside="right")
            
            # remove the ear confidently in the lbbox_small region and remove all foreground connected from lbbox_small till lbbox_big (bleed region)
            neck_mask = self.clear_confident_region_and_bleed(neck_mask, lbbox_small, lbbox_big) # neck_mask wo right ear

        return neck_mask, lapex_pt, rapex_pt

    def remove_small_contours(self, mask, k_area=100):
        # Ensure binary mask [0,255]
        if mask.dtype != np.uint8 or mask.max() != 255:
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create blank output mask
        clean_mask = np.zeros_like(mask)
        
        # Keep only contours with area >= k_area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= k_area:
                cv2.fillPoly(clean_mask, [contour], 255)  # Fill contour with white    
        return clean_mask

    def normalize(self, v):
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n

    def reduce_collinear_points_06(self, nodes):
        '''
            Problem with this reset logic in reduce_collinear_points_05():
            curr_origin = reduced[-1]
            extreme_node = nodes[i+1]

            When a turn occurs at nodes[i], the new segment should logically start at nodes[i], not nodes[i+1].
            But the code skips that point.
            That means the algorithm sometimes anchors the next segment to the wrong origin.

            Failure example: Consider a contour corner:
            (0,0), (1,0), (2,0), (2,1), (2,2)

            Correct simplified result:
            (0,0), (2,0), (2,2)

            This can incorrectly initialize the vertical segment because it resets with:
            extreme_node = nodes[i+1]
            instead of including the actual corner node in the next segment evaluation.

            Solution:
                The corner node (nodes[i]) is treated as the end of the current segment and the potential origin of the next one. 
                If a point is an "extreme" (like your (40,10) case), it is committed before we move on to a new direction.
            
            What this solves:
                Vertex Preservation: By checking if nodes[i][0] != simplified[-1][0], 
                                    the actual corner (the pivot point) is saved even if the "extreme" of the previous line was somewhere else.
                Continuous Anchoring: The curr_origin is now reset to nodes[i], ensuring the next segment's collinearity 
                                    and distance are measured from the actual junction.
                End-Point Accuracy: It correctly identifies that a backtraced line (like your (40 to 20) case) ends at 20, maintaining the contour's intended path.

            Requirement	                  Status
            remove collinear points	        ✅
            handle vertical runs	        ✅
            handle horizontal runs  	    ✅
            handle diagonal runs    	    ✅
            fix base-vector bug	            ✅
            preserve corners	            ✅
            collapse long runs	            ✅
            maintain contour order	        ✅
            safe for most raster contours	✅

        '''
        if len(nodes) <= 2:
            return nodes

        def get_dist_sq(n1, n2):
            p1, p2 = n1[0], n2[0]
            return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

        reduced = [nodes[0]]
        curr_origin = nodes[0]
        extreme_node = nodes[0]

        for i in range(1, len(nodes)):
            p_curr = nodes[i][0]
            p_prev = nodes[i-1][0]
            
            # 1. Update the extreme for the current collinear stretch
            if get_dist_sq(curr_origin, nodes[i]) > get_dist_sq(curr_origin, extreme_node):
                extreme_node = nodes[i]

            # 2. Check if the direction changes at this point
            is_collinear = True
            if i < len(nodes) - 1:
                p_next = nodes[i+1][0]
                v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
                # Cross product check
                if (v1[0] * v2[1] - v1[1] * v2[0]) != 0:
                    is_collinear = False
            else:
                # End of list is a forced "turn" to commit the final points
                is_collinear = False

            if not is_collinear:
                # Direction changed at nodes[i].
                # First, commit the extreme point found in the previous stretch
                if extreme_node[0] != reduced[-1][0]:
                    reduced.append(extreme_node)
                
                # Now, check if the corner itself (nodes[i]) is different from the extreme
                # This handles cases where the 'peak' wasn't the corner
                if nodes[i][0] != reduced[-1][0]:
                    reduced.append(nodes[i])

                # Reset: The next segment starts exactly at the corner
                if i < len(nodes) - 1:
                    curr_origin = nodes[i]
                    extreme_node = nodes[i+1] # Start tracking next stretch

        return reduced

    def extract_extreme_side_points_03(self, side_pts, contour_with_mid, side, mask):
        """
        Extract extreme contour points for a side, preserving cyclic order
        and keeping full node info for segments, and filtering to foreground mask pixels.
        Clips the winning segment to the scan band
        Adds only the extreme portion and
        if needed interpolated points are included
        
        side_pts : list of nodes on this side [(pt, h, w, idx), ...]
        contour_with_mid : full contour list [(pt, h, w, idx), ...] used for connecting segments
        side : "left" or "right"
        mask : binary mask, only foreground pixels are valid
        
        Returns:
            extreme_points : list of nodes [(pt, h, w, idx), ...] in order along contour
                            pt are integer pixels inside mask foreground
        """
        sides_len = len(side_pts)
        if sides_len < 1:
            return []

        extreme_points = []
        # ------------------------------------------------------------
        # 1. Collect unique event heights from segments
        # ------------------------------------------------------------
        event_heights = set()
        for i,node in enumerate(side_pts):
            idx_in_contour = node[3]  # contour index
            p1 = node
            event_heights.add(p1[1])  # h

            # p2 = contour_with_mid[(idx_in_contour + 1) % len(contour_with_mid)]  # next in full contour
            # event_heights.add(p2[1])  # h # duplicate enteries are removed bcoz event_heights is a set data structure

            # Dont include the last point of the last segment
            if i != (sides_len-1) :
                p2 = contour_with_mid[(idx_in_contour + 1) % len(contour_with_mid)]  # next in full contour
                event_heights.add(p2[1])  # h # duplicate enteries are removed bcoz event_heights is a set data structure

        event_heights = sorted(event_heights)

        if len(event_heights) < 2:
            return []

        next_idx = 0  # incrementing integer idx
        h_mask, w_mask = mask.shape[:2]
        # print("mask shape=", h_mask, w_mask)

        # print("mask=",mask[224:229,208:212])
        # ------------------------------------------------------------
        # 2. Scan at midpoints of consecutive heights
        # ------------------------------------------------------------
        for i in range(len(event_heights) - 1):
            h_low  = event_heights[i]
            h_high = event_heights[i + 1]
            h_scan = 0.5 * (h_low + h_high)
            intersections = []

            # ------------------------------------------------------------
            # 3. Find intersecting segments
            # ------------------------------------------------------------
            for node in side_pts:
                idx_in_contour = node[3]
                p1 = node
                p2 = contour_with_mid[(idx_in_contour + 1) % len(contour_with_mid)]

                h1, w1 = p1[1], p1[2]
                h2, w2 = p2[1], p2[2]

                # Check if segment spans scan height
                if (h1 <= h_scan <= h2) or (h2 <= h_scan <= h1):
                    if abs(h2 - h1) < 1e-12:
                        continue  # skip horizontal in neck space

                    # Linear interpolation to find width at scan height
                    t = (h_scan - h1) / (h2 - h1)
                    w_interp = w1 + t * (w2 - w1)
                    intersections.append((p1, p2, w_interp))

                    

            if not intersections:
                continue


            # ------------------------------------------------------------
            # 4. Pick extreme segment for this side
            # ------------------------------------------------------------
            widths = np.array([x[2] for x in intersections])
            idx_ext = np.argmin(widths) if side == "left" else np.argmax(widths)
            # extreme_segm = intersections[idx_ext]
            p1, p2, w_intp = intersections[idx_ext]
            h1, w1 = p1[1], p1[2]
            h2, w2 = p2[1], p2[2]

            # ------------------------------------------------------------
            # 5. Clip segment to slab boundaries (interpolated points)
            # ------------------------------------------------------------
            def interpolate_at_height(h_target):
                if abs(h2 - h1) < 1e-12:
                    return None

                nonlocal next_idx
                t = (h_target - h1) / (h2 - h1)
                pt_interp = p1[0] + t * (p2[0] - p1[0])
                w_interp = w1 + t * (w2 - w1)

                # # Use p1's idx for the new interpolated node
                # t = (h_target - h1) / (h2 - h1)  # 0 ≤ t ≤ 1
                # idx = p1[3] + t
                # idx = p1[3]
                idx = next_idx
                next_idx += 1

                # keep full node structure (pt, h, w, idx)
                return (pt_interp, h_target, w_interp, idx)

            # 6. Interpolate low and high slab endpoints
            pt_low  = interpolate_at_height(h_low)
            pt_high = interpolate_at_height(h_high)  
            
            # Only add if both exist
            if pt_low is not None and pt_high is not None:
                # Convert to integer pixels
                x_low, y_low = int(round(pt_low[0][0])), int(round(pt_low[0][1]))
                x_high, y_high = int(round(pt_high[0][0])), int(round(pt_high[0][1]))

                # Clip to mask
                x_low = max(0, min(w_mask - 1, x_low))
                y_low = max(0, min(h_mask - 1, y_low))
                x_high = max(0, min(w_mask - 1, x_high))
                y_high = max(0, min(h_mask - 1, y_high))

                # Both must be foreground
                if mask[y_low, x_low] > 0 and mask[y_high, x_high] > 0:
                    extreme_points.append(((x_low, y_low), pt_low[1], pt_low[2], pt_low[3]))
                    extreme_points.append(((x_high, y_high), pt_high[1], pt_high[2], pt_high[3]))
                # print("y_low=",y_low, "  x_low=",x_low, " mask[low]=",mask[y_low, x_low], "y_high=",y_high, "  x_high=",x_high, " mask[high]=",mask[y_high, x_high])


        # print("len extreme_points =", len(extreme_points))
        if not extreme_points:
            return []

        # print("extreme_points  --------")
        # ------------------------------------------------------------
        # 7. Remove duplicates after rounding
        # ------------------------------------------------------------
        unique_pts = []
        seen = set()
        for node in extreme_points:
            key = (node[0][0], node[0][1])
            if key not in seen:
                unique_pts.append(node)
                seen.add(key)
        if not unique_pts:
            return []

        # ------------------------------------------------------------
        # 8. Sort top → bottom without breaking cyclic order
        # ------------------------------------------------------------
        h_vals = np.array([node[1] for node in unique_pts])
        start_idx = np.argmin(h_vals)
        unique_pts = unique_pts[start_idx:] + unique_pts[:start_idx]
        h_vals = np.array([node[1] for node in unique_pts])

        # Reverse if bottom → top
        if h_vals[-1] < h_vals[0]:
            unique_pts = unique_pts[::-1]

        # ------------------------------------------------------------
        # 9. Handle flat vertical span: sort by width
        # ------------------------------------------------------------
        if abs(h_vals[-1] - h_vals[0]) < 1e-8:
            w_vals = np.array([node[2] for node in unique_pts])
            order = np.argsort(np.abs(w_vals))
            unique_pts = [unique_pts[i] for i in order]

        # ## debug
        # print("before reduce_collinear_points_03()")
        # dbgpts = [n[0] for n in unique_pts] # [(pt, h, w, idx),
        # print(dbgpts) 

        # ------------------------------------------------------------
        # 10. Reduce collinear points (AFTER sorting)
        # ------------------------------------------------------------
        # unique_pts = reduce_collinear_points_01(unique_pts)
        # unique_pts = reduce_collinear_points_02(unique_pts)
        unique_pts = self.reduce_collinear_points_06(unique_pts)

        # ## debug
        # print("after reduce_collinear_points_03()")
        # dbgpts = [n[0] for n in unique_pts] # [(pt, h, w, idx),
        # print(dbgpts) 

        if not unique_pts:
            return []

        # ------------------------------------------------------------
        # 11. Reassign sequential idx  (no gaps)
        # ------------------------------------------------------------    
        final_pts = [(pt, h, w, idx) for idx, (pt, h, w, cidx) in enumerate(unique_pts)]
    

        # return unique_pts  # full nodes [(pt, h, w, idx), ...]
        return final_pts  # full nodes [(pt, h, w, idx), ...]

    def find_extreme_neck_sides_partA(self, mask, midline_top, midline_bottom):
        """
        Part A: Process each blob independently to find left and right extreme neck sides.
        Returns two lists of nodes [(pt, h, w, contour_idx), ...] per side, preserving contour order
        and another list of all contour points nodes [[(pt, h, w, idx),..]] with additional inserted mid points.
        that is for each side contains a list of for each blobs. In side that list contains list of extreme most points in that blob.
        Each extreme point node contains point, height in neck space from top to bottom, width from neck midline, order of contour index
        """

        # ------------------------------------------------------------
        # 1. Build neck coordinate system
        # ------------------------------------------------------------
        midline_top = np.array(midline_top, dtype=np.float32)
        midline_bottom = np.array(midline_bottom, dtype=np.float32)

        # global v,n,mid_center
        v = self.normalize(midline_bottom - midline_top)  # midline direction
        n = np.array([-v[1], v[0]])                  # perpendicular direction
        mid_center = (midline_top + midline_bottom) / 2.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return [], []


        ## Visual debug
        # global mask_colr
        mask_colr = cv2.cvtColor(mask.copy()[:800,:], cv2.COLOR_GRAY2RGB)
    
            ## Visual debug   
        ## Draw neck direction
        cv2.line(mask_colr, tuple(midline_top.astype(int)), tuple(midline_bottom.astype(int)),  (0,120,255), 1)
        # print("midline_top=",midline_top)
        # print("midline_top=", tuple(midline_top.astype(int)))
        # print("midline_bottom=", tuple(midline_bottom.astype(int)))

        eps_mid = 0.5
        dist_thresh  = 1.5
        # ------------------------------------------------------------
        # 2. Process each blob independently
        # ------------------------------------------------------------
        all_left_extreme = []
        all_right_extreme = []
        all_contour_wit_mid = []

        for blob_idx, contour in enumerate(contours):

            # print("blob_idx=",blob_idx)
            contour = contour.reshape(-1, 2).astype(np.float32)
            con_len = len(contour)
            if con_len < 2:
                continue

            # ------------------------------------------------------------
            # 2a. Project contour into neck coordinate space
            # ------------------------------------------------------------
            contour_nodes = []
            for i, p in enumerate(contour):
                vec = p - mid_center
                h = np.dot(vec, v)
                w = np.dot(vec, n)
                contour_nodes.append((p, h, w))

            # ---------------------------------------------------------------------------------------
            #  Norma Case: Insert a mid point between a segment when it crosses midline.

            #  Case 1: One point is away from midline and other is near midline (say w<0.) 
            #  but has not crossed, 
            #  in this case, dont need a new midpoint 
            
            #  Case 2: Both points are on midline, dont need a new midpoint.
        
            #  Case 3: Points are near midline and segment do cross 
            #  and the distance is > than 1.5 then insert a new midpoint

            #  Case 4: One point near or on midline, the other point is far and away in other side. 
            #   and there is a cossing, in this case, dont need a new midpoint
            # ---------------------------------------------------------------------------------------

            # ------------------------------------------------------------
            # 2b. Insert midline intersection points (preserve contour order)
            # ------------------------------------------------------------
            contour_with_mid = []
            cmid_idx = 0
            for i in range(con_len):
                cur = contour_nodes[i]
                nxt = contour_nodes[(i + 1) % con_len]

                # Append current node with index
                contour_with_mid.append((*cur, cmid_idx))  # Unpacking cur with *
                cmid_idx += 1

                ## Check for midline crossing
                if cur[2] * nxt[2] < 0:  # w1 * w2 < 0, then crossing occured
                    near1 = abs(cur[2]) < eps_mid
                    near2 = abs(nxt[2]) < eps_mid

                    # if i>=111 and i<=113:
                    #     print(i," p=",cur[0], " nx=",nxt[0], " p.w=",cur[2], " nx.w=",nxt[2], " dist=",distance_btwn_points(cur[0], nxt[0]), " nr1=",near1, " nr2=",near2)


                    # Case 4: one near, one far  → SKIP
                    if near1 != near2:
                        continue


                    if self.distance_btwn_points(cur[0], nxt[0]) >= dist_thresh:
                        ## Case 3: both near  → SKIP
                        if near1 and near2:
                            continue


                        ## Normal crossing: both far, insert midpoint
                        else:

                            ## Linear Interpolation:  t = w1 / (w1 - w2)
                            t = cur[2] / (cur[2] - nxt[2])

                            ## Interpolated mid point where intersection occurs
                            M = cur[0] + t * (nxt[0] - cur[0])
                            vecm = M - mid_center
                            hm = np.dot(vecm, v)
                            wm = 0.0

                            ## debug
                            # if 242 245 284 287 
                            # if 117 <= i <= 119 and blob_idx== 0:
                            #     print(i," p=",cur[0], " nx=",nxt[0], " p.w=",cur[2], " nx.w=",nxt[2], " dist=",distance_btwn_points(cur[0], nxt[0]), " M=",M)
                            # print(i," p=",cur[0], " nx=",nxt[0], " p.w=",cur[2], " nx.w=",nxt[2], " dist=",distance_btwn_points(cur[0], nxt[0]), " M=",M)

                            contour_with_mid.append((M, hm, wm, cmid_idx))
                            cmid_idx += 1
                



            # ------------------------------------------------------------
            # 2c. Split sidewise (exclude points on midline)
            # ------------------------------------------------------------
            left_pts = []
            right_pts = []
            for node in contour_with_mid:
                w = node[2]

                # Dont include points that are on midline
                if abs(w) < eps_mid:
                    continue

                if w < 0:
                    left_pts.append(node)
                elif w > 0:
                    right_pts.append(node)

            # print("len left_pts=", len(left_pts), " len right_pts=", len(right_pts))

            # ------------------------------------------------------------
            # 2d. Extract extreme points per side (full nodes kept)
            # ------------------------------------------------------------
            # left_extreme = [(pt, h, w, idx), ...]
            # left_extreme  = extract_extreme_side_points_01(left_pts, contour_with_mid, "left")
            # right_extreme = extract_extreme_side_points_01(right_pts, contour_with_mid, "right")
            # left_extreme  = extract_extreme_side_points_02(left_pts, contour_with_mid, "left")
            # right_extreme = extract_extreme_side_points_02(right_pts, contour_with_mid, "right")
            left_extreme  = self.extract_extreme_side_points_03(left_pts, contour_with_mid, "left", mask)
            right_extreme = self.extract_extreme_side_points_03(right_pts, contour_with_mid, "right", mask)
            # left_extreme = assign_final_pts_in_mask(left_extreme, mask)
            # right_extreme = assign_final_pts_in_mask(right_extreme, mask)


            ## Debug
            # for i, node in enumerate(contour_nodes):
            # for node in contour_with_mid:
            # for node in left_pts:
            for node in left_extreme:
                pt = node[0]
                i = node[3]
                ## Visual debug
                ## left_side
                mask_colr[int(pt[1]),int(pt[0])] = (i,blob_idx,255)
    
                # if abs(node[2]) < eps_mid :
                #     print(i, "  left p=",pt, "  w=",node[2])

            # for node in right_pts:
            for node in right_extreme:
                pt = node[0]
                # h = node[1]
                # w = node[2]
                i = node[3]
                # print("blob_idx=", blob_idx, " cntr_idx= ",i,"  pt=",pt, "  h=",h, "  w=",w)
                ## Visual debug
                ## right_side
                mask_colr[int(pt[1]),int(pt[0])] = (i,255,blob_idx)
                # print(i, "  right p=",pt,)
                # if abs(node[2]) < eps_mid :
                #     print(i, "  right p=",pt, "  w=",node[2])

            # ------------------------------------------------------------
            # 2e. Append to global blob-wise extremes
            # ------------------------------------------------------------
            all_left_extreme.append(left_extreme)
            all_right_extreme.append(right_extreme)
            # print("len right_extreme=",len(right_extreme))

        # cv2.imshow("neck_mask3",mask_colr)
        # cv2.moveWindow("neck_mask3", (300*4)+100, 100)

        return all_left_extreme, all_right_extreme

    def filter_parallel_segments_04_in_blob_partB(self, pts_in_blob, direction, angle_tolerance=30, min_segm_len=1.5, n_direction_points=2):
        """
        Filter segments roughly along 'direction'.
        Special rule: the first entry that enters keep_mask can ignore valid_length 
        if valid_angle=True.

        pts_in_blob : list of tuples (pt, h, w, idx)  
        Returns filtered list of tuples preserving idx.
        """
        if len(pts_in_blob) < n_direction_points:
            return pts_in_blob.copy()

        # Extract just the points for vector math
        pts = np.array([pt for pt, h, w, idx in pts_in_blob], dtype=np.float32)
        v = direction / np.linalg.norm(direction)
        cos_thresh = np.cos(np.deg2rad(angle_tolerance))
        step = n_direction_points - 1

        segments = pts[step:] - pts[:-step]
        seg_lengths = np.linalg.norm(segments, axis=1)
        valid_length = seg_lengths >= min_segm_len

        seg_dirs = np.zeros_like(segments)
        nonzero_mask = seg_lengths > 0
        seg_dirs[nonzero_mask] = segments[nonzero_mask] / seg_lengths[nonzero_mask][:, None]

        cos_angles = np.abs(seg_dirs @ v)
        valid_angle = cos_angles >= cos_thresh
        valid_segments = valid_length & valid_angle

        keep_mask = np.zeros(len(pts), dtype=bool)
        first_insert_done = False

        for i in range(len(valid_segments)):
            if valid_segments[i]:
                # Normal case: valid segment → mark both endpoints
                keep_mask[i:i + n_direction_points] = True
                if not first_insert_done:
                    first_insert_done = True
            else:
                # Special case for first insertion into keep_mask
                if not first_insert_done and valid_angle[i]:
                    # Ignore valid_length for first entry
                    keep_mask[i:i + n_direction_points] = True
                    first_insert_done = True
                # else: do nothing

        # # Always keep last point
        # keep_mask[-1] = True

        # Return filtered tuples with (pt, h, w, idx) 
        filtered = []
        for i in range(len(pts)):
            if keep_mask[i]:
                filtered.append(pts_in_blob[i])
        return filtered

    def extract_global_extreme_points_partB(self, all_extremes, side, mask, direction, angle_tolerance=30):
        """
        Keep only its globally most extreme contour pieces
        Maintain contour order
        Have no collinear noise
        Have clean indexing

        all_extremes : list of blobs, each blob is list of nodes [(pt, h, w, idx), ...]
        side : "left" or "right"
        mask : foreground mask

        Returns:
            extreme_pts_per_blob : list of blobs, each blob keeps only most extreme points in contour order
        """
        h_mask, w_mask = mask.shape[:2]

        extreme_pts_per_blob = []

        # 1. Build global event heights
        event_heights = set()
        for blob_idx, blob_pts in enumerate(all_extremes):
            for node in blob_pts:
                event_heights.add(node[1])

                # if (240 <= node[0][0] <=241  and  288 <= node[0][1]<= 289) or (160 <= node[0][0] <=163  and  293 <= node[0][1]<= 302):   
                #     print("n=", node)    
            # create new empty blobs for  new extreme_pts_per_blob
            extreme_pts_per_blob.append([])

        event_heights = sorted(event_heights)

        # print("len event_heights=", len(event_heights))
        
        if len(event_heights) < 2:
            return all_extremes  # nothing to do

        next_idx = 0
        # extreme_pts_per_blob = [[] for _ in all_extremes]

        # 2. Scan globally
        for i in range(len(event_heights)-1):
            h_low, h_high = event_heights[i], event_heights[i+1]
            h_scan = 0.5*(h_low+h_high)
            # print(f"h_scan={h_scan}, h_low={h_low}, h_high={h_high}")
            intersections = []

            # scan all blobs
            for blob_idx, blob_pts in enumerate(all_extremes):
                if len(blob_pts) < 2:
                    continue
                

                # 3. Find segments in this blob crossing the scan
                # for j in range(len(blob_pts)):
                for j in range(len(blob_pts)-1):
                    p1 = blob_pts[j]
                    # p2 = blob_pts[(j+1) % len(blob_pts)]
                    p2 = blob_pts[(j+1)]
                    h1, w1 = p1[1], p1[2]
                    h2, w2 = p2[1], p2[2]
                        

                    if (h1 <= h_scan <= h2) or (h2 <= h_scan <= h1):
                        if abs(h2-h1) < 1e-12:
                            continue
                        t = (h_scan - h1) / (h2 - h1)
                        w_interp = w1 + t*(w2-w1)
                        # pt_interp = p1[0] + t*(p2[0]-p1[0])
                        pt1 = np.array(p1[0], dtype=np.float32)
                        pt2 = np.array(p2[0], dtype=np.float32)
                        pt_interp = pt1 + t * (pt2 - pt1)
                        intersections.append((p1, p2, pt_interp, w_interp, blob_idx))



            ## after all blobs, if still empty, skip to next h_scan
            if not intersections:
                continue

            ## debug
            # for j in range(len(intersections)):
            #     p1, p2, pt_interp, w_interp, blob_idx = intersections[j]
            #     # if (240 <= p1[0][0] <=241  and  288 <= p1[0][1]<= 289) or (240 <= p2[0][0] <=241  and  288 <= p2[0][1]<= 289):
            #     # or (160 <= p1[0][0] <=163  and  293 <= p1[0][1]<= 302) or (160 <= p2[0][0] <=163  and  293 <= p2[0][1]<= 302):
            #     if i==3:
            #         print(f"i={i}, j={j} intersections.append(({p1}, {p2}, {pt_interp}, {w_interp}, {blob_idx}) and h_scan={h_scan}")

            # 4. Pick extreme segment
            widths = np.array([x[3] for x in intersections])
            idx_ext = np.argmin(widths) if side=="left" else np.argmax(widths)
            p1, p2, pt_low_high, w_val, blob_idx = intersections[idx_ext]


            # Interpolate slab endpoints
            def interp_node(p_start, p_end, h_target):
                t = (h_target - p_start[1]) / (p_end[1]-p_start[1])
                # pt_interp = p_start[0] + t*(p_end[0]-p_start[0])
                pt_start = np.array(p_start[0], dtype=np.float32)
                pt_end = np.array(p_end[0], dtype=np.float32)
                pt_interp = pt_start + t * (pt_end - pt_start)
                w_interp = p_start[2] + t*(p_end[2]-p_start[2])
                x, y = int(round(pt_interp[0])), int(round(pt_interp[1]))
                x = max(0, min(w_mask-1, x))
                y = max(0, min(h_mask-1, y))
                if mask[y, x] == 0:
                    return None
                nonlocal next_idx
                idx = next_idx
                next_idx += 1
                return ((x, y), h_target, w_interp, idx)

            node_low = interp_node(p1, p2, h_low)
            node_high = interp_node(p1, p2, h_high)

            if node_low is not None and node_high is not None:
                extreme_pts_per_blob[blob_idx].append(node_low)
                extreme_pts_per_blob[blob_idx].append(node_high)

        # 5. Remove duplicates per blob while keeping contour order
        for b_idx, blob in enumerate(extreme_pts_per_blob):
            unique_blob = []
            seen = set()
            for n in blob:
                key = (n[0][0], n[0][1])
                if key not in seen:
                    unique_blob.append(n)
                    seen.add(key)
            extreme_pts_per_blob[b_idx] = unique_blob

        for b_idx, blob in enumerate(extreme_pts_per_blob):

            if len(blob) < 2:
                continue

            # ------------------------------------------------------------
            # 6. Sort top → bottom without breaking cyclic order
            # ------------------------------------------------------------
            h_vals = np.array([node[1] for node in blob])
            start_idx = np.argmin(h_vals)
            blob = blob[start_idx:] + blob[:start_idx]

            h_vals = np.array([node[1] for node in blob])

            # Reverse if bottom → top
            if h_vals[-1] < h_vals[0]:
                blob = blob[::-1]
                h_vals = h_vals[::-1]

            # ------------------------------------------------------------
            # 7. Handle flat vertical span
            # ------------------------------------------------------------
            if abs(h_vals[-1] - h_vals[0]) < 1e-8:
                w_vals = np.array([node[2] for node in blob])
                order = np.argsort(np.abs(w_vals))
                blob = [blob[i] for i in order]


            # ------------------------------------------------------------
            # 8. Strong collinear reduction
            # ------------------------------------------------------------
            # blob = reduce_collinear_points_01(blob)
            # blob = reduce_collinear_points_02(blob)
            blob = self.reduce_collinear_points_06(blob)

        

            if len(blob) < 2:
                extreme_pts_per_blob[b_idx] = blob
                continue

            # 9. Filter local segment dirxn relative to the midline dirxn 
            # ------------------------------------------------------------------------------------
            blob = self.filter_parallel_segments_04_in_blob_partB(blob, direction, angle_tolerance=angle_tolerance, min_segm_len=1.5, n_direction_points=2)

            ## debug
            # if b_idx == 0:
            #     print("direction=",direction)
            #     print("after filter_parallel_segments_04_in_blob_partB()")
            #     dbgpts = [n[0] for n in blob]
            #     print(dbgpts) 

            # 10. Reassign sequential indices (no gaps)
            # ------------------------------------------------------------
            # blob = [(pt, h, w, idx) for idx, (pt, h, w, old_idx) in enumerate(blob)]

            extreme_pts_per_blob[b_idx] = blob

        return extreme_pts_per_blob

    def find_extreme_neck_sides_07(self, mask, midline_top, midline_bottom, angle_tolerance=30):

        # ------------------------------------------------------------
        # 1. Build neck-aligned coordinate system
        # ------------------------------------------------------------
        midline_top = np.array(midline_top, dtype=np.float32)
        midline_bottom = np.array(midline_bottom, dtype=np.float32)

        v = self.normalize(midline_bottom - midline_top)  # along midline
        n = np.array([-v[1], v[0]])                  # perpendicular
        mid_center = (midline_top + midline_bottom) / 2.0
        # ------------------------------------------------------------

        left_side_in_blobs, right_side_in_blobs = self.find_extreme_neck_sides_partA(mask, midline_top, midline_bottom)
        # print("left_extreme_global")
        left_extreme_global = self.extract_global_extreme_points_partB(left_side_in_blobs, "left", mask, v, angle_tolerance)
        # print("right_extreme_global")
        right_extreme_global = self.extract_global_extreme_points_partB(right_side_in_blobs, "right", mask, v, angle_tolerance)

        #  [(pt, h, w, idx), ...]

        ## Visual debug
        glbl_mask_colr = cv2.cvtColor(mask.copy()[:800,:], cv2.COLOR_GRAY2RGB)
        ## Draw neck direction
        cv2.line(glbl_mask_colr, tuple(midline_top.astype(int)), tuple(midline_bottom.astype(int)),  (0,120,255), 1)

        # print("len left_extreme_global=", len(left_extreme_global), " right=", len(right_extreme_global))
        for bidx, blob in enumerate(left_extreme_global):
            # print(" LEFT blob index =",bidx, " len=",len(blob))
            for ni, node in enumerate(blob):
                # print(ni , "  ", node)
                pt = node[0]
                i = node[3]
                glbl_mask_colr[int(pt[1]),int(pt[0])] = (i,bidx,255)

        # print("right_extreme_global = extract_global_extreme_points_partB()")
        # for bidx, blob in enumerate(right_side_in_blobs):
        for bidx, blob in enumerate(right_extreme_global):
            # print(" blob index =",bidx)
            for ni, node in enumerate(blob):
                # print(ni , "  ", node)
                pt = node[0]
                h = node[1]
                w = node[2]
                i = node[3]
                # print("blob_idx=", bidx, " cntr_idx= ",i,"  pt=",pt, "  h=",h, "  w=",w)
                glbl_mask_colr[int(pt[1]),int(pt[0])] = (i,255,bidx)


        # cv2.imshow("glbl_mask",glbl_mask_colr)
        # cv2.moveWindow("glbl_mask", (300*2)+100, 100)

        return left_extreme_global, right_extreme_global

    def project_points(self, points, origin, direction):
        """
        Projects points onto a direction vector.
        Returns signed scalar projection values.
        """
        return np.dot(points - origin, direction)

    def trim_top_btm_artifacts(self, extr_nodes, mid_top, mid_btm, top_percent, btm_percent):
        ## extr_nodes = [[(pt, h, w, idx),(pt, h, w, idx),.....],        # blob 1
        ##              [(pt, h, w, idx),.....],                        # blob 2
        ##              .....]

        mid_top = np.array(mid_top, dtype=np.float32)
        mid_btm = np.array(mid_btm, dtype=np.float32)
        mid_dir = self.normalize(mid_btm - mid_top)
        btm_pct = btm_percent / 100.0
        top_pct = top_percent / 100.0


        # ---- collect all contour points ----
        all_pts = np.array([node[0] for blob in extr_nodes for node in blob])

        ## heights is the distance along neck direction from any contour point to mid_top_line
        ## Project vector (contour point - mid_top) onto  mid_dir vector,
        ## this gives us only the  height component from any contour point to mid_top along the direction of mid_dir
        all_heights = self.project_points(all_pts, mid_top, mid_dir)

        min_h, max_h = all_heights.min(), all_heights.max()
        height_range =  (max_h - min_h)

        ## Compute absolute height thresholds
        btm_height  = min_h + btm_pct  * height_range
        top_height = min_h + top_pct * height_range


        trimmed_extr_nodes = []
        for blob in extr_nodes:
            if(len(blob)==0):
                trimmed_extr_nodes.append([])
                continue
            

            # extract contour points
            contours = np.array([node[0] for node in blob])  # shape (N,2)
            ## heights is the distance along neck direction from any contour point to mid_top_line
            ## Project vector (contour point - mid_top) onto  mid_dir vector,
            ## this gives us only the  height component from any contour point to mid_top along the direction of mid_dir
            heights = self.project_points(contours, mid_top, mid_dir)


            ## Apply height filtering 
            valid_mask = (heights >= top_height) & (heights <= btm_height)
            valid_contour = contours[(heights >= top_height) & (heights <= btm_height)]
            valid_blob = [node for node, m in zip(blob, valid_mask) if m]

            trimmed_extr_nodes.append(valid_blob)

            ## debug
            # print(f"blob len = {len(blob)},  trimm blob len ={len(valid_blob)}, min_h={min_h}, max_h={max_h}, btm_height={btm_height}, top_height={top_height} ")
            # if(len(blob) >= len(valid_blob)):
                # print("heights=",heights)
                # print("all_heights=",all_heights)
                # print("contours =",contours) 
                # print("trimmd  =",valid_contour) 
        
        return trimmed_extr_nodes

    def compute_blob_segment_lengths_with_half_influence_last_pt(self, extreme_node):
        """
        extreme_node  [ [(pt1, h, w, idx),[(pt2, h, w, idx), ...],   ## blob1
                        [(pt1, h, w, idx),[(pt2, h, w, idx), ...],   ## blob2
                        ......
                    ]
        For each blob, compute segment lengths using point coordinates.
        Store the computed value as 'wt' in the same node format (pt, h, w, idx, wt).
        Compute the segment lengths (Euclidean distance between consecutive points).
        Only for last endpoint, add the avg of the segment connected to it.
        Suppose if you have points A, B, C, D, your lengths are L_1 (AB), L_2 (BC), L_3 (CD)
            Point 1: w_1 = L_1
            Point 2: w_2 = L_2
            Point 3: w_3 = L_3
            Point 4: w_4 = L_3 / 2
        Returns blob_weights  [ np.array[wt,wt,.....],        ## blob1
                                np.array[wt,wt,.....],        ## blob2
                                .....
                            ]
        """

        blob_weights= []

        for blob in extreme_node:
            bloblen = len(blob)
            if bloblen < 2:
                blob_weights.append(np.array([bloblen/2],dtype=float))
                continue

            # print("weight blob=",blob)
            # Extract points and Convert to numpy arrays
            pts = np.array([node[0] for node in blob])

            # Calculate N-1 segment lengths
            segments = np.diff(pts, axis=0)
            lengths = np.linalg.norm(segments, axis=1)

            #  Append last length
            last_val = lengths[-1] / 2  # Get last length, divide by 2 
            blob_weights.append(np.append(lengths, last_val))

        return blob_weights

    def compute_blob_segment_lengths_with_half_influence_last_pt_and_index_gap_penalty(self, extreme_node):
        """
        extreme_node  [ [(pt1, h, w, idx),[(pt2, h, w, idx), ...],   ## blob1
                        [(pt1, h, w, idx),[(pt2, h, w, idx), ...],   ## blob2
                        ......
                    ]
        For each blob, compute segment lengths using point coordinates.
        Store the computed value as 'wt' in the same node format (pt, h, w, idx, wt).
        Compute the segment lengths (Euclidean distance between consecutive points).
        Only for last endpoint, add the avg of the segment connected to it.
        Suppose if you have points A, B, C, D, your lengths are L_1 (AB), L_2 (BC), L_3 (CD)
            Point 1: w_1 = L_1
            Point 2: w_2 = L_2
            Point 3: w_3 = L_3
            Point 4: w_4 = L_3 / 2

        Also modify the segment weight using an index continuity factor.
        So Final weight = segment_length / ((gap^0.15) * (e^(β*(gap - 1))))
        Returns blob_weights  [ np.array[wt,wt,.....],        ## blob1
                                np.array[wt,wt,.....],        ## blob2
                                .....
                            ]
        """

        blob_weights= []

        for blob in extreme_node:
            bloblen = len(blob)
            if bloblen < 2:
                blob_weights.append(np.array([bloblen/2],dtype=float))
                continue

            # print("weight blob=",blob)
            
            weights = []
            for i in range(bloblen-1):
                pt1, h1, w1, idx1 = blob[i]
                pt2, h2, w2, idx2 = blob[i+1]


                p1 = np.array(pt1)
                p2 = np.array(pt2)

                # segment length
                seg_len = np.linalg.norm(p2 - p1)

                # contour index gap
                idx_gap = abs(idx2 - idx1)

                if idx_gap == 0:
                    idx_gap = 1

                # penalized weight
                penalty = (idx_gap**0.15) * np.exp(0.08*(idx_gap-1))
                wt = seg_len / penalty

                weights.append(wt)

            # last point gets half influence
            weights.append(weights[-1] / 2)

            blob_weights.append(np.array(weights))

        return blob_weights

    def fit_global_line_pca_from_blobs(self, extreme_nodes, blob_weights):
        """
        Fit ONE global weighted PCA line using all points from all blobs.

        Fit 2D line using covariance eigen-decomposition with weighted points.
        Since PCA itself has sign ambiguity and eigenvectors are defined up to ±
        This func explicitly choose the sign based on the input order.

        Input:
            extreme_nodes = [
                                [(pt, h, w, idx), ...],   # blob1
                                [(pt, h, w, idx), ...],   # blob2
                                ...
                            ]

            blob_weights = [ np.array[wt,wt,.....],       ## blob1
                            np.array[wt,wt,.....],       ## blob2
                            ....
                        ]       

        Returns:
            center : np.array
            direction : np.array
            all_points: np.array
        """
        # Convert inputs to numpy arrays for all points in every blob 
        all_points = []
        all_weights = []

        # print("fit len blob=", len(extreme_nodes))
        for blob, wts in zip(extreme_nodes, blob_weights):
            # print("fit len nodes=", len(blob))
            for (pt, h, w, idx), wt in zip(blob, wts):
                all_points.append(pt)
                all_weights.append(wt)

        all_points = np.asarray(all_points, dtype=float)
        all_weights = np.asarray(all_weights, dtype=float)

        sum_of_all_weights = np.sum(all_weights)

        # print("all_points=\n",all_points)
        if len(all_points) < 2:
            return None, None, None

        # if len(all_points) < 2:
        #     return all_points[0], np.array([0,0]), all_points

        # print(f"len(all_points)={len(all_points)}, len(all_weights)={len(all_weights)}, ")
        # --------------------------------
        # Weighted center
        # --------------------------------
        if sum_of_all_weights == 0:
            center = np.mean(all_points, axis=0)    # Fall back to unweighted mean
            # print("center wo weights=",center, " shape=", center.shape)
        else:
            center = np.average(all_points, axis=0, weights=all_weights)
            # print("center wit weights=",center, " shape=", center.shape)

        # Subtract weighted center from all_points
        shifted = all_points - center

        # --------------------------------
        # Weighted covariance matrix
        # --------------------------------
        if sum_of_all_weights == 0:
            cov = np.cov(shifted.T)       # Unweighted covariance
        else:
            cov = np.cov(shifted.T, aweights=all_weights)

        # --------------------------------
        # PCA
        # --------------------------------
        # Eigen decomposition of the weighted covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # The eigenvector corresponding to the largest eigenvalue is the principal direction
        largest_idx = np.argmax(eigenvalues)
        direction = eigenvectors[:, largest_idx]
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # -----------------------------------------------------------
        # Enforce deterministic direction  based on ordering
        # using first and last point overall
        # -----------------------------------------------------------
        order_vector = all_points[-1] - all_points[0]

        # If first and last are (almost) identical
        if np.linalg.norm(order_vector) < 1e-12 and len(all_points) >= 3:
            order_vector = all_points[-2] - all_points[0]

        # Only enforce if we have meaningful direction info
        if np.linalg.norm(order_vector) > 1e-12:
            if np.dot(direction, order_vector) < 0:
                direction = -direction

        return center, direction, all_points

    def to_json_serializable(self, data):
        """Recursively convert ndarray to list in dicts, lists, AND tuples."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            result = {}
            for k, v in data.items():
                result[k] = self.to_json_serializable(v)
            return result
        elif isinstance(data, list):
            result = []
            for item in data:
                result.append(self.to_json_serializable(item))
            return result
        elif isinstance(data, tuple):
            result = []
            for item in data:
                result.append(self.to_json_serializable(item))
            return tuple(result)
        return data
    
    def extract_neck_segment(self, kps, person_mask, face_mask = None, clothes_mask = None, neck_polygon_mode="torso_mid_ortho", l_height_percent=(0,32), r_height_percent=(0,32) , angle_tolerance=30):
        neck_segment = None

        # Handle both list-wrapped and direct JSON structures
        if isinstance(kps, list) and len(kps) > 0:
            kps = kps[0]
        elif not isinstance(kps, dict):
            raise ValueError(f"Error: Invalid openpose JSON structure")

        person = kps.get('people')
        if person is None:
            raise ValueError(f"people key not found in openpose JSON")

        if isinstance(person, list) and len(person) > 0:
            person = person[0]

        canvas_w = kps.get('canvas_width')
        if canvas_w is None:
            raise ValueError(f"canvas_width key not found in openpose JSON")

        canvas_h = kps.get('canvas_height')
        if canvas_h is None:
            raise ValueError(f"canvas_height key not found in openpose JSON")

        # Body Parts
        body = person.get('pose_keypoints_2d')
        if body is None:
            raise ValueError(f"pose_keypoints_2d key not found in openpose JSON")

        if person_mask is None:
            raise ValueError(f"Person mask must be provided")

        # convert mask with [0,1] to [0,255]
        person_mask = (person_mask == 1).astype(np.uint8) * 255
        # # Visualize
        # cv2.imshow("person_mask", person_mask)
        # person_mask_255 = (person_mask == 1.0).astype(np.uint8) * 255
        # cv2.imshow("person_mask_255", person_mask_255)
        # cv2.waitKey(0)

        pose_kps = np.array(body).reshape(-1, 3) #  Nx3: [x,y,confidence]
        pose_pixels = pose_kps[:, :2] * [canvas_w, canvas_h]

        # LEFT HAND Parts
        left_hand = person.get('hand_left_keypoints_2d')
        lhand_kps = lhand_pixels = None
        if left_hand is not None:
            lhand_kps = np.array(left_hand).reshape(-1, 3)  # Nx3: [x,y,confidence]
            lhand_pixels = lhand_kps[:, :2] * [canvas_w, canvas_h]  # Nx2: [[x1,y1], [x2,y2]...]

        # Right HAND Parts
        right_hand = person.get('hand_right_keypoints_2d')
        rhand_kps = rhand_pixels = None
        if right_hand is not None:
            rhand_kps = np.array(right_hand).reshape(-1, 3)  # Nx3: [x,y,confidence]
            rhand_pixels = rhand_kps[:, :2] * [canvas_w, canvas_h]  # Nx2: [[x1,y1], [x2,y2]...]


        # Face Parts
        face = person.get('face_keypoints_2d')
        face_kps = None
        face_pixels = np.zeros((70, 2))
        if face is not None:
            face_kps = np.array(face).reshape(-1, 3)  # Nx3: [x,y,confidence]
            face_pixels = face_kps[:, :2] * [canvas_w, canvas_h]  # Nx2: [[x1,y1], [x2,y2]...]
        
        h, w = person_mask.shape


        # Create neck polygon
        if neck_polygon_mode == "nose_neck_ortho":
            # Create neck polygon using orthogonal lines at nose/neck with (eye,ear,sholder)-based width
            polygon, nose_pt, neck_pt = self.create_neck_polygon(pose_pixels, pose_kps, h, w )
        elif neck_polygon_mode == "hip_side_width":
            # Create neck polygon with hip-oriented side lines and (eye,ear,sholder)-based width
            polygon, nose_pt, neck_pt = self.create_neck_polygon2(pose_pixels, pose_kps, h, w )
        elif neck_polygon_mode == "torso_mid_ortho":
            # Create neck polygon using torso midline and shoulder width
            polygon, nose_pt, neck_pt = self.create_neck_polygon3(pose_pixels, pose_kps, h, w )
        else:
            polygon = nose_pt = neck_pt = None

        if polygon is None:
            return None, None, None, None

        ## Neck Direction based on polygon
        mid_poly_top = ((polygon[0] + polygon[1]) // 2).astype(int)
        mid_poly_btm = ((polygon[2] + polygon[3]) // 2).astype(int)
        neck_mid_poly_vec = np.array(mid_poly_btm) - np.array(mid_poly_top)
        neck_mid_poly_length = np.linalg.norm(neck_mid_poly_vec)
        neck_poly_direction = neck_mid_poly_vec / neck_mid_poly_length

        # Create polygon mask
        neck_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(neck_mask, [polygon], 255)
        neck_segment = cv2.bitwise_and(person_mask, neck_mask)

        # Remove Face from Neck
        face_poly = None
        if face_mask is not None:
            # convert mask with [0,1] to [0,255]
            face_mask = (face_mask == 1).astype(np.uint8) * 255
            face_poly = self.polygon_from_mask(face_mask)
            print("    face_mask shape =", face_mask.shape, " dtype=",face_mask.dtype)
            print(" neck_segment shape =", neck_segment.shape, " dtype=",neck_segment.dtype)
            # Remove Face got from SAM2
            neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(face_mask))
            # # # Visualize
            # cv2.imshow("neck_segment_after_face", neck_segment)

        elif face_pixels is not None and len(face_pixels) > 0 and face_kps is not None:
            face_poly = self.create_face_polygon(face_kps, face_pixels, w, h)
            # Create mask from closed polygon
            face_poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(face_poly_mask, [face_poly], 255)    
            # Slight dilation for complete coverage
            kernel = np.ones((8, 8), np.uint8)
            face_poly_mask =  cv2.dilate(face_poly_mask, kernel, iterations=1)
            # Remove Face using openpose face points
            neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(face_poly_mask))


        # Remove Hand from Neck
        torso_poly = self.create_torso_polygon(pose_pixels,w,h)
        hand_mask = self.hand_segment_mask(person_mask, pose_kps, pose_pixels, lhand_kps, lhand_pixels, rhand_kps, rhand_pixels,  torso_poly, face_poly)

        # # Visualize
        # cv2.imshow("hand_mask", hand_mask)
        # cv2.waitKey(0)

        neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(hand_mask))

        # Remove any facial parts and ears that couldn't be removed earlier
        neck_segment, lapex_pt, rapex_pt = self.remove_remaining_facial_parts2(neck_segment, polygon, pose_kps, pose_pixels)

        # Remove Clothes from Neck
        if clothes_mask is not None:
            # convert mask with [0,1] to [0,255]
            clothes_mask = (clothes_mask == 1).astype(np.uint8) * 255

            print(" clothes_mask shape =", clothes_mask.shape, " dtype=",clothes_mask.dtype)
            print(" neck_segment shape =", neck_segment.shape, " dtype=",neck_segment.dtype)            
            # Remove Clothes got from SAM2
            neck_segment = cv2.bitwise_and(neck_segment, cv2.bitwise_not(clothes_mask))


        ## ------------------------------------------------------------------------------------------------------------------------- ##
        ##        GET NECK SIDELINES
        ## ------------------------------------------------------------------------------------------------------------------------- ##
        cleand_neck_segm = self.remove_small_contours(neck_segment)
        left_extr_nodes, right_extr_nodes = self.find_extreme_neck_sides_07(cleand_neck_segm, mid_poly_top, mid_poly_btm, angle_tolerance)

        ## Remove top & bottom artifacts using height filtering
        left_top_percent, left_btm_percent = l_height_percent
        right_top_percent, right_btm_percent = r_height_percent
        left_extr_nodes  = self.trim_top_btm_artifacts(left_extr_nodes, mid_poly_top, mid_poly_btm, left_top_percent, left_btm_percent)
        right_extr_nodes = self.trim_top_btm_artifacts(right_extr_nodes, mid_poly_top, mid_poly_btm, right_top_percent, right_btm_percent)

        ##  Compute weights based on segment lengths
        # left_weights  = self.compute_blob_segment_lengths_with_half_influence_last_pt(left_extr_nodes)
        # right_weights = self.compute_blob_segment_lengths_with_half_influence_last_pt(right_extr_nodes)
        left_weights  = self.compute_blob_segment_lengths_with_half_influence_last_pt_and_index_gap_penalty(left_extr_nodes)
        right_weights = self.compute_blob_segment_lengths_with_half_influence_last_pt_and_index_gap_penalty(right_extr_nodes)

        ## Fit line to each side
        # print("b4 fit_global_line_pca_from_blobs(left_extr_nodes)")
        left_center, left_dir, left_points = self.fit_global_line_pca_from_blobs(left_extr_nodes, left_weights)

        # print("b4 fit_global_line_pca_from_blobs(right_extr_nodes)")
        right_center, right_dir, right_points = self.fit_global_line_pca_from_blobs(right_extr_nodes, right_weights)

        ## Compute Confidence Score
        confidence = 0.0
        if left_dir is not None  and right_dir is not None :

            ang_thresh = np.cos(np.deg2rad(angle_tolerance))

            ## the cos angle between the left_dir vector and neck_poly_direction vector
            left_cos_angles = left_dir @ neck_poly_direction
            # left_angle_deg = np.degrees(np.arccos(left_cos_angles))
            print(f"left_dir=({left_dir[0]:.2f},{left_dir[1]:.2f}), neck_dir=({neck_poly_direction[0]:.2f},{neck_poly_direction[1]:.2f}), left_center=({left_center[0]:.2f},{left_center[1]:.2f})")

            ## the cos angle between the right_dir vector and neck_poly_direction vector
            right_cos_angles = right_dir @ neck_poly_direction
            # right_angle_deg = np.degrees(np.arccos(right_cos_angles))
            print(f"right_dir=({right_dir[0]:.2f},{right_dir[1]:.2f}), neck_dir=({neck_poly_direction[0]:.2f},{neck_poly_direction[1]:.2f}), right_center=({right_center[0]:.2f},{right_center[1]:.2f})")

            l_conf = np.maximum(0, (left_cos_angles - ang_thresh) / (1 - ang_thresh))
            r_conf = np.maximum(0, (right_cos_angles - ang_thresh) / (1 - ang_thresh))
            confidence = np.minimum(l_conf, r_conf) ## strict confidence score, If both directions must agree

            ## reduce the confidence score by 5% whenever apex pt is not found
            if lapex_pt is None:
                confidence = confidence * 0.95
            if rapex_pt is None:
                confidence = confidence * 0.95


        ## Visual debug
        ## ----------------------------------------------------------------------------------------------------------------------------
        sidline_mask_colr = cv2.cvtColor(neck_segment.copy()[:800,:], cv2.COLOR_GRAY2RGB)
        # sidline_mask_colr = cv2.cvtColor(neck_segment.copy(), cv2.COLOR_GRAY2RGB)
        ## Draw neck direction
        cv2.line(sidline_mask_colr, tuple(mid_poly_top.astype(int)), tuple(mid_poly_btm.astype(int)),  (0,120,255), 1)

        # Left side line
        if left_center is not None:
            l_srt_point = left_center - (left_dir * 40)
            l_end_point = left_center + (left_dir * 40)
            # print("left_center=",left_center, " left_dir=",left_dir, " right_center=",right_center, " right_dir=",right_dir)
            # cv2.line(sidline_mask_colr, tuple(l_srt_point.astype(int)), tuple(l_end_point.astype(int)),  (0,255,200), 1)
            cv2.line(sidline_mask_colr, tuple(left_center.astype(int)), tuple(l_end_point.astype(int)),  (0,255,200), 2)

        # Right side line
        if right_center is not None:
            r_srt_point = right_center - (right_dir * 40)
            r_end_point = right_center + (right_dir * 40)
            # cv2.line(sidline_mask_colr, tuple(r_srt_point.astype(int)), tuple(r_end_point.astype(int)),  (0,120,255), 1)
            cv2.line(sidline_mask_colr, tuple(right_center.astype(int)), tuple(r_end_point.astype(int)),  (0,120,255), 2)

        # left points
        if left_points is not None:
            for i, pt in enumerate(left_points):
                sidline_mask_colr[int(pt[1]),int(pt[0])] = (i,0,255)
                cv2.circle(sidline_mask_colr, (int(pt[0]),int(pt[1])), 2, (0, 140, 255), -1)

        # right points
        if right_points is not None:
            for i, pt in enumerate(right_points):
                sidline_mask_colr[int(pt[1]),int(pt[0])] = (i,255,0)
                cv2.circle(sidline_mask_colr, (int(pt[0]),int(pt[1])), 2, (i,255,0), -1)

        # global window_name
        # window_name = "sideline"
        # cv2.imshow(window_name, sidline_mask_colr)
        # cv2.moveWindow(window_name, (300*2)+100, 100)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ## ----------------------------------------------------------------------------------------------------------------------------

        ## ------------------------------------------------------------------------------------------------------------------------- ##
        ##        GET NECK SIDELINES END
        ## ------------------------------------------------------------------------------------------------------------------------- ## 


        # convert mask with [0,255] to [0,1] # best suited for torch mask
        neck_segment = (neck_segment > 0).astype(np.uint8)

        sideline = {
            "left_line":  (left_center, left_dir),
            "right_line": (right_center, right_dir),
            "left_points": left_points,
            "right_points": right_points
        }

        return neck_segment, neck_poly_direction, sideline, sidline_mask_colr, confidence

    def execute(self, kps, person_mask, face_mask = None, clothes_mask = None, neck_polygon_mode="torso_mid_ortho", trim_l_top=0.0, trim_l_btm=32.0, trim_r_top=0.0, trim_r_btm=32.0, max_ang_tol=30):
        person_mask_cv = convert_to_opencv_mask(person_mask)
        if person_mask_cv is None:
            raise ValueError("person_mask couldn't be converted to opencv")
  
        if kps is None:
            raise ValueError("kps is required")

        # Convert the neck_mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if person_mask_cv.ndim == 3 and person_mask_cv.shape[2] in [1,3]:
            person_mask_cv = person_mask_cv[:, :, 0]

        face_mask_cv = None
        if face_mask is not None:
            face_mask_cv = convert_to_opencv_mask(face_mask)
            if face_mask_cv.ndim == 3 and face_mask_cv.shape[2] in [1,3]:
                face_mask_cv = face_mask_cv[:, :, 0]

        clothes_mask_cv = None
        if clothes_mask is not None:
            clothes_mask_cv = convert_to_opencv_mask(clothes_mask)
            if clothes_mask_cv.ndim == 3 and clothes_mask_cv.shape[2] in [1,3]:
                clothes_mask_cv = clothes_mask_cv[:, :, 0]

        l_height_percent = (trim_l_top, 100.0-trim_l_btm)
        r_height_percent = (trim_r_top, 100.0-trim_r_btm)
        # neck_mask, lapex_pt, rapex_pt = self.extract_neck_segment(kps, person_mask_cv, face_mask_cv, clothes_mask_cv, neck_polygon_mode)
        neck_mask, neck_dir, sidelines, sidline_img, confidence = self.extract_neck_segment(kps, person_mask_cv, face_mask_cv, clothes_mask_cv, neck_polygon_mode, l_height_percent, r_height_percent, max_ang_tol)


        ## if neck_dir is None, it throws TypeError: 'NoneType' object is not subscriptable lapex_pt[0] 
        ## Handling this error
        if  neck_dir is None:
            neck_dir = [0,0]

        if neck_mask is None:
            ## could not extract neck segment so passing the person mask
            neck_mask = person_mask_cv

        if sidline_img is None:
            ## could not extract neck segment so passing the person mask
            sidline_img = cv2.cvtColor(person_mask_cv.copy(), cv2.COLOR_GRAY2RGB)

        neck_mask_torch = convert_opencv_mask_to_torch(neck_mask)
        sidline_img_torch = convert_opencv_image_to_torch(sidline_img)
        nck_dirxn = json.dumps({"x": neck_dir[0], "y": neck_dir[1]})

        sidelines = json.dumps(self.to_json_serializable(sidelines))

        return (neck_mask_torch, nck_dirxn, sidelines, sidline_img_torch, confidence)


class GetNeckSidelines:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "neck_mask":  ("MASK",),  # mask of person's neck
                "l_nck_apx": ("STRING",), # left seed
                "r_nck_apx": ("STRING",), # right seed
                "nck_dirxn": ("STRING",), # neck direction
            },
            "optional": {
                "trim_top":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),  # percentage to trim from top of neck mask
                "trim_btm":  ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}), # percentage to trim from bottom of neck mask
            }
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sidelines")
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"


    def merge_contours(self, mask):
        # 1. Ensure binary mask [0,255]
        if mask.dtype != np.uint8 or mask.max() != 255:
            mask = (mask > 0).astype(np.uint8) * 255
        
        # 2. DILATE to bridge 4px gap (kernel size 5-7 works for 4px)
        kernel = np.ones((5,5), np.uint8)  # Adjust size: larger = bridges bigger gaps
        dilated = cv2.dilate(mask, kernel, iterations=2)
        
        # 3. FILL HOLES within dilated regions
        filled = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        
        # 4. ERODE back to approximate original size
        eroded = cv2.erode(filled, kernel, iterations=2)
        
        return eroded

    def remove_small_contours(self, mask, k_area=100):
        # Ensure binary mask [0,255]
        if mask.dtype != np.uint8 or mask.max() != 255:
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create blank output mask
        result = np.zeros_like(mask)
        
        largest_area = 0
        largest_contour = None
        # Keep only contours with area >= k_area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= k_area:
                cv2.fillPoly(result, [contour], 255)  # Fill contour with white
            if area > largest_area:
                largest_area = area
                largest_contour = contour
        
        return result, largest_contour

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def project_points(self, points, origin, direction):
        """
        Projects points onto a direction vector.
        Returns signed scalar projection values.
        """
        return np.dot(points - origin, direction)

    def compute_segment_lengths_with_half_influence_last_pt(self, points):
        """
        Compute the segment lengths (Euclidean distance between consecutive points).
        Only for last endpoint, add the avg of the segment connected to it.
        Suppose if you have points A, B, C, D, your lengths are L_1 (AB), L_2 (BC), L_3 (CD)
            Point 1: w_1 = L_1
            Point 2: w_2 = L_2
            Point 3: w_3 = L_3
            Point 4: w_4 = L_3 / 2
        """
        # Convert inputs to numpy arrays
        points = np.array(points)
        n = len(points)
        if n < 2:
            raise ValueError("At least 2 points are required to define segments.")

        # Calculate N-1 segment lengths
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)

        # Append last length
        last_val = lengths[-1] / 2  # Get last length, divide by 2    
        return np.append(lengths, last_val)

    def filter_by_direction_and_segment_length(self, points, mid_origin, mid_dir, threshold=0.8, dist_th=2.0):
        """
        Filters neck side points based on alignment with neck direction.
        Uses np.diff and marks both endpoints of valid segments automatically.
        Also filter out iF the segment length < dist_th=2.0

        Args:
            points: np.array of shape (N,2) contour points (either left or right side)
            mid_dir: normalized neck direction vector (top -> bottom)
            mid_origin: reference point for projection (e.g., mid between seeds)
            threshold: minimum alignment with mid_dir (0-1)
            dist_th: minimum magnitude for segment length

        Returns:
            filtered_points: np.array of filtered points
        """
        if len(points) < 2:
            return points

        # # ---- Step 1: Sort points along neck direction for stable ordering ----
        # heights = project_points(points, mid_origin, mid_dir)
        # order = np.argsort(heights)
        # points = points[order]

        # ---- Step 2: Compute local differences ----
        dx = np.diff(points[:, 0], prepend=points[0, 0])
        dy = np.diff(points[:, 1], prepend=points[0, 1])
        dirs = np.stack([dx, dy], axis=1)

        # ---- Step 3: Normalize directions ----
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dirs = dirs / norms

        # ---- Step 4: Compute alignment with neck direction ----
        alignment = np.abs(np.dot(dirs, mid_dir))

        # valid = (alignment >= threshold)
        valid = (alignment >= threshold) & (norms[:,0] > dist_th) # norms[:,0] converts norm.shape = (n,1)  to (n,). 
        # otherwise numpy will broadcast value of (alignment >= threshold) to (n,1) and resulting valid will be of shape = (n,n)

        # ---- Step 5: Include both endpoints of valid segments ----
        filtered_mask = np.zeros(len(points), dtype=bool)
        filtered_mask[1:]  |= valid[1:]   # segment end points
        filtered_mask[:-1] |= valid[1:]   # segment start points

        filtered_points = points[filtered_mask]
        return filtered_points

    def sort_points_by_direction(self, P, v):
        """
        Sort points P along unit direction vector v.
        
        Args:
            P: list of [x,y] points or Nx2 numpy array
            v: unit direction vector [vx, vy]
        
        Returns:
            Sorted list of points
        """
        # Convert P to numpy if needed
        if not isinstance(P, np.ndarray):
            P = np.array(P)

        # Project each point onto v: scalar projection = dot(P, v)
        projections = np.dot(P, v)
        
        # Get sorted indices
        sorted_indices = np.argsort(projections)
        
        return P[sorted_indices]

    def find_mask_edge(self, start_point, direction, mask, coarse_step=1.0, max_distance=1e6):
        """
        Ray march from start_point inside a binary mask and return the
        last pixel inside the mask (edge pixel) 

        Args:
        - start_point: np.array([x, y]) - starting point inside mask
        - direction: np.array([dx, dy]) - direction vector
        - mask: 2D np.array - binary mask (1=inside, 0=outside)
        - coarse_step: float - how far to step along the direction each iteration
        - max_distance: float - maximum distance to march

        Returns:
        - edge_pixel: tuple of ints (y, x) - last pixel inside the mask
        - distance: total distance marched
        """

        start_point = np.asarray(start_point, dtype=float)
        direction   = np.asarray(direction, dtype=float)

        # --- Normalize direction safely ---
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero")
        direction = direction / norm

        shape = np.array(mask.shape)

        def in_bounds(idx):
            return np.all(idx >= 0) and np.all(idx < shape)

        # --- Verify start is valid ---
        current = start_point.copy()
        idx = np.floor(current).astype(int)

        # Verify start is inside mask
        if not in_bounds(np.array((idx[1], idx[0]))):
            print("idx=",idx, " shape=",shape)
            raise ValueError("Start point is out of bounds")
        if not mask[idx[1], idx[0]]:  # mask[y, x]
            raise ValueError("Start point is not inside mask")
        distance = 0.0

        # March along the direction until outside the mask
        while distance < max_distance:

            prev = current.copy()
            current = current + coarse_step * direction
            distance += coarse_step

            idx = np.floor(current).astype(int)

            # If out of bounds, return last pixel inside
            if not in_bounds(np.array((idx[1], idx[0]))):
                edge_pixel = np.floor(prev).astype(int)
                return edge_pixel, distance  # return (y, x)

            # If outside mask, return last pixel inside
            if not mask[idx[1], idx[0]]:
                edge_pixel = np.floor(prev).astype(int)
                return edge_pixel, distance

        # If max_distance reached without leaving mask
        edge_pixel = np.floor(current).astype(int)
        return edge_pixel, distance

    def fit_line_pca_weighted_and_sign_enforced(self, points, weights):
        """
        Fit 2D line using covariance eigen-decomposition with weighted points.
        Since PCA itself has sign ambiguity and eigenvectors are defined up to ±
        This func explicitly choose the sign based on the input order.
        Returns (center_point, direction_vector)
        
        Args:
        points (list or np.array): 2D coordinates of points
        weights (list or np.array): Weights (segment lengths) corresponding to each point
        
        Returns:
        center (np.array): The center point of the points
        direction (np.array): The direction vector of the fitted line
        """

        # Convert inputs to numpy arrays
        points = np.array(points)
        weights = np.array(weights)
        sum_of_weights = np.sum(weights)
        # Compute weighted center (mean)
        if sum_of_weights == 0:
            weighted_center = np.mean(points, axis=0)  # Fall back to unweighted mean
        else:
            weighted_center = np.average(points, axis=0, weights=weights)

        # Subtract weighted center from points
        shifted = points - weighted_center

        # Compute the weighted covariance matrix
        if sum_of_weights == 0:
            weighted_cov = np.cov(shifted.T)  # Unweighted covariance
        else:
            weighted_cov = np.cov(shifted.T, aweights=weights)

        # print("weighted_cov=",weighted_cov)
        # Eigen decomposition of the weighted covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(weighted_cov)

        # The eigenvector corresponding to the largest eigenvalue is the principal direction
        largest_idx = np.argmax(eigenvalues)
        direction = eigenvectors[:, largest_idx]

        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # ======================================================
        # Enforce deterministic sign based on ordering
        # ======================================================
        if len(points) >= 2:
            order_vector = points[-1] - points[0]

            # If first and last are (almost) identical
            if np.linalg.norm(order_vector) < 1e-12 and len(points) >= 3:
                order_vector = points[-2] - points[0]

            # Only enforce if we have meaningful direction info
            if np.linalg.norm(order_vector) > 1e-12:
                if np.dot(direction, order_vector) < 0:
                    direction = -direction

        return weighted_center, direction

    def to_json_serializable(self, data):
        """Recursively convert ndarray to list in dicts, lists, AND tuples."""

        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            result = {}
            for k, v in data.items():
                result[k] = self.to_json_serializable(v)
            return result
        elif isinstance(data, list):
            result = []
            for item in data:
                result.append(self.to_json_serializable(item))
            return result
        elif isinstance(data, tuple):
            result = []
            for item in data:
                result.append(self.to_json_serializable(item))
            return tuple(result)
        return data

    def get_neck_sideline_from_mask(self, neck_mask, left_seed, right_seed, mid_dir, height_percentage=(0, 95)):
        """
        Returns fitted left and right neck side lines.
        Use height_percentage to filter out cavicle and jawline

        Args:
            neck_mask: binary mask where the neck+upper body region is 1/255.
            left_seed: a point that is near the left side of the neck side line
            right_seed: a point that is near the right side of the neck side line
            mid_dir: a unit direction  vector of the neck's midline from top mid of the neck to bottom mid of the neck
            height_percentage: percentage values to keep the top and bottom region of the neck mask

        Returns:
            left_points, right_points: Nx2 arrays of float32 points for both the neck side line.
            left_center, left_center: The center point of the points for both the neck side line (np.array)
            left_dir, right_dir : The direction vector of the fitted line for both the neck side line (np.array)
        """

        ## ---- Step 1: Normalize neck direction ----
        mid_dir = self.normalize(np.array(mid_dir))


        ## ---- Step 2: Compute perpendicular direction ----
        perp_dir = np.array([-mid_dir[1], mid_dir[0]])


        ## ---- Step 3: Extract contour points ----    
        mask = (neck_mask > 0).astype(np.uint8)
        mask = self.merge_contours(mask)
        mask, max_contour = self.remove_small_contours(mask)

        if max_contour is None:
            return None


        ## Visualise
        mask_colr = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2RGB)


        ## ---- Step 4: Estimate midline origin  and mid_top  ----
        left_seed = np.array(left_seed)
        right_seed = np.array(right_seed)
        mid_origin = (left_seed + right_seed) / 2.0
        mid_top, dist_to_top = self.find_mask_edge(mid_origin, -mid_dir, mask, coarse_step=1.0)


        ## ---- Step 5: Remove top & bottom artifacts using height filtering ----

        ## heights is the distance along neck direction from any contour point to mid_top_line
        ## Project vector (contour point - mid_top) onto  mid_dir vector,
        ## this gives us only the  height component from any contour point to mid_top along the direction of mid_dir
        heights = self.project_points(max_contour, mid_top, mid_dir)

        low_pct = height_percentage[0] / 100.0
        high_pct = height_percentage[1] / 100.0
        min_h, max_h = heights.min(), heights.max()
        height_range =  (max_h - min_h)

        ## Compute absolute height thresholds
        low_height  = min_h + low_pct  * height_range
        high_height = min_h + high_pct * height_range

        ## Apply height filtering 
        valid_contour = max_contour[(heights >= low_height) & (heights <= high_height)]

        ## Debug
        # print("min_h=",min_h, " max_h=",max_h, " range=",height_range)
        # print("low_height=",low_height)
        # print("high_height=",high_height)
        # for i,h in enumerate(heights):
        #     print(i, " valid=", ((h >= low_height) & (h <= high_height)),  " c=",max_contour[i], " h=", h)


        ## ---- Step 6: Separate left and right using true midline ----
        side_values = self.project_points(valid_contour, mid_origin, perp_dir)
        left_points  = valid_contour[side_values < 0]
        right_points = valid_contour[side_values > 0]


        ## ---- Step 7 : Remove points that has direction more than 20% from mid_dir for both left and right  ----
        left_points  = self.filter_by_direction_and_segment_length(left_points,  mid_origin, mid_dir, threshold=0.9, dist_th=2.0)
        right_points = self.filter_by_direction_and_segment_length(right_points, mid_origin, mid_dir, threshold=0.9, dist_th=2.0)


        ## ---- Step 8 : Sort points in direction of mid_dir  ----
        left_points  =  self.sort_points_by_direction(left_points, mid_dir)
        right_points =  self.sort_points_by_direction(right_points, mid_dir)


        ## ---- Step 9: Compute weights based on segment lengths ----
        left_weights  = self.compute_segment_lengths_with_half_influence_last_pt(left_points)
        right_weights = self.compute_segment_lengths_with_half_influence_last_pt(right_points)


        ## ---- Step 10: Fit line to each side ----
        left_center, left_dir   = self.fit_line_pca_weighted_and_sign_enforced(left_points, left_weights)
        right_center, right_dir = self.fit_line_pca_weighted_and_sign_enforced(right_points, right_weights)

        
        ## Visualise

        ## mid_line
        end_point = mid_top + (mid_dir * height_range)
        cv2.line(mask_colr, tuple(mid_top.astype(int)), tuple(end_point.astype(int)),  (255,120,0), 1)

        ## perpendicular to mid_dir
        mid_dir_perp = self.normalize(np.array([-mid_dir[1], mid_dir[0]]))
        end_point = mid_top + (mid_dir_perp * height_range)
        cv2.line(mask_colr, tuple(mid_top.astype(int)), tuple(end_point.astype(int)),  (255,120,0), 1)
        end_point = mid_top - (mid_dir_perp * height_range)
        cv2.line(mask_colr, tuple(mid_top.astype(int)), tuple(end_point.astype(int)),  (255,120,0), 1)
    
        # Right side line
        r_srt_point = right_center - (right_dir * 40)
        r_end_point = right_center + (right_dir * 40)
        # cv2.line(mask_colr, tuple(r_srt_point.astype(int)), tuple(r_end_point.astype(int)),  (0,120,255), 1)
        cv2.line(mask_colr, tuple(right_center.astype(int)), tuple(r_end_point.astype(int)),  (0,120,255), 1)

        # Left side line
        l_srt_point = left_center - (left_dir * 40)
        l_end_point = left_center + (left_dir * 40)
        # cv2.line(mask_colr, tuple(l_srt_point.astype(int)), tuple(l_end_point.astype(int)),  (0,255,200), 1)
        cv2.line(mask_colr, tuple(left_center.astype(int)), tuple(l_end_point.astype(int)),  (0,255,200), 1)

        # Draw points in red
        # src_L = self.resample_polyline(right_points)        
        # for pt in src_L:
        for pt in right_points:
            mask_colr[int(pt[1]),int(pt[0])] = (0, 0, 255)

        # Draw points in green
        for pt in left_points:
            mask_colr[int(pt[1]),int(pt[0])] = (0, 255, 0)


        ## mid_origin pt
        cv2.circle(mask_colr, tuple(mid_origin.astype(int)), 2, (0, 140, 255), -1)

        ## top mid pt
        mask_colr[int(mid_top[1]),int(mid_top[0])] = (100, 0, 255)

        # global window_name
        window_name = "Neck Sidelines"
        cv2.imshow(window_name, mask_colr)
        cv2.moveWindow(window_name, (300*2)+100, 100)
        print("Neck Sidelines b4 wait")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Neck Sidelines after wait and wind destroyed")



        ## Return
        return {
            "left_line":  (left_center, left_dir),
            "right_line": (right_center, right_dir),
            "left_points": left_points,
            "right_points": right_points
        }

    def execute(self, neck_mask, l_nck_apx, r_nck_apx, nck_dirxn, trim_top = 0.0, trim_btm = 5.0):
        neck_mask_cv = convert_to_opencv_mask(neck_mask)
        if neck_mask_cv is None:
            raise ValueError("neck_mask couldn't be converted to opencv")
  
        if l_nck_apx is None:
            raise ValueError("l_nck_apx is required")
        if r_nck_apx is None:
            raise ValueError("r_nck_apx is required")
        if nck_dirxn is None:
            raise ValueError("nck_dirxn is required")

        # Convert the neck_mask of shape (H, W, 1) or (H, W, 3) to (H, W)
        if neck_mask_cv.ndim == 3 and neck_mask_cv.shape[2] in [1,3]:
            neck_mask_cv = neck_mask_cv[:, :, 0]

        data = json.loads(l_nck_apx)
        left_seed = np.array([data["x"], data["y"]])

        data = json.loads(r_nck_apx)
        right_seed = np.array([data["x"], data["y"]])

        data = json.loads(nck_dirxn)
        neck_dir = np.array([data["x"], data["y"]])

        height_percentage = (trim_top, 100.0-trim_btm)


        res_sidelines = self.get_neck_sideline_from_mask(neck_mask_cv, left_seed, right_seed, neck_dir, height_percentage)
        
        sidelines = json.dumps(self.to_json_serializable(res_sidelines))

        return (sidelines,)


class FaceNeckAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",), # src img contains face and neck which needs to be aligned to dst face and neck
                "src_sideline": ("STRING",), # sidelines of source face's neck
                "dest_sideline": ("STRING",), # sidelines of destination face's neck
                "dest_width": ("INT",), # width of destination 
                "dest_height": ("INT",), # height of destination 
            }
        }

    RETURN_TYPES = ("IMAGE", "MAT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MAT","width", "height" )
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"



    def resample_polyline(self, points, n=50):
        """Uniformly resample a polyline by arc length."""
        points = np.asarray(points, dtype=np.float32)

        d = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=1))
        s = np.insert(np.cumsum(d), 0, 0)
        s /= s[-1]

        u = np.linspace(0, 1, n)
        out = np.zeros((n, 2), dtype=np.float32)
        for i in range(2):
            out[:, i] = np.interp(u, s, points[:, i])
        return out

    def angle_of_vector(self, v):
        return np.arctan2(v[1], v[0])


    def neck_based_similarity_transform( self, src_img, src_left_neck, src_right_neck, src_left_dir, src_right_dir, tgt_left_neck, tgt_right_neck, tgt_left_dir, tgt_right_dir, tgt_w, tgt_h):
        """
        Returns transformed source image and 2x3 affine matrix.
        """

        # Resample
        src_L = self.resample_polyline(src_left_neck)
        src_R = self.resample_polyline(src_right_neck)
        tgt_L = self.resample_polyline(tgt_left_neck)
        tgt_R = self.resample_polyline(tgt_right_neck)


        # -----------------------------
        # SCALE (top neck width)
        # -----------------------------
        # top 
        pLs, pRs = src_L[0], src_R[0]
        pLt, pRt = tgt_L[0], tgt_R[0]

        w_src = np.linalg.norm(pRs - pLs)
        w_tgt = np.linalg.norm(pRt - pLt)

        scale = w_tgt / (w_src + 1e-6)
        print("SCALE =", scale)

        # -----------------------------
        # ROTATION (average neck direction)
        # -----------------------------
        v_src = 0.5 * (src_left_dir + src_right_dir)
        v_tgt = 0.5 * (tgt_left_dir + tgt_right_dir)

        theta = self.angle_of_vector(v_tgt) - self.angle_of_vector(v_src)

        print("ROTATION =",np.degrees(theta))

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        R = np.array([[cos_t, -sin_t],
                    [sin_t,  cos_t]], dtype=np.float32)

        # -----------------------------
        # TRANSLATION (top neck)
        # -----------------------------
        # ## Apply scale + rotation to source top points
        # pLs_tr = scale * (R @ pLs)
        # pRs_tr = scale * (R @ pRs)

        # ## Hard constraint: both sides aligned
        # ## Translation (average shift)
        # T = 0.5 * ((pLt - pLs_tr) + (pRt - pRs_tr))

        # -----------------------------
        # TRANSLATION (bottom neck)
        # -----------------------------
        ## Bottom part of neck sideline
        pLbs, pRbs = src_L[-1], src_R[-1]
        pLbt, pRbt = tgt_L[-1], tgt_R[-1]

        # Apply rotation + scale to bottom source points
        pLbs_tr = scale * (R @ pLbs)
        pRbs_tr = scale * (R @ pRbs)

        ## Hard constraint: both sides aligned
        ## Translation (average shift)
        T = 0.5 * ((pLbt - pLbs_tr) + (pRbt - pRbs_tr))
        print("Translation =",T)






        # -----------------------------
        # Final affine matrix
        # -----------------------------
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:, 2] = T

        h, w = src_img.shape[:2]
        # warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR)
        warped = cv2.warpAffine(src_img, M, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)

        # for i, pt in enumerate(tgt_L):
        #     warped[int(pt[1]),int(pt[0])] = (100, i, 255)

        # for i, pt in enumerate(tgt_R):
        #     warped[int(pt[1]),int(pt[0])] = (200, i, 255)

        # center
        src_pts = np.vstack((pLs, pRs)) 
        tgt_pts = np.vstack((pLs, pRs)) 
        src_center = np.mean(src_pts, axis=0)
        tgt_center = np.mean(tgt_pts, axis=0)
        # Ls_center = np.mean(pLs, axis=0)
        # Rs_center = np.mean(pRs, axis=0)
        # Lt_center = np.mean(pLt, axis=0)
        # Rt_center = np.mean(pRt, axis=0)

        # avg dir
        end_point = src_center + (v_src * 40)
        cv2.line(warped, tuple(src_center.astype(int)), tuple(end_point.astype(int)), (100, 0, 255), 1)
        end_point = tgt_center + (v_tgt * 40)
        cv2.line(warped, tuple(tgt_center.astype(int)), tuple(end_point.astype(int)), (0, 255, 100), 1)     

        # top
        warped[int(pLs[1]),int(pLs[0])] = (100, 0, 255)
        warped[int(pRs[1]),int(pRs[0])] = (200, 0, 255)
        warped[int(pLt[1]),int(pLt[0])] = (0, 255, 100)
        warped[int(pRt[1]),int(pRt[0])] = (0, 255, 150)

        # btm
        warped[int(pLbs[1]),int(pLbs[0])] = (100, 0, 255)
        warped[int(pRbs[1]),int(pRbs[0])] = (200, 0, 255)
        warped[int(pLbt[1]),int(pLbt[0])] = (0, 255, 100)
        warped[int(pRbt[1]),int(pRbt[0])] = (0, 255, 150)
    
        return warped, M


    def execute(self, src, src_sideline, dest_sideline, dest_width, dest_height):
        
        src_img = convert_to_opencv_image(src)
        if src_img is None:
            raise ValueError("src image couldn't be converted to opencv")

        if src_sideline is None:
            raise ValueError("src_sideline is required")

        if dest_sideline is None:
            raise ValueError("dest_sideline is required")

        if dest_width is None:
            raise ValueError("dest_width is required")
            
        if dest_height is None:
            raise ValueError("dest_height is required")


        src_data = json.loads(src_sideline)
        if src_data is None:
            raise ValueError("src_sideline is required")

        tgt_data = json.loads(dest_sideline)
        if tgt_data is None:
            raise ValueError("dest_sideline is required")

        
        src_left_neck = np.array(src_data["left_points"])
        src_right_neck = np.array(src_data["right_points"])
        src_left_dir = np.array(src_data["left_line"][1])
        src_right_dir = np.array(src_data["right_line"][1])

        
        tgt_left_neck = np.array(tgt_data["left_points"])
        tgt_right_neck = np.array(tgt_data["right_points"])
        tgt_left_dir = np.array(tgt_data["left_line"][1])
        tgt_right_dir = np.array(tgt_data["right_line"][1])

        if not None in [src_left_neck, src_right_neck, src_left_dir, src_right_dir, tgt_left_neck, tgt_right_neck, tgt_left_dir, tgt_right_dir]:
            warped_src_face, M = self.neck_based_similarity_transform( src_img, src_left_neck, src_right_neck, src_left_dir, src_right_dir, tgt_left_neck, tgt_right_neck, tgt_left_dir, tgt_right_dir, dest_width, dest_height)
        else:
            warped_src_face = src_img
            M = np.zeros((2, 3), dtype=np.float32)
        h, w = warped_src_face.shape[:2]
        

            
        align_src_torch = convert_opencv_image_to_torch(warped_src_face)
        
        return (align_src_torch, M, w, h, )

#############################################################################################################################################################
#############################################################################################################################################################
class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)

        return(outputs, outputs.shape[2], outputs.shape[1],)

class ImageFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "axis": (["x", "y", "xy"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, axis):
        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        image = torch.flip(image, dim)

        return(image,)

class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT",)
    RETURN_NAMES = ("IMAGE","x","y",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, width, height, position, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return(image, x, y, )

class ImageTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "overlap": ("FLOAT", { "default": 0, "min": 0, "max": 0.5, "step": 0.01, }),
                "overlap_x": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "tile_width", "tile_height", "overlap_x", "overlap_y",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, rows, cols, overlap, overlap_x, overlap_y):
        h, w = image.shape[1:3]
        tile_h = h // rows
        tile_w = w // cols
        h = tile_h * rows
        w = tile_w * cols
        overlap_h = int(tile_h * overlap) + overlap_y
        overlap_w = int(tile_w * overlap) + overlap_x

        # max overlap is half of the tile size
        overlap_h = min(tile_h // 2, overlap_h)
        overlap_w = min(tile_w // 2, overlap_w)

        if rows == 1:
            overlap_h = 0
        if cols == 1:
            overlap_w = 0
        
        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h + overlap_h
                x2 = x1 + tile_w + overlap_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h - overlap_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w - overlap_w

                tiles.append(image[:, y1:y2, x1:x2, :])
        tiles = torch.cat(tiles, dim=0)

        return(tiles, tile_w+overlap_w, tile_h+overlap_h, overlap_w, overlap_h,)

class ImageUntile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "overlap_x": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, tiles, overlap_x, overlap_y, rows, cols):
        tile_h, tile_w = tiles.shape[1:3]
        tile_h -= overlap_y
        tile_w -= overlap_x
        out_w = cols * tile_w
        out_h = rows * tile_h

        out = torch.zeros((1, out_h, out_w, tiles.shape[3]), device=tiles.device, dtype=tiles.dtype)

        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_y
                if j > 0:
                    x1 -= overlap_x

                y2 = y1 + tile_h + overlap_y
                x2 = x1 + tile_w + overlap_x

                if y2 > out_h:
                    y2 = out_h
                    y1 = y2 - tile_h - overlap_y
                if x2 > out_w:
                    x2 = out_w
                    x1 = x2 - tile_w - overlap_x
                
                mask = torch.ones((1, tile_h+overlap_y, tile_w+overlap_x), device=tiles.device, dtype=tiles.dtype)

                # feather the overlap on top
                if i > 0 and overlap_y > 0:
                    mask[:, :overlap_y, :] *= torch.linspace(0, 1, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on bottom
                #if i < rows - 1:
                #    mask[:, -overlap_y:, :] *= torch.linspace(1, 0, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on left
                if j > 0 and overlap_x > 0:
                    mask[:, :, :overlap_x] *= torch.linspace(0, 1, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                # feather the overlap on right
                #if j < cols - 1:
                #    mask[:, :, -overlap_x:] *= torch.linspace(1, 0, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, tiles.shape[3])
                tile = tiles[i * cols + j] * mask
                out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1 - mask) + tile
        return(out, )

class ImageSeamCarving:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "energy": (["backward", "forward"],),
                "order": (["width-first", "height-first"],),
            },
            "optional": {
                "keep_mask": ("MASK",),
                "drop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials/image manipulation"
    FUNCTION = "execute"

    def execute(self, image, width, height, energy, order, keep_mask=None, drop_mask=None):
        from .carve import seam_carving

        img = image.permute([0, 3, 1, 2])

        if keep_mask is not None:
            #keep_mask = keep_mask.reshape((-1, 1, keep_mask.shape[-2], keep_mask.shape[-1])).movedim(1, -1)
            keep_mask = keep_mask.unsqueeze(1)

            if keep_mask.shape[2] != img.shape[2] or keep_mask.shape[3] != img.shape[3]:
                keep_mask = F.interpolate(keep_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")
        if drop_mask is not None:
            drop_mask = drop_mask.unsqueeze(1)

            if drop_mask.shape[2] != img.shape[2] or drop_mask.shape[3] != img.shape[3]:
                drop_mask = F.interpolate(drop_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")

        out = []
        for i in range(img.shape[0]):
            resized = seam_carving(
                T.ToPILImage()(img[i]),
                size=(width, height),
                energy_mode=energy,
                order=order,
                keep_mask=T.ToPILImage()(keep_mask[i]) if keep_mask is not None else None,
                drop_mask=T.ToPILImage()(drop_mask[i]) if drop_mask is not None else None,
            )
            out.append(T.ToTensor()(resized))

        out = torch.stack(out).permute([0, 2, 3, 1])

        return(out, )

class ImageRandomTransform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "repeat": ("INT", { "default": 1, "min": 1, "max": 256, "step": 1, }),
                "variation": ("FLOAT", { "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, seed, repeat, variation):
        h, w = image.shape[1:3]
        image = image.repeat(repeat, 1, 1, 1).permute([0, 3, 1, 2])

        distortion = 0.2 * variation
        rotation = 5 * variation
        brightness = 0.5 * variation
        contrast = 0.5 * variation
        saturation = 0.5 * variation
        hue = 0.2 * variation
        scale = 0.5 * variation

        torch.manual_seed(seed)

        out = []
        for i in image:
            tramsforms = T.Compose([
                T.RandomPerspective(distortion_scale=distortion, p=0.5),
                T.RandomRotation(degrees=rotation, interpolation=T.InterpolationMode.BILINEAR, expand=True),
                T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=(-hue, hue)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop((h, w), scale=(1-scale, 1+scale), ratio=(w/h, w/h), interpolation=T.InterpolationMode.BICUBIC),
            ])
            out.append(tramsforms(i.unsqueeze(0)))

        out = torch.cat(out, dim=0).permute([0, 2, 3, 1]).clamp(0, 1)

        return (out,)

class RemBGSession:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["u2net: general purpose", "u2netp: lightweight general purpose", "u2net_human_seg: human segmentation", "u2net_cloth_seg: cloths Parsing", "silueta: very small u2net", "isnet-general-use: general purpose", "isnet-anime: anime illustrations", "sam: general purpose"],),
                "providers": (['CPU', 'CUDA', 'ROCM', 'DirectML', 'OpenVINO', 'CoreML', 'Tensorrt', 'Azure'],),
            },
        }

    RETURN_TYPES = ("REMBG_SESSION",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, model, providers):
        from rembg import new_session, remove

        model = model.split(":")[0]

        class Session:
            def __init__(self, model, providers):
                self.session = new_session(model, providers=[providers+"ExecutionProvider"])
            def process(self, image):
                return remove(image, session=self.session)
            
        return (Session(model, providers),)

class TransparentBGSession:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["base", "fast", "base-nightly"],),
                "use_jit": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("REMBG_SESSION",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, mode, use_jit):
        from transparent_background import Remover

        class Session:
            def __init__(self, mode, use_jit):
                self.session = Remover(mode=mode, jit=use_jit)
            def process(self, image):
                return self.session.process(image)

        return (Session(mode, use_jit),)

class ImageRemoveBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rembg_session": ("REMBG_SESSION",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, rembg_session, image):
        image = image.permute([0, 3, 1, 2])
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = rembg_session.process(img)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])
        # output = output[:, :, :, :3]

        return(output, mask,)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Image processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageDesaturate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", { "default": 1.00, "min": 0.00, "max": 1.00, "step": 0.05, }),
                "method": (["luminance (Rec.709)", "luminance (Rec.601)", "average", "lightness"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, factor, method):
        if method == "luminance (Rec.709)":
            grayscale = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        elif method == "luminance (Rec.601)":
            grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        elif method == "average":
            grayscale = image.mean(dim=3)
        elif method == "lightness":
            grayscale = (torch.max(image, dim=3)[0] + torch.min(image, dim=3)[0]) / 2

        grayscale = (1.0 - factor) * image + factor * grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
        grayscale = torch.clamp(grayscale, 0, 1)

        return(grayscale,)

class PixelOEPixelize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_mode": (["contrast", "bicubic", "nearest", "center", "k-centroid"],),
                "target_size": ("INT", { "default": 128, "min": 0, "max": MAX_RESOLUTION, "step": 8 }),
                "patch_size": ("INT", { "default": 16, "min": 4, "max": 32, "step": 2 }),
                "thickness": ("INT", { "default": 2, "min": 1, "max": 16, "step": 1 }),
                "color_matching": ("BOOLEAN", { "default": True }),
                "upscale": ("BOOLEAN", { "default": True }),
                #"contrast": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                #"saturation": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, downscale_mode, target_size, patch_size, thickness, color_matching, upscale):
        from pixeloe.pixelize import pixelize

        image = image.clone().mul(255).clamp(0, 255).byte().cpu().numpy()
        output = []
        for img in image:
            img = pixelize(img,
                           mode=downscale_mode,
                           target_size=target_size,
                           patch_size=patch_size,
                           thickness=thickness,
                           contrast=1.0,
                           saturation=1.0,
                           color_matching=color_matching,
                           no_upscale=not upscale)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0).permute([0, 2, 3, 1])

        return(output,)

class ImagePosterize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, threshold):
        image = image.mean(dim=3, keepdim=True)
        image = (image > threshold).float()
        image = image.repeat(1, 1, 1, 3)

        return(image,)

# From https://github.com/yoonsikp/pycubelut/blob/master/pycubelut.py (MIT license)
class ImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (folder_paths.get_filename_list("luts"),),
                "gamma_correction": ("BOOLEAN", { "default": True }),
                "clip_values": ("BOOLEAN", { "default": True }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1 }),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    # TODO: check if we can do without numpy
    def execute(self, image, lut_file, gamma_correction, clip_values, strength):
        lut_file_path = folder_paths.get_full_path("luts", lut_file)
        if not lut_file_path or not Path(lut_file_path).exists():
            print(f"Could not find LUT file: {lut_file_path}")
            return (image,)
            
        from colour.io.luts.iridas_cube import read_LUT_IridasCube
        
        device = image.device
        lut = read_LUT_IridasCube(lut_file_path)
        lut.name = lut_file

        if clip_values:
            if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
            else:
                if len(lut.table.shape) == 2:  # 3x1D
                    for dim in range(3):
                        lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                else:  # 3D
                    for dim in range(3):
                        lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

        out = []
        for img in image: # TODO: is this more resource efficient? should we use a batch instead?
            lut_img = img.cpu().numpy().copy()

            is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
            dom_scale = None
            if is_non_default_domain:
                dom_scale = lut.domain[1] - lut.domain[0]
                lut_img = lut_img * dom_scale + lut.domain[0]
            if gamma_correction:
                lut_img = lut_img ** (1/2.2)
            lut_img = lut.apply(lut_img)
            if gamma_correction:
                lut_img = lut_img ** (2.2)
            if is_non_default_domain:
                lut_img = (lut_img - lut.domain[0]) / dom_scale

            lut_img = torch.from_numpy(lut_img).to(device)
            if strength < 1.0:
                lut_img = strength * lut_img + (1 - strength) * img
            out.append(lut_img)

        out = torch.stack(out)

        return (out, )

# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
class ImageCAS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials/image processing"
    FUNCTION = "execute"

    def execute(self, image, amount):
        epsilon = 1e-5
        img = F.pad(image.permute([0,3,1,2]), pad=(1, 1, 1, 1))

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]

        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_(cross)
        mx = max_(cross)

        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        mx = mx + mx2
        mn = mn + mn2

        # Computing local weight
        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)

        output = output.permute([0,2,3,1])

        return (output,)

class ImageSmartSharpen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_radius": ("INT", { "default": 7, "min": 1, "max": 25, "step": 1, }),
                "preserve_edges": ("FLOAT", { "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "sharpen": ("FLOAT", { "default": 5.0, "min": 0.0, "max": 25.0, "step": 0.5 }),
                "ratio": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1 }),
        }}

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials/image processing"
    FUNCTION = "execute"

    def execute(self, image, noise_radius, preserve_edges, sharpen, ratio):
        import cv2

        output = []
        #diagonal = np.sqrt(image.shape[1]**2 + image.shape[2]**2)
        if preserve_edges > 0:
            preserve_edges = max(1 - preserve_edges, 0.05)

        for img in image:
            if noise_radius > 1:
                sigma = 0.3 * ((noise_radius - 1) * 0.5 - 1) + 0.8 # this is what pytorch uses for blur
                #sigma_color = preserve_edges * (diagonal / 2048)
                blurred = cv2.bilateralFilter(img.cpu().numpy(), noise_radius, preserve_edges, sigma)
                blurred = torch.from_numpy(blurred)
            else:
                blurred = img

            if sharpen > 0:
                sharpened = kornia.enhance.sharpness(img.permute(2,0,1), sharpen).permute(1,2,0)
            else:
                sharpened = img

            img = ratio * sharpened + (1 - ratio) * blurred
            img = torch.clamp(img, 0, 1)
            output.append(img)
        
        del blurred, sharpened
        output = torch.stack(output)

        return (output,)


class ExtractKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("KEYFRAMES", "indexes")

    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, threshold):
        window_size = 2

        variations = torch.sum(torch.abs(image[1:] - image[:-1]), dim=[1, 2, 3])
        #variations = torch.sum((image[1:] - image[:-1]) ** 2, dim=[1, 2, 3])
        threshold = torch.quantile(variations.float(), threshold).item()

        keyframes = []
        for i in range(image.shape[0] - window_size + 1):
            window = image[i:i + window_size]
            variation = torch.sum(torch.abs(window[-1] - window[0])).item()

            if variation > threshold:
                keyframes.append(i + window_size - 1)

        return (image[keyframes], ','.join(map(str, keyframes)),)

class ImageColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["LAB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": ("INT", { "default": 0, "min": 0, "max": 1024, "step": 1, }),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, reference, color_space, factor, device, batch_size, reference_mask=None):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        image = image.permute([0, 3, 1, 2])
        reference = reference.permute([0, 3, 1, 2]).to(device)
         
        # Ensure reference_mask is in the correct format and on the right device
        if reference_mask is not None:
            assert reference_mask.ndim == 3, f"Expected reference_mask to have 3 dimensions, but got {reference_mask.ndim}"
            assert reference_mask.shape[0] == reference.shape[0], f"Frame count mismatch: reference_mask has {reference_mask.shape[0]} frames, but reference has {reference.shape[0]}"
            
            # Reshape mask to (batch, 1, height, width)
            reference_mask = reference_mask.unsqueeze(1).to(device)
             
            # Ensure the mask is binary (0 or 1)
            reference_mask = (reference_mask > 0.5).float()
             
            # Ensure spatial dimensions match
            if reference_mask.shape[2:] != reference.shape[2:]:
                reference_mask = comfy.utils.common_upscale(
                    reference_mask,
                    reference.shape[3], reference.shape[2],
                    upscale_method='bicubic',
                    crop='center'
                )

        if batch_size == 0 or batch_size > image.shape[0]:
            batch_size = image.shape[0]

        if "LAB" == color_space:
            reference = kornia.color.rgb_to_lab(reference)
        elif "YCbCr" == color_space:
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif "LUV" == color_space:
            reference = kornia.color.rgb_to_luv(reference)
        elif "YUV" == color_space:
            reference = kornia.color.rgb_to_yuv(reference)
        elif "XYZ" == color_space:
            reference = kornia.color.rgb_to_xyz(reference)

        reference_mean, reference_std = self.compute_mean_std(reference, reference_mask)

        image_batch = torch.split(image, batch_size, dim=0)
        output = []

        for image in image_batch:
            image = image.to(device)

            if color_space == "LAB":
                image = kornia.color.rgb_to_lab(image)
            elif color_space == "YCbCr":
                image = kornia.color.rgb_to_ycbcr(image)
            elif color_space == "LUV":
                image = kornia.color.rgb_to_luv(image)
            elif color_space == "YUV":
                image = kornia.color.rgb_to_yuv(image)
            elif color_space == "XYZ":
                image = kornia.color.rgb_to_xyz(image)

            image_mean, image_std = self.compute_mean_std(image)

            matched = torch.nan_to_num((image - image_mean) / image_std) * torch.nan_to_num(reference_std) + reference_mean
            matched = factor * matched + (1 - factor) * image

            if color_space == "LAB":
                matched = kornia.color.lab_to_rgb(matched)
            elif color_space == "YCbCr":
                matched = kornia.color.ycbcr_to_rgb(matched)
            elif color_space == "LUV":
                matched = kornia.color.luv_to_rgb(matched)
            elif color_space == "YUV":
                matched = kornia.color.yuv_to_rgb(matched)
            elif color_space == "XYZ":
                matched = kornia.color.xyz_to_rgb(matched)

            out = matched.permute([0, 2, 3, 1]).clamp(0, 1).to(comfy.model_management.intermediate_device())
            output.append(out)

        out = None
        output = torch.cat(output, dim=0)
        return (output,)

    def compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            # Apply mask to the tensor
            masked_tensor = tensor * mask

            # Calculate the sum of the mask for each channel
            mask_sum = mask.sum(dim=[2, 3], keepdim=True)

            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-6)

            # Calculate mean and std only for masked area
            mean = torch.nan_to_num(masked_tensor.sum(dim=[2, 3], keepdim=True) / mask_sum)
            std = torch.sqrt(torch.nan_to_num(((masked_tensor - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / mask_sum))
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return mean, std

class ImageColorMatchAdobe(ImageColorMatch):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["RGB", "LAB"],),
                "luminance_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_intensity_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "fade_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "neutralization_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "device": (["auto", "cpu", "gpu"],),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def analyze_color_statistics(self, image, mask=None):
        # Assuming image is in RGB format
        l, a, b = kornia.color.rgb_to_lab(image).chunk(3, dim=1)

        if mask is not None:
            # Ensure mask is binary and has the same spatial dimensions as the image
            mask = F.interpolate(mask, size=image.shape[2:], mode='nearest')
            mask = (mask > 0.5).float()
            
            # Apply mask to each channel
            l = l * mask
            a = a * mask
            b = b * mask
            
            # Compute masked mean and std
            num_pixels = mask.sum()
            mean_l = (l * mask).sum() / num_pixels
            mean_a = (a * mask).sum() / num_pixels
            mean_b = (b * mask).sum() / num_pixels
            std_l = torch.sqrt(((l - mean_l)**2 * mask).sum() / num_pixels)
            var_ab = ((a - mean_a)**2 + (b - mean_b)**2) * mask
            std_ab = torch.sqrt(var_ab.sum() / num_pixels)
        else:
            mean_l = l.mean()
            std_l = l.std()
            mean_a = a.mean()
            mean_b = b.mean()
            std_ab = torch.sqrt(a.var() + b.var())

        return mean_l, std_l, mean_a, mean_b, std_ab

    def apply_color_transformation(self, image, source_stats, dest_stats, L, C, N):
        l, a, b = kornia.color.rgb_to_lab(image).chunk(3, dim=1)
        
        # Unpack statistics
        src_mean_l, src_std_l, src_mean_a, src_mean_b, src_std_ab = source_stats
        dest_mean_l, dest_std_l, dest_mean_a, dest_mean_b, dest_std_ab = dest_stats

        # Adjust luminance
        l_new = (l - dest_mean_l) * (src_std_l / dest_std_l) * L + src_mean_l

        # Neutralize color cast
        a = a - N * dest_mean_a
        b = b - N * dest_mean_b

        # Adjust color intensity
        a_new = a * (src_std_ab / dest_std_ab) * C
        b_new = b * (src_std_ab / dest_std_ab) * C

        # Combine channels
        lab_new = torch.cat([l_new, a_new, b_new], dim=1)

        # Convert back to RGB
        rgb_new = kornia.color.lab_to_rgb(lab_new)

        return rgb_new

    def execute(self, image, reference, color_space, luminance_factor, color_intensity_factor, fade_factor, neutralization_factor, device, reference_mask=None):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        # Ensure image and reference are in the correct shape (B, C, H, W)
        image = image.permute(0, 3, 1, 2).to(device)
        reference = reference.permute(0, 3, 1, 2).to(device)

        # Handle reference_mask (if provided)
        if reference_mask is not None:
            # Ensure reference_mask is 4D (B, 1, H, W)
            if reference_mask.ndim == 2:
                reference_mask = reference_mask.unsqueeze(0).unsqueeze(0)
            elif reference_mask.ndim == 3:
                reference_mask = reference_mask.unsqueeze(1)
            reference_mask = reference_mask.to(device)

         # Analyze color statistics
        source_stats = self.analyze_color_statistics(reference, reference_mask)
        dest_stats = self.analyze_color_statistics(image)

        # Apply color transformation
        transformed = self.apply_color_transformation(
            image, source_stats, dest_stats, 
            luminance_factor, color_intensity_factor, neutralization_factor
        )

        # Apply fade factor
        result = fade_factor * transformed + (1 - fade_factor) * image

        # Convert back to (B, H, W, C) format and ensure values are in [0, 1] range
        result = result.permute(0, 2, 3, 1).clamp(0, 1).to(comfy.model_management.intermediate_device())

        return (result,)


class ImageHistogramMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (["pytorch", "skimage"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, reference, method, factor, device):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        if "pytorch" in method:
            from .histogram_matching import Histogram_Matching

            image = image.permute([0, 3, 1, 2]).to(device)
            reference = reference.permute([0, 3, 1, 2]).to(device)[0].unsqueeze(0)
            image.requires_grad = True
            reference.requires_grad = True

            out = []

            for i in image:
                i = i.unsqueeze(0)
                hm = Histogram_Matching(differentiable=True)
                out.append(hm(i, reference))
            out = torch.cat(out, dim=0)
            out = factor * out + (1 - factor) * image
            out = out.permute([0, 2, 3, 1]).clamp(0, 1)
        else:
            from skimage.exposure import match_histograms

            out = torch.from_numpy(match_histograms(image.cpu().numpy(), reference.cpu().numpy(), channel_axis=3)).to(device)
            out = factor * out + (1 - factor) * image.to(device)

        return (out.to(comfy.model_management.intermediate_device()),)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageToDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image, device):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        image = image.clone().to(device)
        torch.cuda.empty_cache()

        return (image,)

class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        return (image.shape[2], image.shape[1], image.shape[0])

class ImageRemoveAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        if image.shape[3] == 4:
            image = image[..., :3]
        return (image,)

class ImagePreviewFromLatent(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE", ),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64})
            }, "optional": {
                "image": (["none"], {"image_upload": False}),
            }, "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, latent, vae, tile_size, prompt=None, extra_pnginfo=None, image=None, filename_prefix="ComfyUI"):
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        ui = None

        if image.startswith("clipspace"):
            image_path = folder_paths.get_annotated_filepath(image)
            if not os.path.exists(image_path):
                raise ValueError(f"Clipspace image does not exist anymore, select 'none' in the image field.")

            img = pillow(Image.open, image_path)
            img = pillow(ImageOps.exif_transpose, img)
            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            ui = {
                "filename": os.path.basename(image_path),
                "subfolder": os.path.dirname(image_path),
                "type": "temp",
            }
        else:
            if tile_size > 0:
                tile_size = max(tile_size, 320)
                image = vae.decode_tiled(latent["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, )
            else:
                image = vae.decode(latent["samples"])
            ui = self.save_images(image, filename_prefix, prompt, extra_pnginfo)

        out = {**ui, "result": (image, mask, image.shape[2], image.shape[1],)}
        return out

class NoiseFromImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strenght": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "noise_size": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "color_noise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_scale_diff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "saturation": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1 }),
            },
            "optional": {
                "noise_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image, noise_size, color_noise, mask_strength, mask_scale_diff, mask_contrast, noise_strenght, saturation, contrast, blur, noise_mask=None):
        torch.manual_seed(0)

        elastic_alpha = max(image.shape[1], image.shape[2])# * noise_size
        elastic_sigma = elastic_alpha / 400 * noise_size

        blur_size = int(6 * blur+1)
        if blur_size % 2 == 0:
            blur_size+= 1

        if noise_mask is None:
            noise_mask = image
        
        # increase contrast of the mask
        if mask_contrast != 1:
            noise_mask = T.ColorJitter(contrast=(mask_contrast,mask_contrast))(noise_mask.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])

        # Ensure noise mask is the same size as the image
        if noise_mask.shape[1:] != image.shape[1:]:
            noise_mask = F.interpolate(noise_mask.permute([0, 3, 1, 2]), size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)
            noise_mask = noise_mask.permute([0, 2, 3, 1])
        # Ensure we have the same number of masks and images
        if noise_mask.shape[0] > image.shape[0]:
            noise_mask = noise_mask[:image.shape[0]]
        else:
            noise_mask = torch.cat((noise_mask, noise_mask[-1:].repeat((image.shape[0]-noise_mask.shape[0], 1, 1, 1))), dim=0)

        # Convert mask to grayscale mask
        noise_mask = noise_mask.mean(dim=3).unsqueeze(-1)

        # add color noise
        imgs = image.clone().permute([0, 3, 1, 2])
        if color_noise > 0:
            color_noise = torch.normal(torch.zeros_like(imgs), std=color_noise)
            color_noise *= (imgs - imgs.min()) / (imgs.max() - imgs.min())

            imgs = imgs + color_noise
            imgs = imgs.clamp(0, 1)

        # create fine and coarse noise
        fine_noise = []
        for n in imgs:
            avg_color = n.mean(dim=[1,2])

            tmp_noise = T.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, fill=avg_color.tolist())(n)
            if blur > 0:
                tmp_noise = T.GaussianBlur(blur_size, blur)(tmp_noise)
            tmp_noise = T.ColorJitter(contrast=(contrast,contrast), saturation=(saturation,saturation))(tmp_noise)
            fine_noise.append(tmp_noise)

        imgs = None
        del imgs

        fine_noise = torch.stack(fine_noise, dim=0)
        fine_noise = fine_noise.permute([0, 2, 3, 1])
        #fine_noise = torch.stack(fine_noise, dim=0)
        #fine_noise = pb(fine_noise)
        mask_scale_diff = min(mask_scale_diff, 0.99)
        if mask_scale_diff > 0:
            coarse_noise = F.interpolate(fine_noise.permute([0, 3, 1, 2]), scale_factor=1-mask_scale_diff, mode='area')
            coarse_noise = F.interpolate(coarse_noise, size=(fine_noise.shape[1], fine_noise.shape[2]), mode='bilinear', align_corners=False)
            coarse_noise = coarse_noise.permute([0, 2, 3, 1])
        else:
            coarse_noise = fine_noise

        output = (1 - noise_mask) * coarse_noise + noise_mask * fine_noise

        if mask_strength < 1:
            noise_mask = noise_mask.pow(mask_strength)
            noise_mask = torch.nan_to_num(noise_mask).clamp(0, 1)
        output = noise_mask * output + (1 - noise_mask) * image

        # apply noise to image
        output = output * noise_strenght + image * (1 - noise_strenght)
        output = output.clamp(0, 1)

        return (output, )

IMAGE_CLASS_MAPPINGS = {
    # Image analysis
    "ImageEnhanceDifference+": ImageEnhanceDifference,

    # Image batch
    "ImageBatchMultiple+": ImageBatchMultiple,
    "ImageExpandBatch+": ImageExpandBatch,
    "ImageFromBatch+": ImageFromBatch,
    "ImageListToBatch+": ImageListToBatch,
    "ImageBatchToList+": ImageBatchToList,

    # Image manipulation
    "ImageCompositeFromMaskBatch+": ImageCompositeFromMaskBatch,
    "ImageComposite+": ImageComposite,
    "ImageCrop+": ImageCrop,
    "ImageFlip+": ImageFlip,
    "ImageRandomTransform+": ImageRandomTransform,
    "ImageRemoveAlpha+": ImageRemoveAlpha,
    "ImageRemoveBackground+": ImageRemoveBackground,
    "ImageResize+": ImageResize,
    "WhiteNoiseGenerator+":WhiteNoiseGenerator,
    "FaceAlign+":FaceAlign,
    "FaceAlignExternalDetector+":FaceAlignExternalDetector,
    "WarpTransformMask+":WarpTransformMask,
    "BlobNearROI+":BlobNearROI,
    "BlobWithinROI+":BlobWithinROI,
    "RefineNeckSegment+":RefineNeckSegment,
    "MaximalRectangleInsideBlob+":MaximalRectangleInsideBlob,
    "MediapipeImageSegmenter+":MediapipeImageSegmenter,
    "Image_Threshold_Mask+":Image_Threshold_Mask,
    "BBox_Padding+":BBox_Padding,
    "Combine_BBoxes+":Combine_BBoxes,
    "BBox_to_BBox_Parameters+":BBox_to_BBox_Parameters,
    "BBox_Parameters_to_BBox+":BBox_Parameters_to_BBox,
    "MaskCombine+":MaskCombine,
    "Find_BBox_of_Src_in_Dest+":Find_BBox_of_Src_in_Dest,
    "Load_POSE_KEYPOINT+":Load_POSE_KEYPOINT,
    "GetNeckSegment+":GetNeckSegment,
    "GetNeckSegment2+":GetNeckSegment2,
    "GetNeckSidelines+":GetNeckSidelines,
    "FaceNeckAlign+":FaceNeckAlign,
    "ImageSeamCarving+": ImageSeamCarving,
    "ImageTile+": ImageTile,
    "ImageUntile+": ImageUntile,
    "RemBGSession+": RemBGSession,
    "TransparentBGSession+": TransparentBGSession,

    # Image processing
    "ImageApplyLUT+": ImageApplyLUT,
    "ImageCASharpening+": ImageCAS,
    "ImageDesaturate+": ImageDesaturate,
    "PixelOEPixelize+": PixelOEPixelize,
    "ImagePosterize+": ImagePosterize,
    "ImageColorMatch+": ImageColorMatch,
    "ImageColorMatchAdobe+": ImageColorMatchAdobe,
    "ImageHistogramMatch+": ImageHistogramMatch,
    "ImageSmartSharpen+": ImageSmartSharpen,

    # Utilities
    "GetImageSize+": GetImageSize,
    "ImageToDevice+": ImageToDevice,
    "ImagePreviewFromLatent+": ImagePreviewFromLatent,
    "NoiseFromImage+": NoiseFromImage,
    #"ExtractKeyframes+": ExtractKeyframes,
}

IMAGE_NAME_MAPPINGS = {
    # Image analysis
    "ImageEnhanceDifference+": "🔧 Image Enhance Difference",

    # Image batch
    "ImageBatchMultiple+": "🔧 Images Batch Multiple",
    "ImageExpandBatch+": "🔧 Image Expand Batch",
    "ImageFromBatch+": "🔧 Image From Batch",
    "ImageListToBatch+": "🔧 Image List To Batch",
    "ImageBatchToList+": "🔧 Image Batch To List",

    # Image manipulation
    "ImageCompositeFromMaskBatch+": "🔧 Image Composite From Mask Batch",
    "ImageComposite+": "🔧 Image Composite",
    "ImageCrop+": "🔧 Image Crop",
    "ImageFlip+": "🔧 Image Flip",
    "ImageRandomTransform+": "🔧 Image Random Transform",
    "ImageRemoveAlpha+": "🔧 Image Remove Alpha",
    "ImageRemoveBackground+": "🔧 Image Remove Background",
    "ImageResize+": "🔧 Image Resize",    
    "WhiteNoiseGenerator+":"🔧 WhiteNoiseGenerator",
    "FaceAlign+": "🔧 Face Align",
    "FaceAlignExternalDetector+": "🔧 Face Align External Detector",
    "WarpTransformMask+":"🔧 Warp Transform Mask",
    "BlobNearROI+":"🔧 Blob Near ROI",
    "BlobWithinROI+":"🔧 Blob Within ROI",
    "RefineNeckSegment+": "🔧 Refine Neck Segment",
    "MaximalRectangleInsideBlob+":"🔧 Maximal Rectangle Box Inside Blob",
    "MediapipeImageSegmenter+":"🔧 Mediapipe Image Segmenter",
    "Image_Threshold_Mask+": "🔧 Image Threshold Mask",
    "BBox_Padding+": "🔧 BBox Padding",
    "Combine_BBoxes+": "🔧 Combine BBoxes",
    "BBox_to_BBox_Parameters+": "🔧 BBox to BBox Parameters",
    "BBox_Parameters_to_BBox+": "🔧 BBox Parameters to BBox",
    "MaskCombine+": "🔧 Combine Masks",
    "Find_BBox_of_Src_in_Dest+": "🔧 Find BBox of Src in Dest",
    "Load_POSE_KEYPOINT+":"🔧 Load POSE KEYPOINT",
    "GetNeckSegment+":"🔧 Get Neck Segment",
    "GetNeckSegment2+":"🔧 Get Neck Segment2",
    "GetNeckSidelines+":"🔧 Get Neck Sidelines",
    "FaceNeckAlign+":"🔧 Face Neck Align",
    "ImageSeamCarving+": "🔧 Image Seam Carving",
    "ImageTile+": "🔧 Image Tile",
    "ImageUntile+": "🔧 Image Untile",
    "RemBGSession+": "🔧 RemBG Session",
    "TransparentBGSession+": "🔧 InSPyReNet TransparentBG",

    # Image processing
    "ImageApplyLUT+": "🔧 Image Apply LUT",
    "ImageCASharpening+": "🔧 Image Contrast Adaptive Sharpening",
    "ImageDesaturate+": "🔧 Image Desaturate",
    "PixelOEPixelize+": "🔧 Pixelize",
    "ImagePosterize+": "🔧 Image Posterize",
    "ImageColorMatch+": "🔧 Image Color Match",
    "ImageColorMatchAdobe+": "🔧 Image Color Match Adobe",
    "ImageHistogramMatch+": "🔧 Image Histogram Match",
    "ImageSmartSharpen+": "🔧 Image Smart Sharpen",

    # Utilities
    "GetImageSize+": "🔧 Get Image Size",
    "ImageToDevice+": "🔧 Image To Device",
    "ImagePreviewFromLatent+": "🔧 Image Preview From Latent",
    "NoiseFromImage+": "🔧 Noise From Image",
}
