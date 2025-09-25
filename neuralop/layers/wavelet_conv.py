""" Load required packages 

It requires the packages
-- "Pytorch Wavelets"
    see https://pytorch-wavelets.readthedocs.io/en/latest/readme.html
    ($ git clone https://github.com/fbcotter/pytorch_wavelets
     $ cd pytorch_wavelets
     $ pip install .)

-- "PyWavelets"
    https://pywavelets.readthedocs.io/en/latest/install.html
    ($ conda install pywavelets)

-- "Pytorch Wavelet Toolbox"
    see https://github.com/v0lta/PyTorch-Wavelet-Toolbox
    ($ pip install ptwt)
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

try:
    import ptwt, pywt
    from ptwt.conv_transform_3 import wavedec3, waverec3
    from pytorch_wavelets import DWT1D, IDWT1D
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    from pytorch_wavelets import DWT, IDWT 
except ImportError:
    print('Wavelet convolution requires <Pytorch Wavelets>, <PyWavelets>, <Pytorch Wavelet Toolbox> \n \
                    For Pytorch Wavelet Toolbox: $ pip install ptwt \n \
                    For PyWavelets: $ conda install pywavelets \n \
                    For Pytorch Wavelets: $ git clone https://github.com/fbcotter/pytorch_wavelets \n \
                                          $ cd pytorch_wavelets \n \
                                          $ pip install .')
    

""" Def: 1d Wavelet convolutional layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='symmetric'):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filter
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x]
                        kernel weights for Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if np.isscalar(size):
            self.size = size
        else:
            raise Exception("size: WaveConv1d accepts signal length in scalar only") 
        self.wavelet = wavelet 
        self.mode = mode
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)
        dummy_data = torch.randn( 1,1,self.size ) 
        mode_data, _ = self.dwt_(dummy_data)
        self.modes1 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x ) 
                  1D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """
        weights = weights.to(dtype=input.dtype)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x]
        
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x]
        """
        if x.shape[-1] > self.size:
            factor = int(np.log2(x.shape[-1] // self.size))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level+factor, mode=self.mode).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size:
            factor = int(np.log2(self.size // x.shape[-1]))
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level-factor, mode=self.mode).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = dwt(x)
            
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet  
            dwt = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = dwt(x)
            
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final low pass wavelet coefficients
        out_ft = self.mul1d(x_ft, self.weights1)
        # Multiply the final high pass wavelet coefficients
        out_coeff[-1] = self.mul1d(x_coeff[-1].clone(), self.weights2)

        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode=self.mode).to(device=x.device, dtype=x.dtype)
        x = idwt((out_ft, out_coeff)) 
        return x


""" Def: 2d Wavelet convolutional layer (discrete) """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric'):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Horizontal-Detailed wavelet coefficients
        self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Vertical-Detailed wavelet coefficients
        self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Diagonal-Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet       
        self.mode = mode
        dummy_data = torch.randn( 1,1,*self.size )        
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        weights = weights.to(dtype=input.dtype)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(device=x.device, dtype=x.dtype)
        x = idwt((out_ft, out_coeff))
        return x

    
""" Def: 2d Wavelet convolutional layer (slim continuous) """
class WaveConv2dCwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1, wavelet2):
        super(WaveConv2dCwt, self).__init__()

        """
        !! It is computationally expensive than the discrete "WaveConv2d" !!
        2D Wavelet layer. It does SCWT (Slim continuous wavelet transform),
                                linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet1     : string, Specifies the first level biorthogonal wavelet filters
        wavelet2     : string, Specifies the second level quarter shift filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights- 15r, 45r, 75r, 105r, 135r, 165r : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for REAL wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        self.weights- 15c, 45c, 75c, 105c, 135c, 165c : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for COMPLEX wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2        
        dummy_data = torch.randn( 1,1,*self.size ) 
        dwt_ = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights0 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights15r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights15c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights45c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights75c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights105c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights135c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165r = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))
        self.weights165c = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes21, self.modes22))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        weights = weights.to(dtype=input.dtype)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """      
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level+factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level-factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(device=x.device, dtype=x.dtype)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights0)
        # Multiply the final detailed wavelet coefficients        
        out_coeff[-1][:,:,0,:,:,0] = self.mul2d(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights15r)
        out_coeff[-1][:,:,0,:,:,1] = self.mul2d(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights15c)
        out_coeff[-1][:,:,1,:,:,0] = self.mul2d(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights45r)
        out_coeff[-1][:,:,1,:,:,1] = self.mul2d(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights45c)
        out_coeff[-1][:,:,2,:,:,0] = self.mul2d(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights75r)
        out_coeff[-1][:,:,2,:,:,1] = self.mul2d(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights75c)
        out_coeff[-1][:,:,3,:,:,0] = self.mul2d(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights105r)
        out_coeff[-1][:,:,3,:,:,1] = self.mul2d(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights105c)
        out_coeff[-1][:,:,4,:,:,0] = self.mul2d(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights135r)
        out_coeff[-1][:,:,4,:,:,1] = self.mul2d(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights135c)
        out_coeff[-1][:,:,5,:,:,0] = self.mul2d(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights165r)
        out_coeff[-1][:,:,5,:,:,1] = self.mul2d(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights165c)        

        # Reconstruct the signal
        icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(device=x.device, dtype=x.dtype)
        x = icwt((out_ft, out_coeff))
        return x
    
    
""" Def: 3d Wavelet convolutional layer """
class WaveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db4', mode='periodic'):
        super(WaveConv3d, self).__init__()

        """
        3D Wavelet layer. It does 3D DWT, linear transform, and Inverse dWT.    
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, Specifies the first level biorthogonal wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Approximate wavelet coefficients
        self.weights_ : tensor, shape-[in_channels * out_channels * x * y * z]
                        kernel weights for Detailed wavelet coefficients 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 3:
                raise Exception('size: WaveConv2dCwt accepts the size of 3D signal in list with 3 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 3D signal is list')
        self.wavelet = wavelet
        self.mode = mode
        dummy_data = torch.randn( [*self.size] ).unsqueeze(0)
        mode_data = wavedec3(dummy_data, pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
        self.modes1 = mode_data[0].shape[-3]
        self.modes2 = mode_data[0].shape[-2]
        self.modes3 = mode_data[0].shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    # Convolution
    def mul3d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(in_channel * x * y * z)
                  3D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y * z)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(out_channel * x * y * z)
        """
        weights = weights.to(dtype=input.dtype)
        return torch.einsum("ixyz,ioxyz->oxyz", input, weights)

    def forward(self, x):
        xr = torch.zeros_like(x)
        for i in range(x.shape[0]):
            
            if x.shape[-1] > self.size[-1]:
                factor = int(np.log2(x.shape[-1] // self.size[-1]))
                
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(x[i, ...], pywt.Wavelet(self.wavelet), level=self.level+factor, mode=self.mode)
            
            elif x.shape[-1] < self.size[-1]:
                factor = int(np.log2(self.size[-1] // x.shape[-1]))
                
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(x[i, ...], pywt.Wavelet(self.wavelet), level=self.level-factor, mode=self.mode)        
            else:
                # Compute single tree Discrete Wavelet coefficients using some wavelet
                x_coeff = wavedec3(x[i, ...], pywt.Wavelet(self.wavelet), level=self.level, mode=self.mode)
            
            # Multiply relevant Wavelet modes
            x_coeff[0] = self.mul3d(x_coeff[0].clone(), self.weights1)
            x_coeff[1]['aad'] = self.mul3d(x_coeff[1]['aad'].clone(), self.weights2)
            x_coeff[1]['ada'] = self.mul3d(x_coeff[1]['ada'].clone(), self.weights3)
            x_coeff[1]['add'] = self.mul3d(x_coeff[1]['add'].clone(), self.weights4)
            x_coeff[1]['daa'] = self.mul3d(x_coeff[1]['daa'].clone(), self.weights5)
            x_coeff[1]['dad'] = self.mul3d(x_coeff[1]['dad'].clone(), self.weights6)
            x_coeff[1]['dda'] = self.mul3d(x_coeff[1]['dda'].clone(), self.weights7)
            x_coeff[1]['ddd'] = self.mul3d(x_coeff[1]['ddd'].clone(), self.weights8)
            
            # Instantiate higher level coefficients as zeros
            for jj in range(2, self.level + 1):
                x_coeff[jj] = {key: torch.zeros_like(x_coeff[jj][key], device=x.device)
                                for key in x_coeff[jj].keys()}
            
            # Return to physical space       
            xr[i, ...] = waverec3(x_coeff, pywt.Wavelet(self.wavelet))
        return xr


class WaveletConv(nn.Module):
    """Adapter providing a unified WaveletConv interface for WNO blocks.

    This wrapper lazily instantiates the appropriate wavelet convolution
    (1D/2D/2D-CWT/3D) based on the incoming tensor dimensionality. It also
    handles optional padding so that the spatial resolution is compatible with
    the requested number of decomposition levels.
    """

    _SUPPORTED_PAD_MODES = {"constant", "reflect", "replicate", "circular"}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_levels: Sequence[int],
        wavelet_type: str = "haar",
        resolution_scaling_factor: Optional[Sequence[int]] = None,
        complex_data: bool = False,
        ensure_even_shapes: bool = False,
        pad_mode: str = "constant",
        adaptive_padding: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(n_levels, Iterable) and not isinstance(n_levels, (str, bytes)):
            self.n_levels = tuple(int(level) for level in n_levels)
        else:
            self.n_levels = (int(n_levels),)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_type = wavelet_type
        self.complex_data = complex_data
        self.ensure_even_shapes = ensure_even_shapes
        self.adaptive_padding = adaptive_padding
        self.n_dim = len(self.n_levels)

        if resolution_scaling_factor is None:
            self.resolution_scaling_factor: Optional[Tuple[int, ...]] = None
        else:
            if isinstance(resolution_scaling_factor, Iterable):
                factors = tuple(int(max(f, 1)) for f in resolution_scaling_factor)
                if len(factors) not in {1, self.n_dim}:
                    raise ValueError(
                        "resolution_scaling_factor must be length 1 or match n_dim"
                    )
                if len(factors) == 1 and self.n_dim > 1:
                    factors = tuple(factors[0] for _ in range(self.n_dim))
                self.resolution_scaling_factor = factors
            else:
                factor = int(max(resolution_scaling_factor, 1))
                self.resolution_scaling_factor = tuple(
                    factor for _ in range(self.n_dim)
                )

        pad_mode = pad_mode.lower()
        if pad_mode not in self._SUPPORTED_PAD_MODES:
            pad_mode = "constant"
        self.pad_mode = pad_mode

        self._conv_cache = nn.ModuleDict()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, output_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        if x.dim() < self.n_dim + 2:
            raise ValueError(
                f"Expected input with at least {self.n_dim + 2} dims, got {x.shape}"
            )

        input_shape = x.shape[2:]
        padded_x, pad_pairs = self._maybe_pad(x)
        target_shape = padded_x.shape[2:]

        conv_module = self._get_or_create_wavelet_module(target_shape)

        target_dtype = padded_x.real.dtype if padded_x.is_complex() else padded_x.dtype
        if target_dtype.is_floating_point:
            conv_module = conv_module.to(device=padded_x.device, dtype=target_dtype)
        else:
            conv_module = conv_module.to(device=padded_x.device)

        if padded_x.is_complex() or self.complex_data:
            real_part = conv_module(padded_x.real)
            imag_part = conv_module(padded_x.imag)
            out = torch.complex(real_part, imag_part)
        else:
            out = conv_module(padded_x)

        out = self._maybe_unpad(out, pad_pairs, input_shape)

        if output_shape is not None:
            output_shape = tuple(output_shape)
            if out.shape[2:] != output_shape:
                out = self._resize_to_shape(out, output_shape)

        return out

    # ------------------------------------------------------------------
    def _get_or_create_wavelet_module(self, target_shape: Tuple[int, ...]) -> nn.Module:
        key = self._conv_cache_key(target_shape)
        if key not in self._conv_cache:
            module = self._instantiate_wavelet_conv(target_shape)
            self._conv_cache[key] = module
        return self._conv_cache[key]

    def _conv_cache_key(self, target_shape: Tuple[int, ...]) -> str:
        shape_part = "x".join(map(str, target_shape))
        level = self._effective_level(target_shape)
        variant = "cwt" if self._use_cwt_variant else "dwt"
        return f"{shape_part}_L{level}_{variant}"

    @property
    def _use_cwt_variant(self) -> bool:
        return isinstance(self.wavelet_type, (tuple, list)) and len(self.wavelet_type) == 2

    def _instantiate_wavelet_conv(self, target_shape: Tuple[int, ...]) -> nn.Module:
        level = self._effective_level(target_shape)
        if level <= 0:
            level = 1

        if self.n_dim == 1:
            conv = WaveConv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                level=level,
                size=target_shape[0],
                wavelet=self._select_wavelet_name(),
                mode=self._wavelet_padding_mode(),
            )
        elif self.n_dim == 2:
            if self._use_cwt_variant:
                wavelet1, wavelet2 = self.wavelet_type  # type: ignore[misc]
                conv = WaveConv2dCwt(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    level=level,
                    size=list(target_shape),
                    wavelet1=wavelet1,
                    wavelet2=wavelet2,
                )
            else:
                conv = WaveConv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    level=level,
                    size=list(target_shape),
                    wavelet=self._select_wavelet_name(),
                    mode=self._wavelet_padding_mode(),
                )
        elif self.n_dim == 3:
            conv = WaveConv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                level=level,
                size=list(target_shape),
                wavelet=self._select_wavelet_name(),
                mode=self._wavelet_padding_mode(default_value="periodic"),
            )
        else:
            raise ValueError(f"Unsupported dimensionality: n_dim={self.n_dim}")

        return conv

    def _select_wavelet_name(self) -> str:
        if self._use_cwt_variant:
            return str(self.wavelet_type[0])  # type: ignore[index]
        return str(self.wavelet_type)

    def _wavelet_padding_mode(self, default_value: Optional[str] = None) -> str:
        wavelet_mode_map = {
            "constant": "symmetric",
            "reflect": "symmetric",
            "replicate": "symmetric",
            "circular": "periodization",
        }
        if default_value is None:
            default_value = "symmetric"
        return wavelet_mode_map.get(self.pad_mode, default_value)

    def _effective_level(self, target_shape: Tuple[int, ...]) -> int:
        max_levels: List[int] = []
        for dim_size in target_shape:
            if dim_size <= 1:
                max_levels.append(0)
            else:
                max_levels.append(int(math.floor(math.log2(dim_size))))
        requested = min(self.n_levels)
        return max(1, min(requested, min(max_levels)))

    def _maybe_pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if not (self.adaptive_padding or self.ensure_even_shapes):
            return x, [(0, 0) for _ in range(self.n_dim)]

        pad_pairs: List[Tuple[int, int]] = []
        spatial_shape = x.shape[2:]
        factors = self._resolve_scaling_factors()

        for idx, size in enumerate(spatial_shape):
            level = min(self.n_levels[idx % len(self.n_levels)], self._max_level_for(size))
            base_multiple = 2 ** max(level, 0)
            pad_total = 0

            if self.adaptive_padding and base_multiple > 1:
                remainder = size % base_multiple
                if remainder != 0:
                    pad_total = base_multiple - remainder

            if self.ensure_even_shapes and (size + pad_total) % 2 != 0:
                pad_total += 1

            # Scale padding if resolution scaling factor demands it
            scale = factors[idx] if factors is not None else 1
            pad_total *= max(scale, 1)

            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_pairs.append((pad_left, pad_right))

        if all(left == 0 and right == 0 for (left, right) in pad_pairs):
            return x, pad_pairs

        pad_sequence: List[int] = []
        for left, right in reversed(pad_pairs):
            pad_sequence.extend([left, right])

        padded = F.pad(x, pad_sequence, mode=self.pad_mode)
        return padded, pad_pairs

    def _maybe_unpad(
        self,
        x: torch.Tensor,
        pad_pairs: List[Tuple[int, int]],
        original_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        if all(left == 0 and right == 0 for (left, right) in pad_pairs):
            return x

        slices = [slice(None), slice(None)]
        for (left, right) in pad_pairs:
            start = left if left > 0 else None
            if right > 0:
                end = -right
                if end == 0:
                    end = None
            else:
                end = None
            slices.append(slice(start, end))
        result = x[tuple(slices)]

        if result.shape[2:] != original_shape:
            result = self._resize_to_shape(result, original_shape)
        return result

    def _resize_to_shape(
        self, x: torch.Tensor, target_shape: Sequence[int]
    ) -> torch.Tensor:
        if x.shape[2:] == tuple(target_shape):
            return x
        # For 1D, interpolate expects (N, C, L)
        if self.n_dim == 1:
            return F.interpolate(x, size=target_shape[0], mode="linear", align_corners=False)
        if self.n_dim == 2:
            return F.interpolate(x, size=tuple(target_shape), mode="bilinear", align_corners=False)
        return F.interpolate(x, size=tuple(target_shape), mode="trilinear", align_corners=False)

    def _resolve_scaling_factors(self) -> Optional[Tuple[int, ...]]:
        return self.resolution_scaling_factor

    @staticmethod
    def _max_level_for(size: int) -> int:
        if size <= 1:
            return 0
        return int(math.floor(math.log2(size)))
