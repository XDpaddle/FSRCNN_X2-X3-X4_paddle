import paddle
from x2paddle import torch2paddle
import math
from paddle import nn
import paddle.nn.initializer 
import paddle.nn.functional 
from paddleseg.cvlibs import param_init



class FSRCNN(nn.Layer):

    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2D(num_channels, d,kernel_size=5, padding=5 // 2),
              nn.PReLU(d)
              )
        self.mid_part = [nn.Conv2D(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2D(s, s, kernel_size=3, padding=3 //2), nn.PReLU(s)])

        self.mid_part.extend([nn.Conv2D(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = torch2paddle.Conv2DTranspose(d, num_channels,kernel_size=9, stride=scale_factor,
                                                       padding=9 // 2,output_padding=scale_factor - 1)
        
        self._initialize_weights()

        
    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, paddle.nn.Conv2D):
                print(m)
                param_init.normal_init(m.weight.data,mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                param_init.constant_init(m.bias.data,value=0)

                # torch2paddle.normal_init_(m.weight, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                # torch2paddle.zeros_init_(m.bias)
        for m in self.mid_part:
            # print(m)
            if isinstance(m, paddle.nn.Conv2D):
                param_init.normal_init(m.weight.data,mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                param_init.constant_init(m.bias.data,value=0)

        #         torch2paddle.normal_init_(m.weight, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
        #         torch2paddle.zeros_init_(m.bias)
        # torch2paddle.normal_init_(self.last_part.weight, mean=0.0, std=0.001)
        # torch2paddle.zeros_init_(self.last_part.bias)
        # print(self.last_part.bias.data)

        param_init.normal_init(self.last_part.weight.data, mean=0.0, std=0.001)
        param_init.constant_init(self.last_part.bias.data,value=0)


    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)

        return x

