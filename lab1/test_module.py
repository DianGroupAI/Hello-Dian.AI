import numpy as np
import torch
import torch.nn.functional as F

from lab1.nn.modules import *

# class dropout(torch.nn.Model):
#     def __init__(self, p=0.5):
#         self.p = p
#         self.mask = None
#
#     def forward(self, x):
#         return self.mask * x / self.p





class testFather():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.types = ['MaxPool', 'AvgPool', 'Conv2D', 'BN', 'FC', 'Dropout']


        assert self.kwargs.get('type', None) is not None, "未指定测试模块类型，请添加'type'关键字"
        self.module_type = self.kwargs.get('type')
        assert self.module_type in self.types, "指定模块无效"

        # 判断类型选择不同初始化方式
        self.input_numpy = None
        if self.module_type == 'MaxPool' or self.module_type == 'AvgPool' or self.module_type == 'Dropout':
            self.input_numpy = np.random.rand(2, 2, 4, 4)
            self.kernel_size = 2
            self.stride = 2
            self.padding = 0
        elif self.module_type == 'Conv2D':
            self.input_numpy = np.random.rand(20, 5, 8, 8)
            self.kernel_size = 3
            self.out_channels = 10
            self.stride = 1
            self.padding = 0
            self.bias = False
        else:
            self.input_numpy = np.random.rand(5, 8)
        self.input_tensor = torch.Tensor(self.input_numpy)
        self.input_tensor.requires_grad = True

        # 根据不同的网络选择方式，预留不同的打印信息的方式
        self.w = None
        self.model_tensor = None
        self.model_numpy = None

    def forward(self):
        self.output_tensor = self.model_tensor(self.input_tensor)
        self.output_numpy = self.model_numpy(self.input_numpy)
        if self.module_type == 'BN':
            self.output_tensor.backward(self.output_tensor_delta)
        else:
            self.output_tensor_delta = self.output_tensor.sum()
            self.output_numpy_delta = np.ones_like(self.output_numpy)
            self.output_tensor_delta.backward()

        self.check()

    def check(self):
        print("{:*^40}".format('Forward Check:'))
        self.isClose(self.output_numpy, self.output_tensor)
        if self.module_type != 'Dropout':
            print("{:*^40}".format('Backward Check:'))
            self.isClose(self.model_numpy.backward(self.output_numpy_delta), self.input_tensor.grad)


    def printInfo(self):
        print("Input shape is: ===============>>>>>\t", self.input_numpy.shape)
        print("\033[1;34;43mThe input matrix is:\033[0m")
        print(self.input_numpy)
        if self.model_numpy == "FC":
            print("The W matrix shape is: ===============>>>>>\t", self.w.shape)
            print("The W matrix is:")
            print(self.w)
        print("{:*^60}".format(''))
        print("{:*^71}".format('\033[0;31m' + self.module_type + ' Layer Test\033[0m'))
        print("{:*^60}".format(''))

        print("1. Using your own code.....\n")
        print(self.output_numpy)
        print("{:*^50}".format("The grad is as follows:"))
        print(self.model_numpy.backward(self.output_numpy_delta))
        print()

        print("2. Here is the official code.....\n")
        print(self.output_tensor)
        print("{:*^50}".format("The grad is as follows:"))
        print(self.input_tensor.grad)

    def isClose(self, x_numpy, x_torch):
        x_torch = x_torch.data.numpy()
        diff = np.abs(np.mean(x_torch - x_numpy))
        print("The error is ", diff, " on average")
        if diff < 1e-5:
            print("Your answer is right!\n")
        else:
            print("Your answer is wrong!\n")







class testModule(testFather):
    def __init__(self, **kwargs):
        super(testModule, self).__init__(**kwargs)
        # 偏置矩阵初始化
        if self.module_type == "FC":
            self.FCInit()
        elif self.module_type == "BN":
            self.BNInit()
        elif self.module_type == "Conv2D":
            self.Conv2dInit()
        elif self.module_type == "MaxPool":
            self.MaxPoolInit()
        elif self.module_type == "AvgPool":
            self.AvgPoolInit()
        elif self.module_type == 'Dropout':
            self.DropoutInit()

    def FCInit(self):
        in_length = self.input_numpy.shape[1]
        out_length = 4
        # self.w = np.random.normal(loc=0.0, scale=0.1, size=(out_length, in_length + 1))
        # self.w_tensor = torch.Tensor(self.w)
        # self.model_tensor.bias.data = self.w_tensor[:, 0]
        # self.model_tensor.weight.data = self.w_tensor[:, 1:]
        # 初始化numpy层
        self.model_numpy = Linear(in_length=in_length, out_length=out_length)

        # 初始化torch层
        self.model_tensor = torch.nn.Linear(in_features=in_length, out_features=out_length, bias=True)

        # 取得numpy模型的权重和偏置参数
        self.w = self.model_numpy.w.T
        self.w_tensor = torch.tensor(self.w)
        self.model_tensor.bias.data = self.w_tensor[:, 0]
        self.model_tensor.weight.data = self.w_tensor[:, 1:]

    def BNInit(self):
        self.output_numpy_delta = np.random.rand(self.input_numpy.shape[0], self.input_numpy.shape[1])
        self.output_tensor_delta = torch.tensor(self.output_numpy_delta, requires_grad=True)
        self.model_tensor = torch.nn.BatchNorm1d(num_features=self.input_numpy.shape[1], eps=1e-5, momentum=0.9, affine=True)
        self.model_numpy = BatchNorm1d(length=self.input_numpy.shape[1])

    def Conv2dInit(self):
        self.output_numpy_delta = np.random.rand(self.input_numpy.shape[0], self.out_channels,
                                                 (self.input_numpy.shape[
                                                      2] + 2 * self.padding - self.kernel_size) // self.stride + 1,
                                                 (self.input_numpy.shape[
                                                      3] + 2 * self.padding - self.kernel_size) // self.stride + 1
                                                 )
        self.output_tensor_delta = torch.tensor(self.output_numpy_delta, requires_grad=True)

        self.model_numpy = Conv2d(in_channels=self.input_numpy.shape[1], out_channels=self.out_channels,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)

        bias = self.model_numpy.bias
        kernel = torch.tensor(self.model_numpy.kernel, requires_grad=True)
        kernel = kernel.type(torch.FloatTensor)
        self.model_tensor = torch.nn.Conv2d(self.input_tensor.shape[1], self.out_channels, kernel_size=self.kernel_size,
                                            stride=self.stride, padding=self.padding, bias=self.bias)
        self.model_tensor.weight.data = kernel
        if self.bias:
            self.model_tensor.bias.data = bias
        print()

    def MaxPoolInit(self):
        self.output_numpy_delta = np.random.rand(self.input_numpy.shape[0], self.input_numpy.shape[1],
                                                 (self.input_numpy.shape[
                                                      2] + 2 * self.padding - self.kernel_size) // self.stride + 1,
                                                 (self.input_numpy.shape[
                                                      3] + 2 * self.padding - self.kernel_size) // self.stride + 1
                                                 )
        self.output_tensor_delta = torch.tensor(self.output_numpy_delta, requires_grad=True)
        self.model_tensor = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.model_numpy = MaxPool(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def AvgPoolInit(self):
        self.output_numpy_delta = np.random.rand(self.input_numpy.shape[0], self.input_numpy.shape[1],
                                                 (self.input_numpy.shape[
                                                      2] + 2 * self.padding - self.kernel_size) // self.stride + 1,
                                                 (self.input_numpy.shape[
                                                      3] + 2 * self.padding - self.kernel_size) // self.stride + 1
                                                 )
        self.output_tensor_delta = torch.tensor(self.output_numpy_delta, requires_grad=True)
        self.model_tensor = torch.nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.model_numpy = AvgPool(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def DropoutInit(self):
        self.output_numpy_delta = np.random.rand(*self.input_numpy.shape)
        self.output_tensor_delta = torch.tensor(self.output_numpy_delta, requires_grad=True)
        # 初始化numpy模型
        self.model_numpy = Dropout(p=0.5)
        # 初始化torch模型
        self.model_tensor = torch.nn.Dropout(p=0.5)

        print()




    def __call__(self):
        self.forward()


t = testModule(type='Dropout')
t()