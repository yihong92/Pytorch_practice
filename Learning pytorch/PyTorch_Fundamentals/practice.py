import torch
print(torch.__version__)

# Scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim) # Check dimensions
print(scalar.item()) # # Get the Python number within a tensor (only works with one-element tensors)

# Vector
vector = torch.tensor([7, 7]) # 有一個小技巧可以快速查看有幾維 看一邊'['的數量
print(vector)
print(vector.ndim) # Check dimensions 
# Check shape of vector
print(vector.shape)

# Matrix
MATRIX = torch.tensor([[7, 8], [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

# Use torch.arange(), torch.range() is deprecated
# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which use the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work
print(float_16_tensor.dtype)

# Create a tensor
some_tensor = torch.rand(3, 4)
# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")

# Basic operations
# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
# Multiply it by 10
print(tensor * 10)
# Tensors don't change unless reassigned
print(tensor)
# Subtract and reassign
tensor = tensor - 10
print(tensor)
# Add and reassign
tensor = tensor + 10
print(tensor)
# Can also use torch functions
print(torch.multiply(tensor, 10))
# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

# Element-wise matrix multiplication
print(tensor * tensor) # 元素間相乘的寫法
# Matrix multiplication
print(torch.matmul(tensor, tensor)) # 矩陣相乘類似於內積
# Can also use the "@" symbol for matrix multiplication, though not recommended, torch.matmul method is faster

# Shapes need to be in the right way
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)
#torch.matmul(tensor_A, tensor_B) # (this will error, the inner dimensions must match)
# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)

# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)

# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shape: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape: {output.shape}")
# torch.mm is a shorcut for matmul
print(torch.mm(tensor_A, tensor_B.T))

# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input
                         out_features=6) # out_features = describes outer value
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# Create a tensor
x = torch.arange(0, 100, 10)
print(x)
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error

print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
# Can also use torch methods
print(torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x))

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)
# Create an int8 tensor
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8)

# Create a tensor
x = torch.arange(1., 8.)
print(x, x.shape)
# Add an extra dimension
x_reshape = x.reshape(1, 7)
print(x_reshape, x_reshape.shape)
# Change view (keeps same data as original but changes view)
# See more: https://stackoverflow.com/a/54507446/7900723
z = x.view(1, 7)
print(z, z.shape)
# Changing z changes x
z[:, 0] = 5
print(z, x)

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=1) # try changeing dim to dim=1 and see what happens
print(x_stacked)

print(f"Previous tensor: {x_reshape}")
print(f"Previous shape: {x_reshape.shape}")
# Remove extra dimension from x_reshaped
x_squeezed = x_reshape.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")
# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# Create tensor with specific shape
x_original = torch.rand(size = (224, 224, 3))
# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")

# Create a tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}")
print(f"Second square bracket: {x[0][0]}")
print(f"Third square bracket {x[0][0][0]}")
# Get all values of 0th dimension and the 0 index of 1st dimension
print(x[:, 0])
# Get all values of 0th & 1th dimensions but only index 1 of 2nd dimension
print(x[:, :, 1])
# Get all valuse of the 0 dimension but only the 1 index value of the 1st and 2nd dimension
print(x[:, 1, 1])
# Gen index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0, 0, :])

# Numpy array to tensor
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # numpy default float64, if want to convert use torch.from_numpy(array).type(torch.float32)
print(array, tensor)

# Change the array, keep the tensor
array = array + 1
print(array, tensor)

# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype = float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
print(tensor, numpy_tensor)
# Change the tensor, keep the array the same
tensor = tensor + 1
print(tensor, numpy_tensor)

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}")
print(f"Tensor B:\n{random_tensor_B}")
print(f"Does Tensor A equal Tensor B? (anywhere)")
print(random_tensor_A == random_tensor_B)

# Set the random seed
import random
RANDOM_SEED = 42 # try changing this to different values and see what happens to the numbers below
torch.manual_seed(seed =RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called
# Without this, tensor_D would be different to tensor_C
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}")
print(f"Tensor D:\n{random_tensor_D}")
print(f"Does Tensor C equal Tensor D? (anywhere)")
print(random_tensor_C == random_tensor_D)

# Check for GPU
print(torch.version.cuda)
print(torch.cuda.is_available())

# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Count number of devices
print(torch.cuda.device_count())

# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

# If tensor is on GPU, can't transform it to NumPy (this will error)
#tensor_on_gpu.numpy()

# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
print(tensor_on_gpu) # copy of the GPU tensor in CPU memory so the original tensor is still on GPU

