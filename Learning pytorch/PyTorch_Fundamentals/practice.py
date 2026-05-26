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

