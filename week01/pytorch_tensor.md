Tensors → specialized data structure that is very similar to arrays and matrices. In PyTorch, they are used to encode the inputs and outputs of a model, as well as its parameters.

| import torch import numpy as np |
| :---- |

Tensors can be initialized in various ways:

| Directly from data Data type is automatically inferred | data \= \[\[1, 2\], \[3, 4\]\] x\_data \= torch.tensor(data) |
| :---- | :---- |
| From a NumPy array | np\_array \= np.array(data) x\_np \= torch.from\_numpy(np\_array) |
| From another tensor Retains properties (x\_ones) Can be overridden (x\_rand) | x\_ones \= torch.ones\_like(x\_data) print(x\_ones) X\_rand \= torch.rand\_like(x\_data, dtype \= torch.float) print(x\_rand) “tensor(\[1, 1\], \[1, 1\])” “tensor(\[0.8461, 0.1849\], \[0.8453, 0.2967\])” |
| With random or constant values shape is a tuple of tensor dimensions. Determines the dimensionality of the output tensor. | Shape \= (2, 3,) rand\_tensor \= torch.rand(shape) print(rand\_tensor) ones\_tensor \= torch.ones(shape) print(ones\_tensor) zeros\_tensor \= torch.zeros(shape) print(zeros\_tensor) “tensor(\[\[0.6388, 0.6338, 0.1391\], \[0.5276, 0.7904, 0.1924\]\])” “tensor(\[1., 1., 1.\]\[1., 1., 1.\])” “tensor(\[0., 0., 0.\], \[0., 0., 0.\])” |

Tensor attributes → describe their shape, datatype, and the device on which they are stored.

| Input | tensor \= torch.rand(3, 4\) print(f”Shape of tensor: {tensor.shape}”) print(f”Datatype of tensor: {tensor.dtype}”) print(f”Device tensor is stored on: {tensor.device}”) |
| :---- | :---- |
| Output | Shape of tensor: torch.Size(\[3, 4\]) Datatype of tensor: torch.float32 Device tensor is stored on: cpu |

Tensor operations → over 100+ operations. Each of them can be run on the GPU.

| Input | \# We move our tensor to the GPU if available if torch.cude.is\_avaliable():     Tensor \= tensor.to(‘cuda’)     print(f”Device tensor is stored on: {tensor.device}”) |
| :---- | :---- |
| Output | Device tensor is stored on: cuda:0 |

Some other operations may include:

- Standard numpy-like indexing and slicing  
  - tensor \= torch.ones(4, 4)  
    tensor\[:,1\] \= 0  
    print(tensor)  
    tensor(\[\[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\]\])  
- Joining tensors  
  - t1 \= torch.cat(\[tensor, tensor, tensor\], dim\=1)  
    print(t1)  
    tensor(\[\[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.\],  
            \[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.\],  
            \[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.\],  
            \[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.\]\])  
- Multiplying tensors  
  - \# This computes the element-wise product  
    print(f"tensor.mul(tensor) \\n {tensor.mul(tensor)} \\n")  
    \# Alternative syntax:  
    print(f"tensor \* tensor \\n {tensor \* tensor}")  
      
    tensor.mul(tensor)  
     tensor(\[\[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\]\])  
      
    tensor \* tensor  
     tensor(\[\[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\]\])  
      
- print(f"tensor.matmul(tensor.T) \\n {tensor.matmul(tensor.T)} \\n")

  \# Alternative syntax:

  print(f"tensor @ tensor.T \\n {tensor @ tensor.T}")


  tensor.matmul(tensor.T)

   tensor(\[\[3., 3., 3., 3.\],

          \[3., 3., 3., 3.\],

          \[3., 3., 3., 3.\],

          \[3., 3., 3., 3.\]\])


  tensor @ tensor.T

   tensor(\[\[3., 3., 3., 3.\],

          \[3., 3., 3., 3.\],

          \[3., 3., 3., 3.\],

          \[3., 3., 3., 3.\]\])

- In-place operations  
  - Operations that have a \_ suffix are in-place.  
    - For example, x.copy\_(y), x.t\_(), will change x   
  - print(tensor, "\\n")  
    tensor.add\_(5)  
    print(tensor)  
      
    tensor(\[\[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\],  
            \[1., 0., 1., 1.\]\])  
      
    tensor(\[\[6., 5., 6., 6.\],  
            \[6., 5., 6., 6.\],  
            \[6., 5., 6., 6.\],  
            \[6., 5., 6., 6.\]\])  
- They have an immediate loss of history. Thus, their use is discouraged.

Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

| Input | t \= torch.ones(5) print(f”t: {t}”) n \= t.numpy() print(f”n: {n}”) |
| :---- | :---- |
| Output | t: tensor(\[1., 1., 1., 1., 1.\]) n: \[1. 1\. 1\. 1\. 1.\] |

A change in the tensor reflects in the NumPy array.

| Input | t.add\_(1) print(f”t: {t}”) print(f”n: {n}”) |
| :---- | :---- |
| Output | t: tensor(\[2., 2., 2., 2., 2.\]) n: \[2. 2\. 2\. 2\. 2.\] |

We can convert a NumPy array to tensor.

| Input | n \= np.ones(5) t \= torch.from\_numpy(n) np.add(n, 1, out \= n) print(f”t: {t}”) print(f”n: {n}”) |
| :---- | :---- |
| Output | t: tensor(\[2., 2., 2., 2., 2.\], dtype \= torch.float64) n: \[2. 2\. 2\. 2\. 2.\] |

