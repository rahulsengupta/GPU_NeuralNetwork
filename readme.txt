This project is an implementation GPU-accelerated neural network-based parity checker.A serial C code was modified to use CUDA C to perform vector multiplication in parallel.After that, a free software was used to implement the parity-bit checker.The project details are explained in the accompanying PowerPoint presentation file.
Simplest and the most straight-forward way to use the C code is to 1st install CUDA toolkit 5.5, then Visual Studio (2012 or 2010). Create a new  CUDA 5.5 project and copy-paste the contents into Kernel.cu file.
Build and run the code.
