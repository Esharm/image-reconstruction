# Sparse Reconstruction Basic

We start with the forward model.

$$y  = \mathbf{S}(\mathcal{A} f(x) + \varepsilon).$$

Where:

$y$ are the measurements, our data.  

$f$ is the 'true image'. In our case we know it, it's the image we start with. In the practical case, this is unknown, which is why we are trying to use imaging techniques to obtain it. 

$\mathcal{A}$ is the forward measurement operator. In the case of interferometric imaging, under the assumptions of the van Cittert-Zernike theorem, sampled visibility data is equivalent to sampling the 2D Fourier Transform of the brightness distribution of the true image. Therefore, for our purposes $\mathcal{A} = \mathcal{F}$, the Fourier Transform operator. Since we are discretizing the image, these are matrices and so $\mathcal{F}$ is the [Discrete Fourier Transform matrix](https://en.wikipedia.org/wiki/DFT_matrix).     

$\varepsilon$ is measurement noise from instruments. Implemented with matlab ` randn ` so normally distributed (Gaussian) noise. To be researched is what real instrument noise looks like.

$\mathbf{S}$ is the "sparseness" operator, which imposes some mask onto $\mathcal{A} f + \varepsilon$ to model a system where it is not possible to sample every point (or basis component) for reconstruction. Almost all VLBI datasets on faraway space objects are sparse. This is implemented with a decaying elliptical mask function ` elliptical_sampling_mask `. This is slightly realistic as measured points in VLBI will trace out an elliptical pattern on static objects due to the rotation of the Earth. However this was roughly implemented and not much time was put in to making this realistic to VLBI sampling. I maintained the fraction sampled at around 20% though this is also a stat we are unsure of that requires research.

## Inverse Problem
Given the forward model, we want to solve for $f$. In the ideal case, we could invert (ignoring noise) and be pretty happy about $f = A^{-1} S^{-1}y$. However, $S$ has no inverse, which can be intuited since it is fundamentally destroying information from the measured info. The DFT is invertible just fine, but specifically since the collected data is sparse, we have an underconstrained inverse problem. The simple reconstruction ignores sparseness entirely and takes $f = \mathcal{F}^{-1}(y)$. This creates some reconstruction of the image but we can do better.

The standard error function we want to minimize over choices of $f$ is $||\mathbf{S}\mathcal{A}f - y||^2$. Note that there are essentially infinite solutions to this inverse problem. We could have $f = f' + g$ where $f'$ is the true image and $g$ is any image such that $\mathbf{S}\mathcal{A}g = 0$ (which can be constructed). With just the standard error there is no way to choose between these solutions which is preferred. This is why we introduce a **regularization** function.

The regularization used here is called [Tikhonov Regularization](https://en.wikipedia.org/wiki/Ridge_regression). The objective function becomes

$$O(f) = ||\mathbf{S}\mathcal{A}f - y||^2 + \lambda||f||^2.$$

How does this solve the issue? The difference between $O(f)$ and just the error function itself is that $O(f)$ is globally convex, and therefore has a unique minimum. In this case, convexity of $O(f)$ means that the Hessian of $O$ is positive definite. We can compute

$$\nabla^2O(f) = \nabla(2\mathcal{A}^* \mathbf{S}^* (\mathbf{S}\mathcal{A}f - y) + 2\lambda f) = 2\mathcal{A}^* \mathbf{S}^* \mathbf{S}\mathcal{A} + 2\lambda \mathbf{I}.$$

Now, since $\lambda > 0$ (our regularization parameter), $\nabla^2O$ is positive for all nonzero inputs $f$ (note this is not true if we did not have the $2\lambda \mathbf{I}$ term). Therefore, there should now be a global minimum to this optimization problem, and we just have to find it. 

The process implemented is **gradient descent**. I'll briefly explain the relevant theory. We iteratively take

$$f_{k+1} = f_k - \alpha \nabla O(f_k)$$

Until $f_{k+1}-f_k$ is small enough that we have essentially reached the minimum. We aim to prove that there exists a threshold on the step size $\alpha$ that guarantees convergence in finite time.

As above, the gradient $\nabla O = 2\mathcal A^* \mathbf{S}^* (\mathbf{S}\mathcal{A}f - y) + 2\lambda f = 2\nabla^2O - 2\mathcal{A}^* \mathbf{S}^* y$. Let $f^* $ be the minimizer of $\nabla O(f)$, in which case we can write $\nabla O(f_k) = 2\nabla^2 O(f_k - f^*)$ (in other words define $f^* $ such that $\nabla^2 O f^* =  A^* \mathbf{S}^* y$).

We define the error term $e_k = f_k - f^*$. Then, 

$$f_{k+1} - f^* = f_k - f^* - \alpha 2\nabla^2O(f_k - f^*)$$

$$e_{k+1} = (I - 2\alpha \nabla^2O)e_k.$$

For the error term to converge, we need the constant $I - 2\alpha \nabla^2O$ to have magnitude less than 1. Luckily, this factor is entirely diagonal. We can see this since $2\mathcal{A}^* \mathbf{S}^* \mathbf{S}\mathcal{A} + 2\lambda \mathbf{I} = 2(\mathcal{A}^* \mathbf{S}\mathcal{A} + \lambda \mathbf I)$ which comes from the fact that $\mathbf{S}$ is a diagonal (real valued, since only 0 or 1) projection matrix. Then, we are essentially transforming $\mathbf{S}$ into the Fourier basis when applying $\mathcal{A}^* \mathbf{S}\mathcal{A}$ (since the FT is the change of base matrix to the Fourier Plane), and it is still diagonal. 

The error term evolves in a geometric series fashion, dependent on the eigenvalues of $I - 2\alpha \nabla^2O$. Since it is diagonal, we can easily see it is $1 - 2\alpha(\delta + \lambda)$, where $\delta \in \{0,1\}$ based on the entry in the diagonal of $\mathbf{S}$. Therefore, to ensure convergence, we want $|1 - 2\alpha(\delta + \lambda)| < 1$, which can be rearranged to $\alpha < \frac{1}{\delta + \lambda}$. The strictest bound is thus $\alpha < \frac{1}{1+\lambda}$. Abiding by this, we can be sure that the numerical algorithm will terminate. The time complexity is roughly $\mathcal{O}(N^2\log N)$ for an $N \times N$ image, dominated by the runtime of the FFT on a 2D image. We also have dependence on the regularization factor $\lambda$ and the tolerance $\epsilon$, and more explicitly we see a runtime along the lines of $\mathcal{O}((\frac{1+\lambda}{\lambda})\log(\frac{1}{\epsilon})N^2\log N)$, where we can see smaller $\epsilon$ and $\lambda \to 0$ will start to increase runtime significantly.
