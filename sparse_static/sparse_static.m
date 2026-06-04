function [mask] = elliptical_sampling_mask(N, M, fraction, a_frac, b_frac, decay, seed)
% Elliptical variable-density k-space sampling mask (operator S).
%
% Inputs:
%   N, M      - k-space grid size (rows x cols), typically size of your image
%   fraction  - fraction of measurements to keep (e.g. 0.25 for 25%)
%   a_frac    - ellipse semi-axis, horizontal, as fraction of M/2 (default 0.9)
%   b_frac    - ellipse semi-axis, vertical,   as fraction of N/2 (default 0.9)
%   decay     - power-law decay exponent (default 2). Higher = more center-heavy.
%   seed      - RNG seed for reproducibility (default 42)
%
% Output:
%   mask      - logical N×M array. Apply as: y = fftshift(fft2(f)) .* mask;

if nargin < 4, a_frac = 0.9; end
if nargin < 5, b_frac = 0.9; end
if nargin < 6, decay  = 2;   end
if nargin < 7, seed   = 42;  end

rng(seed);

% Normalized frequency coordinates in [-1, 1]
u = linspace(-1, 1, M);
v = linspace(-1, 1, N);
[U, V] = meshgrid(u, v);

% Normalized elliptical radius (0 at center, 1 at ellipse boundary)
r = sqrt((U / a_frac).^2 + (V / b_frac).^2);

% Points outside the ellipse are never sampled
inside = r <= 1.0;

% Variable-density probability: p(r) = (1 - r^decay) inside, 0 outside
% This peaks at 1 for r=0 (DC) and falls to 0 at the ellipse boundary.
p = zeros(N, M);
p(inside) = (1 - r(inside).^decay);

% Normalize so expected fraction matches the requested value
% E[#sampled] = sum(p) after scaling = fraction * N*M
p = p * (fraction * N * M) / sum(p(:));
p = min(p, 1.0);   % cap at 1 (center region may saturate)

% Draw the mask
mask = rand(N, M) < p;
end


function [f_rec, history] = tikhonov_gd(y, mask, lambda, step_size, maxIter, tol)
% Gradient descent for: min_f 0.5*||mask.*fft2(f) - y||^2 + (lambda/2)*||f||^2
%
% Inputs:
%   y         - masked k-space measurements (N x M complex)
%   mask      - logical sampling mask
%   lambda    - regularization strength (try 1e-3 to 1e-1)
%   step_size - learning rate. Must be < 2/(1+lambda) for convergence.
%               Safe default: 0.9 / (1 + lambda)
%   maxIter   - max iterations (500-2000)
%   tol       - stop when ||gradient|| < tol (e.g. 1e-5)

if nargin < 4, step_size = 0.9 / (1 + lambda); end
if nargin < 5, maxIter   = 1000; end
if nargin < 6, tol       = 1e-5; end

f = real(ifft2(ifftshift(y)));   % zero-filled warm start

history.obj  = zeros(maxIter, 1);
history.grad = zeros(maxIter, 1);

for k = 1:maxIter

    % Forward: f -> k-space, apply mask, compute residual
    residual = fftshift(fft2(f)) .* mask - y;   % SAf - y

    % Gradient: A'S'(SAf - y) + lambda*f
    grad = real(ifft2(ifftshift(residual))) + lambda * f;

    % Gradient descent step
    f = f - step_size * grad;

    % Diagnostics
    obj = 0.5 * norm(residual(:))^2 + (lambda/2) * norm(f(:))^2;
    history.obj(k)  = obj;
    history.grad(k) = norm(grad(:));

    if norm(grad(:)) < tol
        fprintf('Converged at iteration %d\n', k);
        break
    end
end

f_rec = f;
history.obj  = history.obj(1:k);
history.grad = history.grad(1:k);
end


%main
N = 256; M = 256;
f = imread('cameraman.tif');
%size(f)
%f = rgb2gray(f);
f = double(f)/255;

figure
imshow(f)
title('Original image')

Af = fftshift(fft2(f));
noise_std = 0.01;
eps = noise_std * (randn(N,M) + 1i*randn(N,M));
mask = elliptical_sampling_mask(N, M, .1,0.4,0.6,2); 

Af_noisy = fftshift(fft2(f)) + eps;              % instrument measures noisy k-space
y = Af_noisy .* mask;

figure
imshow(y)


[f_rec, hist] = tikhonov_gd(y, mask, 1e-2);

figure
subplot(1,3,1); imagesc(f);                          colormap gray; axis image off; title('truth')
subplot(1,3,2); imagesc(real(ifft2(ifftshift(y))));       colormap gray; axis image off; title('zero-filled')
subplot(1,3,3); imagesc(f_rec);                           colormap gray; axis image off; title('Tikhonov GD')

figure; semilogy(hist.obj); xlabel('iteration'); ylabel('objective'); title('convergence')
