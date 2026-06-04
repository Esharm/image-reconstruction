%P = phantom(1024);
P = imread('saturn.png');
P = rgb2gray(P);
figure
imshow(P)
title('Original image')

% 2-D Fourier transform and center the DC component
%This is what we are 'measuring'
Y = fft2(P);
Yc = fftshift(Y);

sigma = 100000;
noise = sigma*(randn(size(Yc)) + 1i*randn(size(Yc)));
Ym = Y + noise;
Ymc = fftshift(Ym);


% Magnitude with log scaling for visibility
mag = log(1 + abs(Yc));
phase = angle(Yc);
mag2 = log(1 + abs(Ymc));
phase2 = angle(Ymc);


% Display with colormap and no axis ticks
figure
subplot(1,2,1)
imagesc(mag), colormap gray, axis image off
title('Log-Magnitude Spectrum')
subplot(1,2,2)
imagesc(mag2), colormap gray, axis image off
title('Log-Magnitude Spectrum (w/noise)')


figure
subplot(1,2,1)
imagesc(phase), colormap gray, axis image off
title('phase')
subplot(1,2,2)
imagesc(phase2), colormap gray, axis image off
title('phase w noise')

Z = ifft2(Ym);
figure
subplot(1,2,1)
imagesc(real(Z)), colormap gray, axis image off
title('Real(Z)')

subplot(1,2,2)
imagesc(imag(Z)), colormap gray, axis image off
title('Imag(Z)')

