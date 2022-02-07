function data_output=shape_adaptive(y,gm_ICI)
% Anisotropic LPA-ICI Denoising Demo (demo_DenoisingGaussian)
%
% Alessandro Foi - Tampere University of Technology - 2003-2005
% -----------------------------------------------------------------------
%
% Performs the anisotropic LPA-ICI denoising on observations which are
% contaminated by additive Gaussian White noise.
%
%
% Observation model:
% z=y+n
% z : noisy observation
% y : true image (assumed as unknown)
% n : Gaussian white noise
%
% Other key variables:
% y_hat   : anisotropic "fused" estimate
% y_hat_Q : adaptive directional estimate
% h_opt_Q : adaptive directional scales
%
%

% clear all
close all
global scale_sp;
sharparam=-1;              % -1 zero order 0 first order (no sharpening) >0 sharpening
gammaICI=3.5;%3.5for hyperspectral image paviaU.etc;%1.05原始;             % ICI Gamma threshold
directional_resolution=8;  % number of directions
fusing=1;                  % fusing type   (1 classical fusing, 2 piecewise regular)
addnoise=0;                % add noise to observation
sigma_noise=0.0031;           % standard deviation of the noise 多少没有关系

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
% LPA KERNELS SIZES
%--------------------------------------------------------------------------
h1 = scale_sp;
% h1=[1 2 3 5 7 9];
h2=max(1,ceil(h1*tan(0.5*pi/directional_resolution)));  % row vectors h1 and h2 need to have the same lenght
%h2=ones(size(h1));
lenh=length(h1);

%--------------------------------------------------------------------------
% WINDOWS PARAMETERS
%--------------------------------------------------------------------------
sig_winds=[ones(size(h1)); ones(size(h2))];    % Gaussian parameter
beta=1;                     % Parameter of window 6

window_type=1;%112;%112;  % window=1 for uniform, window=2 for Gaussian
% window=6 for exponentions with beta
% window=8 for Interpolation
% window=11 for sectorial windows
% window=112 for sectorial unifrm windows

TYPE=10;            % TYPE IS A SYMMETRY OF THE WINDOW
% 00 SYMMETRIC
% 10 NONSYMMETRIC ON X1 and SYMMETRIC ON X2
% 11 NONSYMMETRIC ON X1,X2  (Quadrants)
%
% for rotated directional kernels the method that is used for rotation can be specified by adding
% a binary digit in front of these types, as follows:
%
% 10
% 11  ARE "STANDARD" USING NN (Nearest Neighb.) (you can think of these numbers with a 0 in front)
% 00
%
% 110
% 111  ARE EXACT SAMPLING OF THE EXACT ROTATED KERNEL
% 100
%
% 210
% 211  ARE WITH BILINEAR INTERP
% 200
%
% 310
% 311  ARE WITH BICUBIC INTERP (not reccomended)
% 300


%--------------------------------------------------------------------------
% MODELLING
%--------------------------------------------------------------------------

% y=im2double(imread('image_Cameraman256.png'));
% % % % %
% y=im2double(imread('PCA_img.tif'));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[mm nn bb]=size(y);
y=mat2gray(y);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%--------------------------------------------------------------------------
% MODELLING
%--------------------------------------------------------------------------

% y=im2double(imread('image_Cameraman256.png'));
init=0;%2055615866;
randn('seed', init);
%---------------------------------------------------------
% Images SIMULATION
%---------------------------------------------------------
[size_z_1,size_z_2]=size(y);

if addnoise==1
    n=repmat(sigma_noise,size_z_1,size_z_2).*randn(size(y));
    z = y + n;
else
    z = y;
end

sig_p = gm_ICI;
sigma=function_stdEst2D(z,1);   %%% estimates noise standard deviation
% gammaICI=max(0.8,2.4/(log(1+105*sigma)));%105 在85到200之间
gammaICI=max(0.8,2.4/(log(1+sig_p*sigma)));
% gammaICI = gm_ICI;
%---------------------------------------------------------
% Kernels construction
%---------------------------------------------------------
tic;
% calling kernel creation function
[kernels, kernels_higher_order]=function_CreateLPAKernels([0 0],h1,h2,TYPE,window_type,directional_resolution,sig_winds,beta);
[kernelsb, kernels_higher_orderb]=function_CreateLPAKernels([1 0],h1,h2,TYPE,window_type,directional_resolution,sig_winds,beta);


sigmaiter=repmat(sigma,size_z_1,size_z_2);
stop_condition=0;



clear yh h_opt_Q y_hat_Q var_opt_Q stdh
YICI_Final1=0; var_inv=0;         YICI_Final2=0;
CWW=0;
CWW2=0;

data_output=ones(mm,nn,directional_resolution);
%%%%% loops %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kernel_record=zeros(15,15,48);
kernel_record=cell(8,6);

for s2=1:lenh     % kernel size index
    gha=kernels_higher_order{1,s2,1}(:,:,1);   % gets single kernel from the cell array (ZERO ORDER)
    ghb=kernels_higher_orderb{1,s2,1}(:,:,1);  % gets single kernel from the cell array (FIRST ORDER)
    gh{s2}=(1+sharparam)*ghb-sharparam*gha; % combines kernels into "order-mixture" kernel
    gh{s2}=single(gh{s2}((end+1)/2,(end+1)/2:end));
end
Ker1toc=toc;
disp(['LPA kernels created in ',num2str(Ker1toc),' seconds.   Total time: ',num2str(Ker1toc),' seconds.'])
%---------------------------------------------------------
% Anisotropic LPA-ICI
%---------------------------------------------------------
h_opt_Q=function_AnisLPAICI8(single(z),gh,single(sigma),single(gammaICI));   %%% Anisotropic LPA-ICI scales for 8 directions
data_output = h_opt_Q;
end