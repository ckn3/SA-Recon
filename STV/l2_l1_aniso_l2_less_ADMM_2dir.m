function denoise_result = l2_l1_aniso_l2_less_ADMM_2dir(F, beta1, beta2, fixed, mu)
%D = (D_l, D_u), D_l left, D_u up 
fixed0 = (fixed==0);
fixed1 = (fixed==1);
N1 = size(F,1); N2 = size(F,2); N3= size(F,3);N = N1*N2; % N1, N2: geometric  N3: third dimensional
denoise_result = zeros(N1,N2,N3);

DuTDu = zeros(N,1); DuTDu(1,1) = 2; DuTDu(2,1) = -1; DuTDu(N1,1) = -1;
DlTDl = zeros(N,1); DlTDl(1,1) = 2; DlTDl(N1+1,1) = -1; DlTDl(N1*(N2-1)+1,1) = -1;
DTD = DuTDu + DlTDl;

vec_SigmaD = fft2(reshape(full(DTD),N1,N2));
mu13pSigmaD = ((beta2 + mu) * vec_SigmaD + mu + 1).^(-1); 
for i = 1:N3     %% N3 == k
    f = F(:,:,i); %% f=vk
    
sigma1 = zeros(2*N1,N2); sigma2 = zeros(N1,N2); %% lamda1, lamda2
l = zeros(2*N1,N2);  w = zeros(N1,N2); %% s w
gn = ones(N1,N2); gnp1 = zeros(N1,N2);
g = f; %% u
w(fixed0) = f(fixed0);
iter = 1;
iter_error = norm(gn-gnp1,'fro')/norm(gn,'fro');
while iter_error > 1e-3 %% criteria 

   gn = g; 
   DT = mu * (l+sigma1); 
   temp_g_1 = DT(1:N1,:) - circshift(DT(1:N1,:),-1,2) + DT(N1+1:2*N1,:) - circshift(DT(N1+1:2*N1,:),-1,1);
   % -1,2: minus right -1,1: minus down
   temp_g_2 = mu * (w+sigma2) + temp_g_1 + f;
   g = ifft2((mu13pSigmaD .* fft2(temp_g_2)));

   Dg = [g-circshift(g,1,2); g-circshift(g,1,1)]; %
   R =  Dg - sigma1; % r
   l = sign(R) .* max(abs(R) - beta1/mu, 0); % s
   
   gmS3 = g - sigma2;
   w(fixed1) = gmS3(fixed1);
   
   sigma1 = sigma1 - Dg + l;
   sigma2 = sigma2 - g + w;

   gnp1 = g;
   iter = iter +1;
   iter_error = norm(gn-gnp1,'fro')/norm(gn,'fro');
%    fprintf('Iteration: %d, Error: %1.6f\n', iter, iter_error);
end
denoise_result(:,:,i) = w;
end
end