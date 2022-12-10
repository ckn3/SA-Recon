% clc
% clear
% warning off;

% Load datasets

prompt = 'Which dataset? \n 1) Indian Pines \n 2) Pavia University \n 3) Salinas \n ';
DataSelected = input(prompt);

if DataSelected == 1
    load('Data\IndianPines_data.mat')
    load('Data\IndianPines_label.mat')
    im_3d = indian_pines_corrected;
    label_original = IndianPines_label;
elseif DataSelected == 2
    load('Data\PaviaU_data.mat')
    load('Data\PaviaU_label.mat')
    im_3d = paviaU;
    label_original = paviaU_gt;
elseif DataSelected == 3
    
else
end
    
global scale_sp;
scale_sp = [1 2 3 5 7 9];

% Parameters
num_perclass = [10 10 10 10 10 10 10 10 10];
trial_num =10;

row_max = size(im_gt,1);
col_max = size(im_gt,2);
im_gt_1d = reshape(im_gt,1,row_max*col_max);
ind_map = reshape(1:length(im_gt_1d),[row_max,col_max]);

[pca1, PAR] = PCA_img(im_3d,1);

index = ind_map(:);

gm = 385;
len = shape_adaptive(pca1,gm);

len = reshape(len,row_max*col_max,[])';
len = scale_sp(len);
recon=[];

for i=1:length(index)
    row = mod(index(i),row_max);
    if row == 0
        row = row_max;
    end
    col = ceil(index(i)/row_max);
    lens = len(:,index(i));
   
    pixs_xy=PtsSaR(lens,row,col);

    
    pixs_num = size(pixs_xy,1);
    X_ind = [];
    for j = 1:1:size(pixs_xy,1)
        temp = ind_map(pixs_xy(j,1),pixs_xy(j,2));
        X_ind = [X_ind temp];
    end

    Sneigh=[];
    
    for j=1:pixs_num
        Sneigh=[Sneigh;im_3d(pixs_xy(j,1),pixs_xy(j,2),:)];
        if pixs_xy(j,1)==row && pixs_xy(j,2)==col
            k=j;
        end
    end
    Sneigh=reshape(Sneigh,size(Sneigh,1)*size(Sneigh,2),size(Sneigh,3));

    vec = weightVec(Sneigh,k)';
    R=vec*Sneigh;
    recon=[recon;R];

end

save('reconstructed.mat','recon')

function vec = weightVec(Sneigh,k)
% Input
% 	Sneigh: n*B matrix, n is the number of pixels in the center pixel's SaR
%   k     : the k-th row of the Sneigh is the spectrum of the center pixel
% Output
%   vec   : weighted vector, representing the weight of the n pixels

pcr = corrcoef(Sneigh');
vec = pcr(:,k);
vec(vec<0)=0;

vec = vec./sum(vec);


end
