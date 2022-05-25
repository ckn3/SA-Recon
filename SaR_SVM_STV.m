clc
close all
clear all

datasets = {'IndianPinesCorrected','PaviaU','IndianPinesCorrectedSar','PaviaUSar'};

% If you plan to use the SaR-SVM-STV algorithm:
% You need to reconstruct 'IndianPinesCorrected'/'PaviaU' data using "SaR_main.m" 
% and rename it as 'IndianPinesCorrectedSar.mat'/'PaviaUSar.mat'.

% The datasets should be formatted as:
% M,N,D : The number of rows, columns, and spectral bands in HSI.
% X     : (M*N)*D matrix.
% HSI   : M*N*D array.
% GT    : Ground Truth labels.

prompt = 'Which dataset? \n 1) IndianPines \n 2) PaviaU \n 3) Reconstructed IndianPines \n 4) Reconstructed PaviaU \n ';
DataSelected = input(prompt);
load(strcat(datasets{DataSelected},'.mat'))

% Set the denoising parameters in STV.
if mod(DataSelected,2)==1
    a1=0.2;a2=4;
else 
    a1=0.2;a2=1;
end
recon = X;
label_original = GT;
if min(label_original,[],'all') == 1
    label_original = label_original - 1;
end
clear X GT HSI prompt

%%

rng('default')
rng(1) %random seed
trial_num = 10; %number of trials, in the paper we used 10
num_train_per_class = 5;

img=pca_HSI(recon,0.999);
re_img=reshape(img,M,N,size(img,2));

Nonzero_index = find(label_original ~= 0);

no_classes=length(unique(label_original))-1;
overall_OA = []; overall_AA = []; overall_kappa = []; overall_CA = [];

[no_rows,no_lines, no_bands] = size(re_img);
% img=reshape(re_img,[no_rows*no_lines,no_bands]);
prediction_map = zeros(M,N,trial_num);
%% Select the number of training samples for each class

% modify the vector if you would like to change the number of training
% pixels
RandSampled_Num = repmat(num_train_per_class,1,no_classes);


Nonzero_map = zeros(no_rows,no_lines);
Nonzero_index =  find(label_original ~= 0);      %% find labeled pixels(colored pixels)
Nonzero_map(Nonzero_index)=1;                     %% labeled pixels = 1

%% Create the experimental set based on groundtruth of HSI
Train_Label = [];
Train_index = [];
for ii = 1: no_classes
    index_ii =  find(label_original == ii);
    class_ii = ones(length(index_ii),1)* ii;
    Train_Label = [Train_Label class_ii'];
    Train_index = [Train_index index_ii'];
end
%%% Train_Label and Train_index are row vectors
%%% Train_Label, the indices are the vectorized location
trainall = zeros(2,length(Train_index));
trainall(1,:) = Train_index;
trainall(2,:) = Train_Label;
%% Create the Training set with randomly sampling 3-D Dataset and its correponding index

train_SL = [];
indexes =[];

for trial_idx = 1: trial_num

    indexes =[];

    for i = 1: no_classes
        W_Class_Index = find(Train_Label == i); % based on trainall
        Random_num = randperm(length(W_Class_Index));
        Random_Index = W_Class_Index(Random_num);
        Tr_Index = Random_Index(1:RandSampled_Num(i));
        indexes = [indexes Tr_Index];
    end
    indexes = indexes';
    train_SL = trainall(:,indexes); % based on teainall
    train_samples = img(train_SL(1,:),:); % vectors of training datas
    train_labels= train_SL(2,:)';

    %% Create the Testing set with randomly sampling 3-D Dataset and its correponding index
    test_SL = trainall;
    test_SL(:,indexes) = [];
    test_samples = img(test_SL(1,:),:);
    test_labels = test_SL(2,:)';

    %% Generate spectral feature
    train_img=zeros(no_rows,no_lines);
    train_img(train_SL(1,:))=train_SL(2,:);

    %% Classification based on two feature image
    in_param.other.turning = false; 

    [class_label_svm, out_param, bestng] = classify_svm(re_img,train_img);
    in_param.other.n = bestng(1,1); in_param.other.gamma = bestng(1,2);

    [class_label, out_param] = classify_svm_prob(re_img,train_img,in_param);
    predict_label_prob = reshape(out_param.prob_estimates, no_rows, no_lines, no_classes);

    denoise_predict_tensor = zeros(no_rows, no_lines, no_bands);
    predict_label_prob = reshape(out_param.prob_estimates, no_rows, no_lines, no_classes);

    train_map  = zeros(no_rows,no_lines);
    train_map(train_SL(1,:)) = 1;

    for i = 1:size(train_SL,2)
        ix = ceil(train_SL(1,i)/no_rows);
        iy = train_SL(1,i)-(ix-1)*no_rows;
        temp = zeros(1,1,no_classes);
        temp(1,1,train_SL(2,i)) = 1;
        predict_label_prob(iy,ix,:) = temp;
    end      %% compute prob in stage 1


    denoise_predict_tensor = l2_l1_aniso_l2_less_ADMM_2dir(predict_label_prob,a1,a2,train_map==0,5);

    [~,class_label] = max(denoise_predict_tensor,[],3); %% classification rule in stage 2

    class_label = reshape(class_label,[],1);
    %% Calculate the error based on predict label and truth label
    [OA,kappa,AA,CA] = calcError(test_SL(2,:)-1,class_label(test_SL(1,:))'-1,[1:no_classes]);

    overall_OA = [overall_OA;OA]; overall_AA= [overall_AA;AA]; overall_kappa = [overall_kappa;kappa]; overall_CA = [overall_CA, CA];
    prediction_map(:,:,trial_idx) = reshape(class_label,no_rows,no_lines);
end
fprintf('OA: %1.4f, AA: %1.4f, kappa: %1.4f', mean(overall_OA), mean(overall_AA), mean(overall_kappa))
'Average accuracy of each class:'
mean(overall_CA,2)

