function [outdata, out_param, bestng] = classify_svm(varargin)
%CLASSIFYSVM Classify with libSVM an image
%
%		[outdata, out_param] = classify_svm(img, train, opt)


% INPUT
%   img    Multispectral image to be classified.
%   train  Training set image (zero is unclassified and will not be
%           considered).
%   opt    input parameters. Structure with each field correspondent to a
%           libsvm parameter
%           Below the availabel fields. The letters in the brackets corresponds to the flags used in libsvm:
%             "svm_type":	(-s) set type of SVM (default 0)
%                   0 -- C-SVC
%                   1 -- nu-SVC
%     	            2 -- one-class SVM
%     	            3 -- epsilon-SVR
%     	            4 -- nu-SVR
%             "kernel_type": (-t) set type of kernel function (default 2)
%                   0 -- linear: u'*v
%                   1 -- polynomial: (gamma*u'*v + coef0)^degree
%                   2 -- radial basis function: exp(-gamma*|u-v|^2)
%                   3 -- sigmoid: tanh(gamma*u'*v + coef0)
%                   4 -- precomputed kernel (kernel values in training_instance_matrix)
%          3   "kernel_degree": (-d) set degree in kernel function (default 3)
%             "gamma": set gamma in kernel function (default 1/k, k=number of features)
%           1  "coef0": (-r) set coef0 in kernel function (default 0)     1
%             "cost": (-c) set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%             "nu": (-n) parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%             "epsilon_regr": (-p) set the epsilon in loss function of epsilon-SVR (default 0.1)
%             "chache": (-m) set cache memory size in MB (default 100)
%             "epsilon": (-e) set tolerance of termination criterion (default 0.001)
%             "shrinking": (-h) whether to use the shrinking heuristics, 0 or 1 (default 1)
%        1     "probability_estimates": (-b) whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%             "weight": (-wi) set the parameter C of class i to weight*C, for C-SVC (default 1)
%             "nfold": (-v) n-fold cross validation mode
%             "quite": (-q) quiet mode (no outputs)
%           For setting other default values, modify generateLibSVMcmd.
%
% OUTPUT
%   outdata    Classified image
%   out_param  structure reports the values of the parameters
%
% DESCRIPTION
% This routine classify an image according to the training set provided
% with libsvm. By default, the data are scaled and normalized to have unit
% variance and zero mean for each band of the image. If the parameters
% defining the model of the svm (e.g., the cost C and gamma) are not
% provided, the function call the routin MODSEL and which optimizes the
% parameters. Once the model is trained the image is classified and is
% returned as output.
%
% SEE ALSO
% EPSSVM, MODSEL, GETDEFAULTPARAM_LIBSVM, GENERATELIBSVMCMD, GETPATTERNS

% Mauro Dalla Mura
% Remote Sensing Laboratory
% Dept. of Information Engineering and Computer Science
% University of Trento
% E-mail: dallamura@disi.unitn.it
% Web page: http://www.disi.unitn.it/rslab

% Parse inputs
if nargin == 2
    data_set = varargin{1};
    train = varargin{2};
    in_param = struct;
%    in_param.kernel_type = 2;   % default RBF
elseif nargin == 3
    data_set = varargin{1};
    train = varargin{2};
    in_param = varargin{3};
end

% data_set = img;
% train = train_img;

% Default Parameters - Scaling the data
scaling_range = true;       % Scale each feature of the data in the range [-1,1]
scaling_std = true;         % Scale each feature of the data in order to have std=1

% Read in_param
% if (isfield(in_param, 'scaling_range'))
%     scaling_range = in_param.scaling_range;       % scaling_range
% else
%     in_param.scaling_range = scaling_range;
% end
% if (isfield(in_param, 'scaling_std'))
%     scaling_std = in_param.scaling_std;           % scaling_range
% else
%     in_param.scaling_std = scaling_std;
% end
% ------------------------

[nrows ncols nfeats] = size(data_set);
for i = 1: nfeats
    temp_y = reshape(data_set(:,:,i),1,[]);
    mm = mean(temp_y, 2);
    sdd = std(temp_y, 1, 2);
    data_set(:,:,i) = (data_set(:,:,i) - mm) / (sdd);
end
Ximg = double(reshape(data_set, nrows*ncols, nfeats));

% Transform training set in a format compliant to RF
% [X, L] = getPatterns(data_set, train);
% nclasses = length(unique(L));

% % % % [X,row_factor] = removeconstantrows(X);   % Remove redundant features
% % % % Ximg = Ximg(:,row_factor.keep); % Remove redundant features

% ========= Preprocessing =========
% Scale each feature of the data in the range [-1,1]
% if (scaling_range)
%     [X,scale_factor] = mapminmax(X);   % Perform the scaling on the training set
%     nfold = 10;
%     nelem = round(size(Ximg,1)/nfold);
%     for i=1:nfold-1                     % Apply the same scaling on the whole set
%         Ximg((i-1)*nelem+1:i*nelem,:) = (mapminmax('apply',Ximg((i-1)*nelem+1:i*nelem,:)',scale_factor))';
%     end
%     Ximg((nfold-1)*nelem+1:end,:) = (mapminmax('apply',Ximg((nfold-1)*nelem+1:end,:)',scale_factor))';
% end
% % Scale each feature in order to have std=1
% if (scaling_std)
%     [X,scale_factor] = mapstd(X);  % Perform the scaling on the training set
%     nfold = 5;
%     nelem = round(size(Ximg,1)/nfold);
%     for i=1:nfold-1                 % Apply the same scaling on the whole set
%         Ximg((i-1)*nelem+1:i*nelem,:) = (mapstd('apply',Ximg((i-1)*nelem+1:i*nelem,:)',scale_factor))';
%     end
%     Ximg((nfold-1)*nelem+1:end,:) = (mapstd('apply',Ximg((nfold-1)*nelem+1:end,:)',scale_factor))';
% end

tic
% Train the model
% for i = 1: nfeats
%     temp_y = reshape(data_set(:,:,i),1,[]);
%     mm = mean(temp_y, 2);
%     sdd = std(temp_y, 1, 2);
%     data_set(:,:,i) = (data_set(:,:,i) - mm) / (2*sdd);
% end

[X, L] = getPatterns(data_set, train);  %% 16*304

% if ~(in_param.other.turning)
bestcv = 0;
for n = 0.005:0.005:0.1
    for g = 0.005:0.005:0.1
% for n = 0.1:0.1:0.5
%     for g = 0.1:0.1:0.5        
        %             cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cmd = ['-q -s 1 -t 2 -n' ' ' num2str(n) ' ' '-g' ' ' num2str(g) ' ' '-v 5']; % '-b 1'
        %             cmd = ['-q -s 0 -t 2 -c' ' ' num2str(in_param.other.CCC) ' ' '-g' ' ' num2str(in_param.other.gamma)]; % '-b 1'
        
        
        %     cmd = ['-s 0 -t 2 -g' ' ' num2str(in_param.other.gamma) ' ' '-v 5' ]; % '-b 1'
        model= svmtrain(double(L)',double(X)',cmd);
        if (model >= bestcv)
            bestcv = model;
            bestn = n;
            bestg = g;
%             bestg = 2^log2g;
        end
        fprintf('%g %g %g (best n=%g, g=%g, rate=%g)\n', n, g, model, bestn, bestg, bestcv);
    end
end
bestng = [];
bestng = [bestn bestg];
% save best_para;
% else
%     accur_old = 0;
%     if (scaling_range)
%         [X,scale_factor] = mapminmax(X);   % Perform the scaling on the training set
%         nfold = 10;
%         nelem = round(size(Ximg,1)/nfold);
%         for i=1:nfold-1                     % Apply the same scaling on the whole set
%             Ximg((i-1)*nelem+1:i*nelem,:) = (mapminmax('apply',Ximg((i-1)*nelem+1:i*nelem,:)',scale_factor))';
%         end
%         Ximg((nfold-1)*nelem+1:end,:) = (mapminmax('apply',Ximg((nfold-1)*nelem+1:end,:)',scale_factor))';
%     end
%     % Scale each feature in order to have std=1
%     if (scaling_std)
%         [X,scale_factor] = mapstd(X);  % Perform the scaling on the training set
%         nfold = 5;
%         nelem = round(size(Ximg,1)/nfold);
%         for i=1:nfold-1                 % Apply the same scaling on the whole set
%             Ximg((i-1)*nelem+1:i*nelem,:) = (mapstd('apply',Ximg((i-1)*nelem+1:i*nelem,:)',scale_factor))';
%         end
%         Ximg((nfold-1)*nelem+1:end,:) = (mapstd('apply',Ximg((nfold-1)*nelem+1:end,:)',scale_factor))';
%     end
%     for ic = in_param.other.CCCiter
%         for ig = in_param.other.gammaiter
%             cmd = ['-q -s 1 -t 2 -c' ' ' num2str(ic) ' ' '-g' ' ' num2str(ig) ' ' '-v 5'];
% %             cmd = ['-s 0 -t 2 -g' ' ' num2str(ig) ' ' '-v 5'];
%             accur = svmtrain(double(L)',double(X)',cmd);
%             if accur > accur_old 
%                 opt_cmd = cmd(1:end-5);
%                 accur_old = accur;
%             end
%         end
%     end
%     out_param.other.opt_cmd = opt_cmd;
%     model = svmtrain(double(L)',double(X)',opt_cmd);
% end
% cmd = '-s 0 -t 2 -g 0.01 -c 10 -v 3';
% cmd
% [model, out_param] = epsSVM(double(X)', double(L)', in_param);
% % % out_param.time_tr = toc;
% % % out_param.nfeats = length(row_factor.keep);

% Classify the whole data
%Ximg = double(reshape(data_set, nrows*ncols, nfeats));

% 
% nfold = 5;
% nelem = round(size(Ximg,1)/nfold);
% 
% for i=1:nfold-1
%     Ximg((i-1)*nelem+1:i*nelem,:) = (mapminmax('apply',Ximg((i-1)*nelem+1:i*nelem,:)',scale_factor))';
% end
% Ximg((nfold-1)*nelem+1:end,:) = (mapminmax('apply',Ximg((nfold-1)*nelem+1:end,:)',scale_factor))';

%Ximg = Ximg*scale_factor;
% cmd = generateLibSVMcmd(out_param, 'predict');      % this is needed when the training is done with -b enabled (probabilities estimated)

opt_cmd = ['-q -s 1 -t 2 -n' ' ' num2str(bestn) ' ' '-g' ' ' num2str(bestg)];
model = svmtrain(double(L)',double(X)',opt_cmd);



% [predicted_labels, out_param.accuracy, out_param.prob_estimates] = svmpredict(ones(nrows*ncols, 1), Ximg, model, '-b 1');
[predicted_labels, out_param.accuracy] = svmpredict(ones(nrows*ncols, 1), Ximg, model, '-b 0');


% if isempty(cmd)
%     [predicted_labels, out_param.accuracy] = svmpredict(ones(nrows*ncols, 1), Ximg, model);
% else
%     [predicted_labels, out_param.accuracy, out_param.prob_estimates] = svmpredict(ones(nrows*ncols, 1), Ximg, model, cmd);
% end

% reshape the array of labels to the original dimensions of the image
% outdata = reshape(predicted_labels,nrows,ncols,1);
outdata = predicted_labels;
out_param.time_tot = toc;

