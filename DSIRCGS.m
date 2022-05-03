%% DSIRCGS
% Extracts performances for DSIRC

%% Grid Search Parameters
   
% Set number of nearest neighbors to use in graph and KDE construction.
NNs = 10:10:100;

% Set the percentiles of nearest neighbor distances to be used in KDE construction. 
prcts{1} =  41:(100-41)/19:100;
prcts{2} =  41:(100-41)/19:100;

numReplicates = 10;

%% Grid searches
datasets = {'IndianPinesCorrected','IndianPinesCorrectedSar'};

% You need to reconstruct 'IndianPinesCorrected' data using "SaR_main.m" and 
% rename it as 'IndianPinesCorrectedSar.mat' before setting dataIdx =  2.

% The datasets should be formatted as:
% M,N,D : The number of rows, columns, and spectral bands in HSI.
% X     : (M*N)*D matrix.
% HSI   : M*N*D array.
% GT    : Ground Truth labels.


for dataIdx =  2
    prctiles = prcts{dataIdx};
    if dataIdx > 1
        % Load the original dataset for calculating \zeta(x)
        load(datasets{1},'X')
        X = X./vecnorm(X,2,2);
        X1 = X;
        [Idx_NN1, Dist_NN1] = knnsearch(X, X, 'K', 1000);
        Dist_NN1 = Dist_NN1(:,2:902);
        Idx_NN1 = Idx_NN1(:,2:902);
    end

    % ===================== Load and Preprocess Data ======================
    
    % Load data
    load(datasets{dataIdx})
    
    % Normalization
    X = X./vecnorm(X,2,2);
    HSI = reshape(X, size(HSI, 1),size(HSI, 2), size(HSI,3));
    
    [Idx_NN, Dist_NN] = knnsearch(X, X, 'K', 1000);
    Dist_NN = Dist_NN(:,2:end);
    Idx_NN = Idx_NN(:,2:end);

    [M,N,D] = size(HSI);

    newGT = zeros(size(GT));
    uniqueClass = unique(GT);
    K = length(uniqueClass);
    for k = 1:K
    newGT(GT==uniqueClass(k)) = k;
    end
    Y = reshape(newGT,M*N,1);
    GT = newGT;

    Idx_NN = Idx_NN(:,1:901);
    Dist_NN = Dist_NN(:,1:901);
    clear uniqueClass k 
 
    % Set Default parameters
    Hyperparameters.SpatialParams.ImageSize = [M,N];
    Hyperparameters.NEigs = 10;
    Hyperparameters.NumDtNeighbors = 200;
    Hyperparameters.Beta = 2;
    Hyperparameters.Tau = 10^(-5);
    Hyperparameters.Tolerance = 1e-8;
    K = length(unique(Y));
    Hyperparameters.K_Known = K;
    
    if dataIdx == 1
        X1 = X;
        Dist_NN1 = Dist_NN;
    end

    % ============================== DSIRC ==============================

    % Preallocate memory
    OAs     = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    kappas  = NaN*zeros(length(NNs), length(prctiles), numReplicates);
    Cs      = zeros(M*N,length(NNs), length(prctiles), numReplicates);

    currentPerf = 0;
    % Run Grid Searches
    for i = 1:length(NNs)
        for j = 1:length(prctiles)
            for k = 1:numReplicates

                Hyperparameters.DiffusionNN = NNs(i);
                Hyperparameters.DensityNN = NNs(i); % must be ≤ 1000
                Hyperparameters.Sigma0 = prctile(Dist_NN1(Dist_NN1>0), prctiles(j), 'all');
                if dataIdx >=12
                    Hyperparameters.EndmemberParams.K = K; % compute hysime to get best estimate for number of endmembers
                else
                    Hyperparameters.EndmemberParams.K = hysime(X1'); % compute hysime to get best estimate for number of endmembers
                end
                Hyperparameters.EndmemberParams.Algorithm = 'ManyAVMAX';
                Hyperparameters.EndmemberParams.NumReplicates = 100;

                tic
                [pixelPurity, U, A] = compute_purity(X1,Hyperparameters);
                toc

                density = KDE_large(Dist_NN1, Hyperparameters);
                [G,W] = extract_graph_large(X, Hyperparameters, Idx_NN, Dist_NN);

                if G.EigenVals(2)<1
                    Clusterings = MLUND_large(X, Hyperparameters, G, harmmean([density./max(density), pixelPurity./max(pixelPurity)],2));
                    [~,~, OAs(i,j,k), ~, kappas(i,j,k), tIdx]= measure_performance(Clusterings, Y);
                    C =  Clusterings.Labels(:,tIdx);
                    Cs(:,i,j,k) = C;
                end
    
                disp(['DSIRS: '])
                disp([i/length(NNs), j/length(prctiles), k/numReplicates, currentPerf])

            end
            currentPerf = max(nanmean(OAs,3),[],'all');
        end 
        [n1,n2] = size(nanmean(OAs,3));
        [maxOA, k] = max(reshape(nanmean(OAs,3),n1*n2,1));
        [l,j] = ind2sub(size(mean(OAs,3)), k);
        stdOA = nanstd(squeeze(OAs(l,j,:)));
        save(strcat(datasets{dataIdx}, 'DSIRC'),  'OAs', 'kappas', 'Cs', 'NNs', 'prctiles', 'numReplicates', 'maxOA', 'stdOA')
        
    end

end
