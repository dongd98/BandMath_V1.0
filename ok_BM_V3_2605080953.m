function Results = BM_V3(X_train, y_train, X_test, y_test, wavelengths, maxLV, fold)
% BANDMATH: Two-stage spectral feature extraction and optimization framework
%
% Inputs:
%   X_train, y_train: Calibration data and response
%   X_test, y_test: Validation/Test data and response
%   wavelengths: Vector of wavelength values (e.g., [400, 402, ..., 1000])
%   maxLV: Maximum latent variables for PLS
%   fold: Cross-validation folds

% BANDMATH: 光谱特征提取与优化的两阶段框架
%
% 输入:
%   X_train, y_train: 校正集光谱矩阵及响应变量
%   X_test, y_test: 测试集光谱矩阵及响应变量
%   wavelengths: 波长向量 (例如: [400, 402, ..., 1000])
%   maxLV: PLS 最大潜变量数
%   fold: 交叉验证折数

    tic;
    [N, P] = size(X_train);
    
    %% ====================================================================
    % PHASE 1: MONTE CARLO SAMPLING & STABILITY ANALYSIS
    % ====================================================================
    fprintf('Phase 1: Monte Carlo Sampling for VIP Stability...\n');
    N_MC = 100; % Fixed to 100 as per guidelines
    Q = round(0.8 * N); % 80% sampling ratio
    VIP_counts = zeros(1, P);
    
    for iter = 1:N_MC
        % Random sampling (CARS style)
        idx = randperm(N, Q);
        X_cal = X_train(idx, :);
        y_cal = y_train(idx);
        
        % Automatically determine optimal LV using CV
        [~, ~, ~, ~, ~, PLS_MSE] = plsregress(X_cal, y_cal, maxLV, 'CV', fold);
        [~, optLV] = min(PLS_MSE(2, 2:end)); % Minimize RMSECV
        
        % Build PLS model with optimal LV and calculate VIP
        % [Xloadings, Yloadings, Xscores, Yscores, beta, PLS_PCTVAR] = plsregress(X_cal, y_cal, optLV);
        % VIP = calculate_VIP(X_cal, Yloadings, Xscores, PLS_PCTVAR, optLV);
        VIP = calculate_VIP(X_cal,y_cal,optLV);
        
        % Record frequency of VIP > 1
        VIP_counts(VIP > 1) = VIP_counts(VIP > 1) + 1;
    end
    
    % Calculate Stability array
    Stability = VIP_counts / N_MC;
    
    fprintf('\n===== Stability Analysis =====\n');
    fprintf('Mean Stability : %.4f\n',mean(Stability));
    fprintf('Std Stability  : %.4f\n',std(Stability));
    fprintf('Max Stability  : %.4f\n',max(Stability));
    fprintf('Min Stability  : %.4f\n',min(Stability));
    
    fprintf('Stable Bands (S>0.8): %d\n',sum(Stability>0.8));
    fprintf('Candidate Bands (S>0.5): %d\n',sum(Stability>0.5));
    fprintf('Low Stable Bands (S<0.2): %d\n',sum(Stability<0.2));
    %% ====================================================================
    % PHASE 1.5: FIRST NSGA-II OPTIMIZATION (Conceptual Wrapper)
    % ====================================================================
    fprintf('Phase 1.5: NSGA-II Optimization for Single Wavelengths...\n');
    % Note: gamultiobj is used here. Variables are binary (1=selected, 0=not).
    % Fitness function minimizes: [RMSECV, -MeanStability, N_features]
    
    % For practical runtimes in this script, we simulate the pareto front selection 
    % by utilizing wavelengths that meet an initial stability threshold, but standard 
    % implementation should call gamultiobj passing the @fitness_func.
    
    % candidate_idx = find(Stability > 0);
    
    [~,idx_sort] = sort(Stability,'descend');
    
    Kmin = 50;
    Kmax = 100;
    
    K = round(0.1*P);
    K = max(Kmin,min(Kmax,K));
    
    candidate_idx = idx_sort(1:K);
    
    % selected_base_idx = idx_sort(1:min(K,length(idx_sort))); % 保留50~100个候选波长
    
    %% NSGA-II Feature Optimization
    candidate_idx = find(Stability > 0);

    X_candidate = X_train(:,candidate_idx);
    Stability_candidate = Stability(candidate_idx);

    nVar = length(candidate_idx);

    LB = zeros(1,nVar);
    UB = ones(1,nVar);

    opts = optimoptions('gamultiobj', 'PopulationSize',20, 'MaxGenerations',10, 'Display','iter');
    % opts = optimoptions('gamultiobj', 'PopulationSize',100, 'MaxGenerations',80, 'Display','iter');
    
    fitnessFcn = @(x) NSGA_Objective(x, X_candidate, y_train, Stability_candidate, maxLV, fold);
    
    % IntCon = 1:nVar;
    [xPareto,fPareto] = gamultiobj(fitnessFcn, nVar, [],[],[],[], LB,UB, opts);
    fprintf('Pareto solutions: %d\n',size(xPareto,1));

    % Pareto Knee Point Selection
    f_norm = normalize(fPareto);
    dist = sqrt(sum(f_norm.^2,2));
    [~,bestIdx] = min(dist);
    bestSolution = xPareto(bestIdx,:);
    selected_base_idx = candidate_idx(bestSolution > 0.5);
    fprintf('Selected wavelengths after NSGA-II: %d\n', length(selected_base_idx));

    %% ====================================================================
    % PHASE 2: TARGETED OPERATOR DERIVATION
    % ====================================================================
    fprintf('Phase 2: Operator Derivation (Diff, Ratio, NDI, Sum)...\n');
    % Do not use nchoosek(all,2). Use Top 5 highly correlated with > 20nm gap.
    k_max = 3; 
    X_operator_train = [];
    X_operator_test = [];
    operator_names = {};
    
    % Correlation matrix for all variables
    corr_matrix = corrcoef(X_train);
    
    for i = 1:length(selected_base_idx)
        idx_i = selected_base_idx(i);
        wv_i = wavelengths(idx_i);
        
        % Find absolute correlations for the current wavelength
        corrs = abs(corr_matrix(idx_i, :));
        
        % Apply > 20nm interval constraint
        wv_gaps = abs(wavelengths - wv_i);
        valid_mask = wv_gaps > 20;
        corrs(~valid_mask) = -1; % Exclude invalid pairs
        
        % Sort and select top k=5
        [~, sorted_corr_idx] = sort(corrs, 'descend');
        k_use = min(k_max,length(sorted_corr_idx));
        top_k_idx = sorted_corr_idx(1:k_use);
        
        for j = 1:length(top_k_idx)
            idx_j = top_k_idx(j);
            
            % Extract vectors
            Ri_tr = X_train(:, idx_i); Rj_tr = X_train(:, idx_j);
            Ri_te = X_test(:, idx_i);  Rj_te = X_test(:, idx_j);
            
            % Generate Operators
            % 1. Difference
            X_operator_train = [X_operator_train, Ri_tr - Rj_tr];
            X_operator_test = [X_operator_test, Ri_te - Rj_te];
            operator_names{end+1} = sprintf('Diff(%d,%d)', wv_i, wavelengths(idx_j));
            
            % 2. Ratio
            X_operator_train = [X_operator_train, Ri_tr ./ (Rj_tr + eps)];
            X_operator_test = [X_operator_test, Ri_te ./ (Rj_te + eps)];
            operator_names{end+1} = sprintf('Ratio(%d,%d)', wv_i, wavelengths(idx_j));
            
            % 3. NDI
            X_operator_train = [X_operator_train, (Ri_tr - Rj_tr) ./ (Ri_tr + Rj_tr + eps)];
            X_operator_test = [X_operator_test, (Ri_te - Rj_te) ./ (Ri_te + Rj_te + eps)];
            operator_names{end+1} = sprintf('NDI(%d,%d)', wv_i, wavelengths(idx_j));
            
            % 4. Sum
            X_operator_train = [X_operator_train, Ri_tr + Rj_tr];
            X_operator_test = [X_operator_test, Ri_te + Rj_te];
            operator_names{end+1} = sprintf('Sum(%d,%d)', wv_i, wavelengths(idx_j));
        end
    end
    
    %% ====================================================================
    % PHASE 3: FEATURE FUSION & VIP SECONDARY DISTILLATION
    % ====================================================================
    fprintf('Phase 3: Feature Fusion and Secondary VIP Distillation...\n');
    
    % X_fusion = [X_single, X_operator]
    X_fusion_train = [X_train(:, selected_base_idx), X_operator_train];
    X_fusion_test = [X_test(:, selected_base_idx), X_operator_test];
    
    % Second PLS run for VIP Distillation
    [~, ~, ~, ~, ~, PLS_MSE_fusion] = plsregress(X_fusion_train, y_train, min(maxLV, size(X_fusion_train,2)), 'CV', fold);
    [~, optLV_fusion] = min(PLS_MSE_fusion(2, 2:end));
    
    % [Xloadings_f, Yloadings_f, Xscores_f, Yscores_f, ~, PLS_PCTVAR_f] = plsregress(X_fusion_train, y_train, optLV_fusion);
    % VIP_fusion = calculate_VIP(X_fusion_train, Yloadings_f, Xscores_f, PLS_PCTVAR_f, optLV_fusion);
    VIP_fusion = calculate_VIP(X_fusion_train, y_train, optLV_fusion);
    
    % VIP > 1 rule, fallback to Top 20 VIP
    distilled_idx = find(VIP_fusion > 1);
    % if length(distilled_idx) < 20
    %     [~, sorted_vip_idx] = sort(VIP_fusion, 'descend');
    %     distilled_idx = sorted_vip_idx(1:min(20, length(VIP_fusion)));
    % end
    if length(distilled_idx)>50
        [~,idx]=sort(VIP_fusion,'descend');
        distilled_idx=idx(1:50);
    end
    
    X_final_train = X_fusion_train(:, distilled_idx);
    X_final_test = X_fusion_test(:, distilled_idx);
    
    %% ====================================================================
    % PHASE 4: FINAL PLSR MODELING & EVALUATION
    % ====================================================================
    fprintf('Phase 4: Final PLSR Modeling and Metric Evaluation...\n');
    
    % Final cross-validation to get CV metrics
    [~, ~, ~, ~, ~, PLS_MSE_final] = plsregress(X_final_train, y_train, min(maxLV, size(X_final_train,2)), 'CV', fold);
    [RMSECV_final, finalLV] = min(sqrt(PLS_MSE_final(2, 2:end)));
    
    % Final PLS calibration
    [~, ~, ~, ~, beta_final] = plsregress(X_final_train, y_train, finalLV);
    
    % Predict Calibration
    y_pred_c = [ones(size(X_final_train,1),1) X_final_train] * beta_final;
    RMSEC = sqrt(mean((y_train - y_pred_c).^2));
    Rc2 = 1 - sum((y_train - y_pred_c).^2) / sum((y_train - mean(y_train)).^2);
    
    % Predict Test Set
    y_pred_p = [ones(size(X_final_test,1),1) X_final_test] * beta_final;
    RMSEP = sqrt(mean((y_test - y_pred_p).^2));
    Rp2 = 1 - sum((y_test - y_pred_p).^2) / sum((y_test - mean(y_test)).^2);
    RPD = std(y_test) / RMSEP;
    
    % Compile Results
    Results.N_features = length(distilled_idx);
    Results.Final_LV = finalLV;
    Results.Metrics.RMSEC = RMSEC;
    Results.Metrics.Rc2 = Rc2;
    Results.Metrics.RMSECV = RMSECV_final;
    Results.Metrics.RMSEP = RMSEP;
    Results.Metrics.Rp2 = Rp2;
    Results.Metrics.RPD = RPD;
    
    fprintf('Modeling Complete. Final Features: %d | Rp2: %.4f | RPD: %.4f\n', ...
        Results.N_features, Rp2, RPD);
    time = toc;
    fprintf('Total Time: %.2f seconds.\n', time);

end

% ========================================================================
%% HELPER FUNCTION: Calculate VIP for PLS
% ========================================================================
function VIP = calculate_VIP(X,y,A)
    [n,p] = size(X); 
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,y,A);
    W = stats.W;
    SSY = sum(YS.^2,1).*sum(YL.^2,1);
    VIP = zeros(p,1);
    for j = 1:p
        VIP(j) = sqrt(p * sum(SSY.*(W(j,1:A).^2)./sum(W(:,1:A).^2,1)) / sum(SSY) );
    end
end

%% NSGA-II
function F = NSGA_Objective(x, X, y, Stability, maxLV, fold)
    selected = find(x>0.5);
    if length(selected)<3, F = [1e6 1e6 1e6]; return; end
    Xsub = X(:,selected);
    maxLV_use = min(maxLV,size(Xsub,2)-1);
    if maxLV_use<1, F = [1e6 1e6 1e6]; return; end
    try
        [~,~,~,~,~,~,MSE] = plsregress(Xsub, y, maxLV_use, 'CV',fold);
        RMSECV = min(sqrt(MSE(2,2:end)));
    catch
        RMSECV = 1e6;
    end
    MeanStability = mean(Stability(selected));
    Nfeature = length(selected);
    F = [RMSECV, -MeanStability, Nfeature];
end
