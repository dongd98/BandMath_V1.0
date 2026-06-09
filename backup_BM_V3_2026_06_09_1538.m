function Results = BM_V3(X_train, y_train, X_test, y_test, wavelengths, maxLV, fold, MODE, SWMODE)
% BANDMATH: Two-stage spectral feature extraction and optimization framework
%
% Inputs:
%   X_train, y_train: Calibration data and response
%   X_test, y_test: Validation/Test data and response
%   wavelengths: Vector of wavelength values (e.g., [400, 402, ..., 1000])
%   maxLV: Maximum latent variables for PLS
%   fold: Cross-validation folds
%   MODE: 'b':balance mode, 'm':RMSECV prior, 'f':Feature number prior
%   SWMODE: 'pct':稳定性波段百分比保留; 'acu':基于累计稳定性贡献保留; 
%           'thr':基于VIP阈值保留；'mix':pct+acu混合策略保留

% BANDMATH: 光谱特征提取与优化的两阶段框架
%
% 输入:
%   X_train, y_train: 校正集光谱矩阵及响应变量
%   X_test, y_test: 测试集光谱矩阵及响应变量
%   wavelengths: 波长向量 (例如: [400, 402, ..., 1000])
%   maxLV: PLS 最大潜变量数
%   fold: 交叉验证折数
if nargin<8
    MODE='b';
end

if nargin<9
    SWMODE='mix';
end

%%
    tic;
    [N, P] = size(X_train); 
    global ALL_SCORE;
    ALL_SCORE = [];
    global EVOLUTION_NFEATURE;    % <--- 新增：声明全局变量
    EVOLUTION_NFEATURE = [];      % <--- 新增：初始化为空数组
    FeatureInfo = struct();
    %% PHASE 1: Monte-Carlo + VIP Stability
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

    %% PHASE 1.5: NSGA-II 单波长优化
    fprintf('Phase 1.5: NSGA-II Optimization for Single Wavelengths...\n');
    % Note: gamultiobj is used here. Variables are binary (1=selected, 0=not).
    % Fitness function minimizes: [RMSECV, -MeanStability, N_features]
    
    % For practical runtimes in this script, we simulate the pareto front selection 
    % by utilizing wavelengths that meet an initial stability threshold, but standard 
    % implementation should call gamultiobj passing the @fitness_func.
    
    % candidate_idx = find(Stability > 0);
    
    switch lower(SWMODE)
        case 'pct'
        % 方案1：稳定性波段百分比保留（不太行
        [~,idx_sort] = sort(Stability,'descend');
        keep_ratio = 0.5;   % 波长保留比例
        N_keep = max(30, round(length(idx_sort)*keep_ratio));
        % candidate_idx = idx_sort(1:N_keep);
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); % <--- 修复：防止越界
        
        case 'acu'
        % 方案2：基于累计稳定性贡献（还行）
        [stab_sort,idx_sort] = sort(Stability,'descend');
        cum_stab = cumsum(stab_sort);
        cum_stab = cum_stab/cum_stab(end);
        N_keep = find(cum_stab>=0.9,1);
        % candidate_idx = idx_sort(1:N_keep);
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); % <--- 修复：防止越界
        
        case 'thr'
        % 原方案：保留大于0.5的波长
        [~,idx_sort] = sort(Stability,'descend');
        Kmin = 50;
        Kmax = 100;
        K = round(0.1*P);
        K = max(Kmin,min(Kmax,K));
        % candidate_idx = idx_sort(1:min(K,length(idx_sort)));
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); % <--- 修复：防止越界

        otherwise 
        % 方案3：二者结合稳定性贡献率+最小搜索空间（中间）
        [stab_sort,idx_sort] = sort(Stability,'descend');
        cum_stab = cumsum(stab_sort);
        cum_stab = cum_stab/cum_stab(end);
        N_keep = find(cum_stab>=0.9,1);
        N_keep = max(N_keep,30);
        % candidate_idx = idx_sort(1:N_keep);        
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); % <--- 修复：防止越界
    end
    %% --- NSGA-II Feature Optimization
    % candidate_idx = find(Stability > 0);

    X_candidate = X_train(:,candidate_idx);
    Stability_candidate = Stability(candidate_idx);
    nVar = length(candidate_idx);
    LB = zeros(1,nVar);
    UB = ones(1,nVar);
    % ===== light ===== 
    % opts = optimoptions('gamultiobj', 'PopulationSize',20, ...
    %     'MaxGenerations',10, 'Display','iter', 'OutputFcn',@save_population); 
    % % ===== Medium =====
    % opts = optimoptions('gamultiobj', 'PopulationSize',50, ...
    %     'MaxGenerations',30, 'Display','iter', 'OutputFcn',@save_population);
    % % ===== Heavy ===== 
    % opts = optimoptions('gamultiobj', 'PopulationSize',100, ...
    %     'MaxGenerations',80, 'Display','iter', 'OutputFcn',@save_population);
    
    % AUTO STOP
    opts = optimoptions('gamultiobj',...
    'PopulationSize',100,...
    'MaxGenerations',200,...
    'OutputFcn',{@NSGA_OutputFcn, @save_population},...
    'Display','iter');
    
    fitnessFcn = @(x) NSGA_Objective(x, X_candidate, y_train, Stability_candidate, maxLV, fold);
    
    % NSGA-II Feature Subset Selection
    IntCon = 1:nVar;
    [xPareto,fPareto] = gamultiobj(...
                        fitnessFcn,...
                        nVar,...
                        [],[],[],[],...
                        LB,UB,...
                        [],...
                        IntCon,...
                        opts); 
    fprintf('Pareto solutions: %d\n',size(xPareto,1));
    
    %% --- --- NSGA-II 寻优结果可视化绘图区  --- --- 
    % Pareto Knee Point Selection
    f_norm = normalize(fPareto);
    dist = sqrt(sum(f_norm.^2,2));
    % === === MODE 参数选择模型倾向 === === 
    switch lower(MODE)
        case 'm', [~,bestIdx] = min(fPareto(:,1));      % 模型性能优先（RMSECV最小）
        case 'f', [~,bestIdx] = min(fPareto(:,3));   % 特征压缩效率优先（Feature Number最小）
        otherwise, [~,bestIdx] = min(dist);                 % 平衡模式（knee Point选择）
    end
    bestSolution = xPareto(bestIdx,:);
    selected_base_idx = candidate_idx(bestSolution > 0.5);
    % 保留单波长波长信息
    % FeatureInfo = struct();
    FeatureInfo = struct('Type',{}, 'Band1',{}, 'Band2',{}, 'Formula',{}); % <--- 修复
    for i = 1:length(selected_base_idx)
        % 保留单波长波长信息
        idx_i = selected_base_idx(i);
        FeatureInfo(end+1).Type = 'Raw';
        FeatureInfo(end).Band1 = wavelengths(idx_i);
        FeatureInfo(end).Band2 = [];
        FeatureInfo(end).Formula = 'Raw';
    end
    fprintf('Selected wavelengths after NSGA-II: %d\n', length(selected_base_idx));
    
    Results.Pareto.X = xPareto;
    Results.Pareto.F = fPareto;
    
    % 保存 AUTO STOP 最佳代记录 
    global NSGA_HISTORY
    Results.NSGA.BestGeneration = NSGA_HISTORY.globalBestGeneration;
    Results.NSGA.BestRMSECV = NSGA_HISTORY.globalBestRMSECV;
    Results.NSGA.BestSolution = NSGA_HISTORY.globalBestSolution;

    % 1.Pareto 解空间可视化
    figure('Name','NSGA-II Pareto Search Space');
    scatter3(ALL_SCORE(:,1), -ALL_SCORE(:,2), ALL_SCORE(:,3), 15, [0.8 0.8 0.8], 'filled'); hold on
    scatter3(fPareto(:,1), -fPareto(:,2), fPareto(:,3), 80, 'r', 'filled', '^');
    xlabel('RMSECV'); ylabel('Mean Stability'); zlabel('Feature Number');
    title('NSGA-II Search Space and Pareto Front');
    legend('All Solutions','Pareto Front', Location='best');
    grid on;

    % 2.Evolution curve 进化曲线可视化
    global NSGA_HISTORY;
    figure;
    yyaxis left
    plot(NSGA_HISTORY.bestRMSECV, 'LineWidth',2); ylabel('Best RMSECV')
    yyaxis right
    plot(NSGA_HISTORY.spread, 'LineWidth',2); ylabel('Pareto Spread'); xlabel('Generation');
    title('NSGA-II Evolution Process'); grid on;
    
    % 3.Feature Number Evolution 特征数量进化曲线
    figure
    plot(EVOLUTION_NFEATURE,'LineWidth',2);
    xlabel('Generation'); ylabel('Mean Feature Number');
    title('Feature Number Evolution'); grid on;
    
    %% PHASE 2: 算子衍生（TARGETED OPERATOR DERIVATION）
    fprintf('Phase 2: Operator Derivation (Diff, Ratio, NDI, Sum)...\n');
    % Do not use nchoosek(all,2). Use Top 5 highly correlated with > 20nm gap.
    k_max = 2; % k_max=3 也可以
    X_operator_train = [];
    X_operator_test = [];
    operator_names = {};

    
    OperatorInfo = struct('Band1',{}, 'Band2',{}, 'Operator',{});

    % Correlation matrix for all variables
    corr_y = abs(corr(X_train, y_train));

    for i = 1:length(selected_base_idx)
        idx_i = selected_base_idx(i);
        wv_i = wavelengths(idx_i);
                
        % Find absolute correlations for the current wavelength
        candidate_corr = corr_y;
        candidate_corr(idx_i)=0;

        % Apply > 20nm interval constraint
        wv_gap = abs(wavelengths - wavelengths(idx_i));
        candidate_corr(wv_gap<20)=0;

        % Sort and select top k=k_max
        [~,idx_sort]=sort(candidate_corr,'descend');
        k_use=min(k_max,sum(candidate_corr>0));
        top_k_idx=idx_sort(1:k_use);

        for j = 1:length(top_k_idx)
            idx_j = top_k_idx(j);
            
            % Extract vectors
            Ri_tr = X_train(:, idx_i); Rj_tr = X_train(:, idx_j);
            Ri_te = X_test(:, idx_i);  Rj_te = X_test(:, idx_j);
            
            % 算子衍生特征 Generate Operators
            % 1. Difference
            X_operator_train = [X_operator_train, Ri_tr - Rj_tr];
            X_operator_test = [X_operator_test, Ri_te - Rj_te];
            operator_names{end+1} = sprintf('Diff(%d,%d)', wv_i, wavelengths(idx_j));
            OperatorInfo(end+1).Band1 = wv_i;
            OperatorInfo(end).Band2   = wavelengths(idx_j);
            OperatorInfo(end).Operator = 'Diff';
            FeatureInfo(end+1).Type='Diff';
            FeatureInfo(end).Band1=wv_i;
            FeatureInfo(end).Band2=wavelengths(idx_j);
            FeatureInfo(end).Formula='Ri-Rj';

            % % 2. Sum
            % X_operator_train = [X_operator_train, Ri_tr + Rj_tr];
            % X_operator_test = [X_operator_test, Ri_te + Rj_te];
            % operator_names{end+1} = sprintf('Sum(%d,%d)', wv_i, wavelengths(idx_j));
            % OperatorInfo(end+1).Band1 = wv_i;
            % OperatorInfo(end).Band2   = wavelengths(idx_j);
            % OperatorInfo(end).Operator = 'Sum';
            % FeatureInfo(end+1).Type='Sum';
            % FeatureInfo(end).Band1=wv_i;
            % FeatureInfo(end).Band2=wavelengths(idx_j);
            % FeatureInfo(end).Formula='Ri+Rj';

            % 3. Ratio
            X_operator_train = [X_operator_train, Ri_tr ./ (Rj_tr + eps)];
            X_operator_test = [X_operator_test, Ri_te ./ (Rj_te + eps)];
            operator_names{end+1} = sprintf('Ratio(%d,%d)', wv_i, wavelengths(idx_j));
            OperatorInfo(end+1).Band1 = wv_i;
            OperatorInfo(end).Band2   = wavelengths(idx_j);
            OperatorInfo(end).Operator = 'Ratio';
            FeatureInfo(end+1).Type='Ratio';
            FeatureInfo(end).Band1=wv_i;
            FeatureInfo(end).Band2=wavelengths(idx_j);
            FeatureInfo(end).Formula='Ri/Rj';

            % 4. NDI
            X_operator_train = [X_operator_train, (Ri_tr - Rj_tr) ./ (Ri_tr + Rj_tr + eps)];
            X_operator_test = [X_operator_test, (Ri_te - Rj_te) ./ (Ri_te + Rj_te + eps)];
            operator_names{end+1} = sprintf('NDI(%d,%d)', wv_i, wavelengths(idx_j));
            OperatorInfo(end+1).Band1 = wv_i;
            OperatorInfo(end).Band2   = wavelengths(idx_j);
            OperatorInfo(end).Operator = 'NDI';
            FeatureInfo(end+1).Type='NDI';
            FeatureInfo(end).Band1=wv_i;
            FeatureInfo(end).Band2=wavelengths(idx_j);
            FeatureInfo(end).Formula='Ri-Rj/Ri+Rj';
        end
    end
       
    %% PHASE 3: 特征融合 single + Operator FEATURE FUSION & VIP SECONDARY DISTILLATION
    fprintf('Phase 3: Feature Fusion and Secondary VIP Distillation...\n');
    
    X_fusion_train = [X_train(:, selected_base_idx), X_operator_train];
    X_fusion_test = [X_test(:, selected_base_idx), X_operator_test];
    
    % Second PLS run for VIP Distillation
    [~, ~, ~, ~, ~, PLS_MSE_fusion] = plsregress(X_fusion_train, y_train, min(maxLV, size(X_fusion_train,2)), 'CV', fold);
    [~, optLV_fusion] = min(PLS_MSE_fusion(2, 2:end));
    
    VIP_fusion = calculate_VIP(X_fusion_train, y_train, optLV_fusion);
    % 生成FeatureBank_Before，保存单波长+算子特征+VIP
    nRaw = length(selected_base_idx);
    nFusion = length(VIP_fusion);
    FeatureBank_Before = struct();
    for i = 1:nFusion
        FeatureBank_Before(i).VIP = VIP_fusion(i);
        if i <= nRaw
            FeatureBank_Before(i).Type = 'Raw';
            FeatureBank_Before(i).Band1 = wavelengths(selected_base_idx(i));
            FeatureBank_Before(i).Band2 = [];
            FeatureBank_Before(i).Operator = 'Raw';
        else
            opIdx = i - nRaw;
            FeatureBank_Before(i).Type = 'Operator';
            FeatureBank_Before(i).Band1 = OperatorInfo(opIdx).Band1;
            FeatureBank_Before(i).Band2 = OperatorInfo(opIdx).Band2;
            FeatureBank_Before(i).Operator = OperatorInfo(opIdx).Operator;
        end
    end
    
    % VIP > 1 rule, fallback to Top 20 VIP
    candidate_nums = round(linspace(5,length(VIP_fusion),30));
    best_rmsecv = inf;
    best_idx = [];
    for k = 1:length(candidate_nums)
        keep_num = candidate_nums(k);
        [~,sort_idx] = sort(VIP_fusion,'descend');
        temp_idx = sort_idx(1:keep_num);
        Xtemp = X_fusion_train(:,temp_idx);
        maxLV_use = min(maxLV,size(Xtemp,2)-1);
        [~,~,~,~,~,~,MSE] = plsregress(Xtemp, y_train, maxLV_use, 'CV',fold);
        rmsecv = min(sqrt(MSE(2,2:end)));
        if rmsecv < best_rmsecv
            best_rmsecv = rmsecv;
            best_idx = temp_idx;
        end
    end
    distilled_idx = best_idx;

    % distilled_idx = find(VIP_fusion > 1);
    % if length(distilled_idx)>50
    %     [~,idx]=sort(VIP_fusion,'descend');
    %     distilled_idx=idx(1:50);
    % end

    %% === ===  局部冗余压缩层 (Local Wavelength Competition, LWC) === === 
    keep = true(length(distilled_idx),1);
    gap_nm = 12; % 比较12nm间隔波长的VIP值
    
    for i = 1:length(distilled_idx)
        if ~keep(i); continue; end
        for j = i+1:length(distilled_idx)
            wl_i = FeatureBank_Before(distilled_idx(i)).Band1;
            wl_j = FeatureBank_Before(distilled_idx(j)).Band1;
            if isempty(wl_i) || isempty(wl_j); continue; end
            if abs(wl_i - wl_j) < gap_nm
                vip_i = VIP_fusion(distilled_idx(i));
                vip_j = VIP_fusion(distilled_idx(j));
                if vip_i >= vip_j; keep(j) = false; else; keep(i) = false; end
            end
        end
    end
    
    distilled_idx = distilled_idx(keep);
    % === === 局部冗余压缩 结束 === ===    

    X_final_train = X_fusion_train(:, distilled_idx);
    X_final_test = X_fusion_test(:, distilled_idx);
        %% --- Phase 3.1: 融合特征 结构体
    nFusion = size(X_fusion_train,2);
    FeatureBank_Before = struct();
    for i = 1:nFusion
        FeatureBank_Before(i).VIP = VIP_fusion(i);
        if i <= length(selected_base_idx)
            FeatureBank_Before(i).Type = 'Raw';
            FeatureBank_Before(i).Band1 = wavelengths(selected_base_idx(i));
            FeatureBank_Before(i).Band2 = [];
            FeatureBank_Before(i).Operator = 'Raw';
        else   
            opID = i - length(selected_base_idx);    
            FeatureBank_Before(i).Type = 'Operator';  
            FeatureBank_Before(i).Band1 = OperatorInfo(opID).Band1;
            FeatureBank_Before(i).Band2 = OperatorInfo(opID).Band2;
            FeatureBank_Before(i).Operator = OperatorInfo(opID).Operator;  
        end
    end
    %% PHASE 4: VIP二次蒸馏 FINAL PLSR MODELING & EVALUATION
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

    % === === 最终保留波长 === === 
    FeatureBank_After = FeatureBank_Before(distilled_idx);
    Results.FeatureBank_Before = FeatureBank_Before;
    Results.FeatureBank_After = FeatureBank_After;
    % === === 打印最终保留波长 === === 
    fprintf('\n===== Final Selected Features =====\n');
    for i=1:length(FeatureBank_After)
        if strcmp(FeatureBank_After(i).Type,'Raw')
            fprintf('%3d | Raw | %d nm | VIP=%.3f\n', i, FeatureBank_After(i).Band1, FeatureBank_After(i).VIP);
        else
            fprintf('%3d | %s | (%d,%d) | VIP=%.3f\n', i, FeatureBank_After(i).Operator, FeatureBank_After(i).Band1, FeatureBank_After(i).Band2, FeatureBank_After(i).VIP);
        end
    end

    fprintf('Modeling Complete. Final Features: %d | Rp2: %.4f | RPD: %.4f\n', ...
        Results.N_features, Rp2, RPD);
    time = toc;
    fprintf('Total Time: %.2f seconds.\n', time);

    %% --- Phase 4.1: VIP 二次蒸馏后的特征保留 结构体
    FeatureBank_After = FeatureBank_Before(distilled_idx); % 统计蒸馏后的特征
    all_idx = 1:length(FeatureBank_Before);
    remove_idx = setdiff(all_idx, distilled_idx);
    DistillationReport = struct();
    DistillationReport.KeepIdx = distilled_idx;
    DistillationReport.RemoveIdx = remove_idx;
    DistillationReport.N_Before = length(all_idx);
    DistillationReport.N_After = length(distilled_idx);
    DistillationReport.CompressionRatio = 100*(1-length(distilled_idx)/length(all_idx));
    DistillationReport.KeepNames = {};
    DistillationReport.RemoveNames = {};

    for i = 1:length(distilled_idx) % 保留特征
        DistillationReport.KeepNames{i} = featureName(FeatureBank_Before(distilled_idx(i)));
    end
    for i = 1:length(remove_idx) % 删除特征
        DistillationReport.RemoveNames{i} = featureName(FeatureBank_Before(remove_idx(i)));
    end
    % 保存到Results
    Results.FeatureBank_Before = FeatureBank_Before;
    Results.FeatureBank_After = FeatureBank_After;
    Results.DistillationReport = DistillationReport;
    
    VIP_sort = sort(VIP_fusion,'descend'); % 蒸馏后VIP排序
    VIP_after = VIP_fusion(distilled_idx);
    VIP_after = sort(VIP_after,'descend');
    %% --- --- 二次蒸馏结果可视化绘图区 --- --- 
    % % 1.二次蒸馏VIP演化图
    % figure;
    % plot(1:length(VIP_sort),VIP_sort,'LineWidth',2); hold on; 
    % yline(1,'r--','VIP=1');
    % xlabel('Feature Rank'); ylabel('VIP')
    % title('VIP Distribution Before Distillation'); grid on
    % 
    % figure;
    % plot(VIP_after,'LineWidth',2); hold on
    % yline(1,'r--');
    % xlabel('Feature Rank'); ylabel('VIP')
    % title('VIP Distribution After Distillation'); grid on    

    % 2.二次蒸馏VIP演化图(VIP Before/After同一张图)
    figure
    plot(VIP_sort,'LineWidth',2); hold on
    plot(1:length(VIP_after), VIP_after,'LineWidth',2);
    yline(1,'k--','VIP=1')
    xlabel('Feature Rank'); ylabel('VIP'); legend('Before Distillation', 'After Distillation')
    title('VIP Distribution Comparison');
    grid on
       
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

function Fname = featureName(feature)
    if strcmp(feature.Operator,'Raw')
        Fname = sprintf('%dnm',feature.Band1);
    else
        Fname = sprintf('%s(%d,%d)', feature.Operator, feature.Band1, feature.Band2);
    end
end


%% NSGA-II
%% --- NSGA_Objective --- 
function F = NSGA_Objective(x, X, y, Stability, maxLV, fold)
    x = double(x > 0.5); % 保证所有评价都是对子集进行
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

%% --- NSGA_Objective ---  
function [state,options,optchanged] = NSGA_OutputFcn(options,state,flag)
global NSGA_HISTORY;
optchanged = false;
switch flag
    case 'init'
        NSGA_HISTORY = struct(); % 彻底清空上一轮残留的全局最优记录
        NSGA_HISTORY.bestRMSECV = [];
        NSGA_HISTORY.spread = [];
        NSGA_HISTORY.noImprove = 0;
    case 'iter'
        scores = state.Score;
        % current_best = min(scores(:,1));
        [current_best, best_idx] = min(scores(:,1)); % 同时获取最小值和它的索引
        % AUTO STOP最佳代记录
        if ~isfield(NSGA_HISTORY,'globalBestRMSECV')
            NSGA_HISTORY.globalBestRMSECV = current_best;
            NSGA_HISTORY.globalBestGeneration = state.Generation;
            NSGA_HISTORY.globalBestSolution = state.Population(best_idx,:);
        elseif current_best < NSGA_HISTORY.globalBestRMSECV
            NSGA_HISTORY.globalBestRMSECV = current_best;
            NSGA_HISTORY.globalBestGeneration = state.Generation;
            NSGA_HISTORY.globalBestSolution = state.Population(best_idx,:);
        end

        if isempty(NSGA_HISTORY.bestRMSECV)
            NSGA_HISTORY.bestRMSECV(end+1)=current_best;
        else
            old_best = min(NSGA_HISTORY.bestRMSECV);
            if current_best < old_best - 1e-5
                NSGA_HISTORY.noImprove = 0;
            else
                NSGA_HISTORY.noImprove = NSGA_HISTORY.noImprove + 1;
            end
            NSGA_HISTORY.bestRMSECV(end+1)=current_best;
        end
        spread = std(scores(:,1));
        NSGA_HISTORY.spread(end+1)=spread;
        if length(NSGA_HISTORY.spread)>20
            recent = NSGA_HISTORY.spread(end-19:end);
            spread_change = max(recent)-min(recent);
            if spread_change < 1e-4 && NSGA_HISTORY.noImprove >=20
                state.StopFlag = 'Pareto converged';
            end
        end
    end
end

%% Output Function （OutputFcn）
function [state,options,optchanged] = save_population(options,state,flag)
    global ALL_SCORE;
    global EVOLUTION_NFEATURE; % 声明全局变量
    optchanged = false;
    if strcmp(flag,'init') % 算法初始化阶段，清空历史数据，防止多次运行 test 时数据累积
        ALL_SCORE = [];
        EVOLUTION_NFEATURE = [];
    elseif strcmp(flag,'iter')
        ALL_SCORE = [ALL_SCORE; state.Score]; % 算法迭代阶段，记录分数和特征数
        nfeat = sum(state.Population>0.5, 2); % 记录每一代的平均特征数
        EVOLUTION_NFEATURE(end+1) = mean(nfeat);
    end
end

% function [state,options,optchanged] = save_population(options,state,flag)
%     global ALL_SCORE;
%     optchanged = false;
%     if strcmp(flag,'iter')
%         ALL_SCORE = [ALL_SCORE; state.Score];
%     end
%     % 加入了特征数 Evolution Curve
%     global EVOLUTION_NFEATURE;
%     nfeat = sum(state.Population>0.5,2);
%     EVOLUTION_NFEATURE(end+1)=mean(nfeat);
% end



