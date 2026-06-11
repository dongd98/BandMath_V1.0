function Results = BM_V3(X_train, y_train, X_test, y_test, wavelengths, maxLV, fold, MODE, SWMODE, nsga_mode, op_switch)
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
%   nsga_mode: 'light', 'medium', 'heavy', 'auto' (控制第一阶段的迭代强度)
%   op_switch: 1x4 array [Diff, Sum, Ratio, NDI], e.g., [1,0,1,0] (1开启, 0关闭)

% BANDMATH: 光谱特征提取与优化的两阶段框架
%
if nargin<8
    MODE='b';
end

if nargin<9
    SWMODE='mix';
end

if nargin<10
    nsga_mode='auto'; % 默认使用 AUTO STOP 策略
end

if nargin<11
    op_switch=[1, 1, 1, 1]; % 默认开启全部四个算子 [Diff, Sum, Ratio, NDI]
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
    
    switch lower(SWMODE)
        case 'pct'
        [~,idx_sort] = sort(Stability,'descend');
        keep_ratio = 0.5;   
        N_keep = max(30, round(length(idx_sort)*keep_ratio));
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); 
        
        case 'acu'
        [stab_sort,idx_sort] = sort(Stability,'descend');
        cum_stab = cumsum(stab_sort);
        cum_stab = cum_stab/cum_stab(end);
        N_keep = find(cum_stab>=0.9,1);
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); 
        
        case 'thr'
        [~,idx_sort] = sort(Stability,'descend');
        Kmin = 50; Kmax = 100;
        K = round(0.1*P);
        K = max(Kmin,min(Kmax,K));
        candidate_idx = idx_sort(1:min(K, length(idx_sort))); 

        otherwise 
        [stab_sort,idx_sort] = sort(Stability,'descend');
        cum_stab = cumsum(stab_sort);
        cum_stab = cum_stab/cum_stab(end);
        N_keep = find(cum_stab>=0.9,1);
        N_keep = max(N_keep,30);
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); 
    end
    
    %% --- NSGA-II Feature Optimization
    X_candidate = X_train(:,candidate_idx);
    Stability_candidate = Stability(candidate_idx);
    nVar = length(candidate_idx);
    LB = zeros(1,nVar);
    UB = ones(1,nVar);
    
    % === === NSGA-II 寻优空间复杂度开关 === === 
    fprintf('NSGA-II Mode selected: %s\n', upper(nsga_mode));
    switch lower(nsga_mode)    
        case 'light'
        %% === === 1. NSGA-II light === === 
            opts = optimoptions('gamultiobj', 'PopulationSize',20, ...
                'MaxGenerations',10, 'Display','iter', 'OutputFcn',@save_population); 
        case 'medium'
        %% === === 2. NSGA-II medium === === 
            opts = optimoptions('gamultiobj', 'PopulationSize',50, ...
                'MaxGenerations',30, 'Display','iter', 'OutputFcn',@save_population);
        case 'heavy'
        %% === === 3. NSGA-II heavy === ===
            opts = optimoptions('gamultiobj', 'PopulationSize',100, ...
                'MaxGenerations',80, 'Display','iter', 'OutputFcn',@save_population);
        case 'auto'
        %% === === 4. NSGA-II auto === ===
            opts = optimoptions('gamultiobj',...
                'PopulationSize',100,...
                'MaxGenerations',200,...
                'OutputFcn',{@NSGA_OutputFcn, @save_population},...
                'Display','iter');
        otherwise
            error('未知的 nsga_mode。请使用 light, medium, heavy 或 auto');
    end
    % ====================================== 

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
    f_norm = normalize(fPareto);
    dist = sqrt(sum(f_norm.^2,2));
    switch lower(MODE)
        case 'm', [~,bestIdx] = min(fPareto(:,1));      
        case 'f', [~,bestIdx] = min(fPareto(:,3));   
        otherwise, [~,bestIdx] = min(dist);                 
    end
    bestSolution = xPareto(bestIdx,:);
    selected_base_idx = candidate_idx(bestSolution > 0.5);
    
    FeatureInfo = struct('Type',{}, 'Band1',{}, 'Band2',{}, 'Formula',{}); 
    for i = 1:length(selected_base_idx)
        idx_i = selected_base_idx(i);
        FeatureInfo(end+1).Type = 'Raw';
        FeatureInfo(end).Band1 = wavelengths(idx_i);
        FeatureInfo(end).Band2 = [];
        FeatureInfo(end).Formula = 'Raw';
        FeatureInfo(end).Index=idx_i;
    end
    fprintf('Selected wavelengths after NSGA-II: %d\n', length(selected_base_idx));
    
    Results.Pareto.X = xPareto;
    Results.Pareto.F = fPareto;
    
    % 保存 AUTO STOP 最佳代记录 
    global NSGA_HISTORY
    if isfield(NSGA_HISTORY, 'globalBestGeneration')
        Results.NSGA.BestGeneration = NSGA_HISTORY.globalBestGeneration;
        Results.NSGA.BestRMSECV = NSGA_HISTORY.globalBestRMSECV;
        Results.NSGA.BestSolution = NSGA_HISTORY.globalBestSolution;
    end

    figure('Name','NSGA-II Pareto Search Space');
    scatter3(ALL_SCORE(:,1), -ALL_SCORE(:,2), ALL_SCORE(:,3), 15, [0.8 0.8 0.8], 'filled'); hold on
    scatter3(fPareto(:,1), -fPareto(:,2), fPareto(:,3), 80, 'r', 'filled', '^');
    xlabel('RMSECV'); ylabel('Mean Stability'); zlabel('Feature Number');
    title('NSGA-II Search Space and Pareto Front');
    legend('All Solutions','Pareto Front', Location='best');
    grid on;

    if strcmpi(nsga_mode, 'auto') && isfield(NSGA_HISTORY, 'bestRMSECV')
        figure;
        yyaxis left
        plot(NSGA_HISTORY.bestRMSECV, 'LineWidth',2); ylabel('Best RMSECV')
        yyaxis right
        plot(NSGA_HISTORY.spread, 'LineWidth',2); ylabel('Pareto Spread'); xlabel('Generation');
        title('NSGA-II Evolution Process'); grid on;
    end
    
    figure
    plot(EVOLUTION_NFEATURE,'LineWidth',2);
    xlabel('Generation'); ylabel('Mean Feature Number');
    title('Feature Number Evolution'); grid on;
    
    %% PHASE 2: 算子衍生（TARGETED OPERATOR DERIVATION）
    fprintf('Phase 2: Operator Derivation...\n');
    fprintf('Operator Switch Status -> Diff:%d | Sum:%d | Ratio:%d | NDI:%d\n', op_switch(1), op_switch(2), op_switch(3), op_switch(4));
    
    k_max = 2; 
    X_operator_train = [];
    X_operator_test = [];
    operator_names = {};
    OperatorInfo = struct('Band1',{}, 'Band2',{}, 'Operator',{});

    corr_y = abs(corr(X_train, y_train));

    for i = 1:length(selected_base_idx)
        idx_i = selected_base_idx(i);
        wv_i = wavelengths(idx_i);
                
        candidate_corr = corr_y;
        candidate_corr(idx_i)=0;

        wv_gap = abs(wavelengths - wavelengths(idx_i));
        candidate_corr(wv_gap<20)=0;

        [~,idx_sort]=sort(candidate_corr,'descend');
        k_use=min(k_max,sum(candidate_corr>0));
        top_k_idx=idx_sort(1:k_use);

        for j = 1:length(top_k_idx)
            idx_j = top_k_idx(j);
            
            Ri_tr = X_train(:, idx_i); Rj_tr = X_train(:, idx_j);
            Ri_te = X_test(:, idx_i);  Rj_te = X_test(:, idx_j);
            
            %% === === 算子自由组合开关 === === 
            % 1. Difference
            if op_switch(1) == 1
                %% 1. Diff
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
            end
            if op_switch(2) == 1
                %% 2. Sum
                X_operator_train = [X_operator_train, Ri_tr + Rj_tr];
                X_operator_test = [X_operator_test, Ri_te + Rj_te];
                operator_names{end+1} = sprintf('Sum(%d,%d)', wv_i, wavelengths(idx_j));
                OperatorInfo(end+1).Band1 = wv_i;
                OperatorInfo(end).Band2   = wavelengths(idx_j);
                OperatorInfo(end).Operator = 'Sum';
                FeatureInfo(end+1).Type='Sum';
                FeatureInfo(end).Band1=wv_i;
                FeatureInfo(end).Band2=wavelengths(idx_j);
                FeatureInfo(end).Formula='Ri+Rj';
            end

            if op_switch(3) == 1
                %% 3. Ratio
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
            end
 
            if op_switch(4) == 1
                %% 4. NDI
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
    end
       
    %% PHASE 3: 特征融合 single + Operator FEATURE FUSION & VIP SECONDARY DISTILLATION
    fprintf('Phase 3: Feature Fusion and Secondary VIP Distillation...\n');
    
    X_fusion_train = [X_train(:, selected_base_idx), X_operator_train];
    X_fusion_test = [X_test(:, selected_base_idx), X_operator_test];
    
    % [修复3] 安全确定 LV 数，防止无算子时特征数过少报错
    maxLV_fusion = max(1, min(maxLV, size(X_fusion_train,2)));
    [~, ~, ~, ~, ~, PLS_MSE_fusion] = plsregress(X_fusion_train, y_train, maxLV_fusion, 'CV', fold);
    [~, optLV_fusion] = min(PLS_MSE_fusion(2, 2:end));
    
    VIP_fusion = calculate_VIP(X_fusion_train, y_train, optLV_fusion);
    
    nRaw = length(selected_base_idx);
    Score_fusion = VIP_fusion;
    
    % [修复1] 只有在存在算子特征时，才进行打分衰减
    if length(Score_fusion) > nRaw
        Score_fusion(nRaw+1:end) = 0.6 * Score_fusion(nRaw+1:end); % 算子打6折
    end
    
    nFusion = length(VIP_fusion);
    % 规范化结构体预分配，防止空算子时产生异构报错
    FeatureBank_Before = struct('Type',cell(nFusion,1), 'Band1',cell(nFusion,1), ...
        'Band2',cell(nFusion,1), 'Operator',cell(nFusion,1), 'Index',cell(nFusion,1), ...
        'VIP',cell(nFusion,1), 'Score',cell(nFusion,1));

    for i = 1:nFusion
        FeatureBank_Before(i).VIP = VIP_fusion(i);
        FeatureBank_Before(i).Score = Score_fusion(i); 
        if i <= nRaw
            FeatureBank_Before(i).Type = 'Raw';
            FeatureBank_Before(i).Band1 = wavelengths(selected_base_idx(i));
            FeatureBank_Before(i).Band2 = [];
            FeatureBank_Before(i).Operator = 'Raw';
            FeatureBank_Before(i).Index = selected_base_idx(i); 
        else
            opIdx = i - nRaw;
            FeatureBank_Before(i).Type = 'Operator';
            FeatureBank_Before(i).Band1 = OperatorInfo(opIdx).Band1;
            FeatureBank_Before(i).Band2 = OperatorInfo(opIdx).Band2;
            FeatureBank_Before(i).Operator = OperatorInfo(opIdx).Operator;
            FeatureBank_Before(i).Index = []; 
        end
    end
    
    % [修复2] 安全生成保留特征的候选数量，防止越界溢出
    min_feat = min(5, length(VIP_fusion));
    candidate_nums = unique(round(linspace(min_feat, length(VIP_fusion), min(30, length(VIP_fusion)))));
    
    best_rmsecv = inf;
    best_idx = [];
    for k = 1:length(candidate_nums)
        keep_num = candidate_nums(k);
        % 双保险：确保 keep_num 绝对不会超过实际存在的特征数
        keep_num = min(keep_num, length(Score_fusion));
        
        [~,sort_idx] = sort(Score_fusion,'descend'); 
        temp_idx = sort_idx(1:keep_num);
        Xtemp = X_fusion_train(:,temp_idx);
        
        % [修复3] 再次安全限制，防止 maxLV_use 变为 0 导致崩溃
        maxLV_use = max(1, min(maxLV, size(Xtemp,2)));
        
        try
            [~,~,~,~,~,~,MSE] = plsregress(Xtemp, y_train, maxLV_use, 'CV',fold);
            rmsecv = min(sqrt(MSE(2,2:end)));
            if rmsecv < best_rmsecv
                best_rmsecv = rmsecv;
                best_idx = temp_idx;
            end
        catch
            % 如果遇到 CV 划分的极小样本异常则安全跳过
            continue;
        end
    end
    
    % 终极兜底策略：如果因为极低概率原因没选出任何特征，则全保留
    if isempty(best_idx)
        best_idx = 1:length(Score_fusion);
    end
    distilled_idx = best_idx;
    
    %% === ===  局部冗余压缩层 (Local Wavelength Competition, LWC) === ===     
    gap_nm = 12;
    gap_nm_operator = 12;
    cluster_gap = 15; 
    anchor_gap = 30;  
    
    keep_raw = true(length(distilled_idx),1);
    for i = 1:length(distilled_idx)
    %% Stage 1: Raw-LWC
        if ~keep_raw(i); continue; end
        feature_i = FeatureBank_Before(distilled_idx(i));
        for j = i+1:length(distilled_idx)
            if ~keep_raw(j); continue; end
            feature_j = FeatureBank_Before(distilled_idx(j));
            
            if strcmp(feature_i.Type,'Raw') && strcmp(feature_j.Type,'Raw')
                wl_i = feature_i.Band1; wl_j = feature_j.Band1;
                if abs(wl_i - wl_j) < gap_nm
                    score_i = Score_fusion(distilled_idx(i));
                    score_j = Score_fusion(distilled_idx(j));
                    score_i = score_i * Stability(feature_i.Index);
                    score_j = score_j * Stability(feature_j.Index); 
                    if score_i >= score_j
                        keep_raw(j) = false;
                    else
                        keep_raw(i) = false;
                        break;
                    end
                end
            end
        end
    end
    distilled_idx = distilled_idx(keep_raw);
    n_after_raw = length(distilled_idx);
    
    keep_oper = true(length(distilled_idx),1);
    for i = 1:length(distilled_idx)
    %% Stage 2: Operator-LWC
        if ~keep_oper(i); continue; end
        Fi = FeatureBank_Before(distilled_idx(i));
        for j = i+1:length(distilled_idx)
            if ~keep_oper(j); continue; end
            Fj = FeatureBank_Before(distilled_idx(j));
            
            if ~strcmp(Fi.Type,'Raw') && ~strcmp(Fj.Type,'Raw')
                sameType = strcmp(Fi.Type,Fj.Type);
                if sameType
                    cond1 = abs(Fi.Band1 - Fj.Band1) < gap_nm_operator;
                    cond2 = abs(Fi.Band2 - Fj.Band2) < gap_nm_operator;
                    if cond1 && cond2
                        score_i = Score_fusion(distilled_idx(i));
                        score_j = Score_fusion(distilled_idx(j));
                        idx_i_b1 = find(wavelengths == Fi.Band1);
                        idx_i_b2 = find(wavelengths == Fi.Band2);
                        idx_j_b1 = find(wavelengths == Fj.Band1);
                        idx_j_b2 = find(wavelengths == Fj.Band2);
                        if ~isempty(idx_i_b1) && ~isempty(idx_i_b2)
                           score_i = score_i * mean([Stability(idx_i_b1), Stability(idx_i_b2)]);
                        end
                        if ~isempty(idx_j_b1) && ~isempty(idx_j_b2)
                           score_j = score_j * mean([Stability(idx_j_b1), Stability(idx_j_b2)]);
                        end
                        if score_i >= score_j 
                            keep_oper(j)=false; 
                        else 
                            keep_oper(i)=false; 
                            break; 
                        end
                    end
                end
            end
        end
    end
    distilled_idx = distilled_idx(keep_oper);
    n_after_operator = length(distilled_idx);

    keep_pcc = true(length(distilled_idx),1);
    for i = 1:length(distilled_idx)
    %% Stage 3: Peak Cluster Competition (PCC)
        if ~keep_pcc(i); continue; end
        Fi = FeatureBank_Before(distilled_idx(i));
        for j = i+1:length(distilled_idx)
            if ~keep_pcc(j); continue; end
            Fj = FeatureBank_Before(distilled_idx(j));
            
            if ~strcmp(Fi.Type,'Raw') && ~strcmp(Fj.Type,'Raw')
                if strcmp(Fi.Type,Fj.Type)
                    cluster1_match = abs(Fi.Band1 - Fj.Band1) <= cluster_gap;
                    cluster2_match = abs(Fi.Band2 - Fj.Band2) <= cluster_gap;
                    if cluster1_match && cluster2_match
                        score_i = Score_fusion(distilled_idx(i));
                        score_j = Score_fusion(distilled_idx(j));
                        idx_i_b1 = find(wavelengths == Fi.Band1);
                        idx_i_b2 = find(wavelengths == Fi.Band2);
                        idx_j_b1 = find(wavelengths == Fj.Band1);
                        idx_j_b2 = find(wavelengths == Fj.Band2);
                        if ~isempty(idx_i_b1) && ~isempty(idx_i_b2)
                           score_i = score_i * mean([Stability(idx_i_b1), Stability(idx_i_b2)]);
                        end
                        if ~isempty(idx_j_b1) && ~isempty(idx_j_b2)
                           score_j = score_j * mean([Stability(idx_j_b1), Stability(idx_j_b2)]);
                        end
                        if score_i >= score_j 
                            keep_pcc(j)=false; 
                        else 
                            keep_pcc(i)=false; 
                            break; 
                        end
                    end
                end
            end
        end
    end
    distilled_idx = distilled_idx(keep_pcc);
    n_after_pcc = length(distilled_idx);
    
    keep_oac = true(length(distilled_idx),1);
    for i = 1:length(distilled_idx)
    %% Stage 4: Operator Anchor Competition (OAC)
        if ~keep_oac(i); continue; end
        Fi = FeatureBank_Before(distilled_idx(i));
        
        for j = i+1:length(distilled_idx)
            if ~keep_oac(j); continue; end
            Fj = FeatureBank_Before(distilled_idx(j));
            
            if ~strcmp(Fi.Type,'Raw') && ~strcmp(Fj.Type,'Raw') && strcmp(Fi.Type, Fj.Type)
                shared_anchor = false;
                dist_other = inf;
                
                if Fi.Band1 == Fj.Band1
                    shared_anchor = true; dist_other = abs(Fi.Band2 - Fj.Band2);
                elseif Fi.Band2 == Fj.Band2
                    shared_anchor = true; dist_other = abs(Fi.Band1 - Fj.Band1);
                elseif Fi.Band1 == Fj.Band2
                    shared_anchor = true; dist_other = abs(Fi.Band2 - Fj.Band1);
                elseif Fi.Band2 == Fj.Band1
                    shared_anchor = true; dist_other = abs(Fi.Band1 - Fj.Band2);
                end
                
                if shared_anchor && (dist_other < anchor_gap)
                    score_i = Score_fusion(distilled_idx(i));
                    score_j = Score_fusion(distilled_idx(j));
                    
                    idx_i_b1 = find(wavelengths == Fi.Band1);
                    idx_i_b2 = find(wavelengths == Fi.Band2);
                    idx_j_b1 = find(wavelengths == Fj.Band1);
                    idx_j_b2 = find(wavelengths == Fj.Band2);
                    
                    if ~isempty(idx_i_b1) && ~isempty(idx_i_b2)
                       score_i = score_i * mean([Stability(idx_i_b1), Stability(idx_i_b2)]);
                    end
                    if ~isempty(idx_j_b1) && ~isempty(idx_j_b2)
                       score_j = score_j * mean([Stability(idx_j_b1), Stability(idx_j_b2)]);
                    end
                    
                    if score_i >= score_j 
                        keep_oac(j) = false; 
                    else 
                        keep_oac(i) = false; 
                        break; 
                    end
                end
            end
        end
    end
    distilled_idx = distilled_idx(keep_oac);
    n_after_oac = length(distilled_idx);

    %% === === 生成双层统计特征体系 (Dual-Layer Statistics) === ===
    FeatureBank_After = FeatureBank_Before(distilled_idx);
    all_unique_wl = [];
    num_raw = 0;
    num_oper = 0;
    oper_counts = struct('Diff', 0, 'Ratio', 0, 'NDI', 0, 'Sum', 0);
    
    for i = 1:length(FeatureBank_After)
        F = FeatureBank_After(i);
        if strcmp(F.Type, 'Raw')
            all_unique_wl(end+1) = F.Band1;
            num_raw = num_raw + 1;
        else
            all_unique_wl(end+1) = F.Band1;
            all_unique_wl(end+1) = F.Band2;
            num_oper = num_oper + 1;
            if isfield(oper_counts, F.Operator)
                oper_counts.(F.Operator) = oper_counts.(F.Operator) + 1;
            end
        end
    end
    
    KeyWavelengths = unique(all_unique_wl);
    N_KeyWavelengths = length(KeyWavelengths);

    %% Output Statistics
    n_before = length(keep_raw);
    fprintf('\n===== Local Competition Process =====\n');
    fprintf('Before LWC         : %d\n', n_before);
    fprintf('After Raw-LWC      : %d\n', n_after_raw);
    fprintf('After Oper-LWC     : %d\n', n_after_operator);
    fprintf('After PCC          : %d\n', n_after_pcc);
    fprintf('After OAC (Anchor) : %d\n', n_after_oac);
    fprintf('Removed Total      : %d\n', n_before - n_after_oac);
    
    fprintf('\n===== Dual-Layer Feature Statistics =====\n');
    fprintf('Final Features     : %d\n', length(FeatureBank_After));
    fprintf('  - Raw Features   : %d\n', num_raw);
    fprintf('  - Operator Feat. : %d\n', num_oper);
    fprintf('      * Diff       : %d\n', oper_counts.Diff);
    fprintf('      * Ratio      : %d\n', oper_counts.Ratio);
    fprintf('      * NDI        : %d\n', oper_counts.NDI);
    fprintf('      * Sum        : %d\n', oper_counts.Sum);
    fprintf('Unique Wavelengths : %d\n', N_KeyWavelengths);
    fprintf('=========================================\n');

    X_final_train = X_fusion_train(:, distilled_idx);
    X_final_test = X_fusion_test(:, distilled_idx);
    
    %% PHASE 4: VIP二次蒸馏 FINAL PLSR MODELING & EVALUATION
    fprintf('Phase 4: Final PLSR Modeling and Metric Evaluation...\n');
    
    [~, ~, ~, ~, ~, PLS_MSE_final] = plsregress(X_final_train, y_train, min(maxLV, size(X_final_train,2)), 'CV', fold);
    [RMSECV_final, finalLV] = min(sqrt(PLS_MSE_final(2, 2:end)));
    
    [~, ~, ~, ~, beta_final] = plsregress(X_final_train, y_train, finalLV);
    
    y_pred_c = [ones(size(X_final_train,1),1) X_final_train] * beta_final;
    RMSEC = sqrt(mean((y_train - y_pred_c).^2));
    Rc2 = 1 - sum((y_train - y_pred_c).^2) / sum((y_train - mean(y_train)).^2);
    
    y_pred_p = [ones(size(X_final_test,1),1) X_final_test] * beta_final;
    RMSEP = sqrt(mean((y_test - y_pred_p).^2));
    Rp2 = 1 - sum((y_test - y_pred_p).^2) / sum((y_test - mean(y_test)).^2);
    RPD = std(y_test) / RMSEP;
    
    Results.N_features = length(distilled_idx);
    Results.Final_LV = finalLV;
    Results.Metrics.RMSEC = RMSEC;
    Results.Metrics.Rc2 = Rc2;
    Results.Metrics.RMSECV = RMSECV_final;
    Results.Metrics.RMSEP = RMSEP;
    Results.Metrics.Rp2 = Rp2;
    Results.Metrics.RPD = RPD;

    Results.FeatureBank_Before = FeatureBank_Before;
    Results.FeatureBank_After = FeatureBank_After;
    
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

    DistillationReport = struct();
    DistillationReport.KeepIdx = distilled_idx;
    DistillationReport.N_Before = length(FeatureBank_Before);
    DistillationReport.N_After = length(distilled_idx);
    Results.DistillationReport = DistillationReport;
    
    VIP_sort = sort(VIP_fusion,'descend'); 
    VIP_after = VIP_fusion(distilled_idx);
    VIP_after = sort(VIP_after,'descend');
    
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

%% NSGA-II Objective
function F = NSGA_Objective(x, X, y, Stability, maxLV, fold)
    x = double(x > 0.5); 
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

%% NSGA-II OutputFcn
function [state,options,optchanged] = NSGA_OutputFcn(options,state,flag)
global NSGA_HISTORY;
optchanged = false;
switch flag
    case 'init'
        NSGA_HISTORY = struct(); 
        NSGA_HISTORY.bestRMSECV = [];
        NSGA_HISTORY.spread = [];
        NSGA_HISTORY.noImprove = 0;
    case 'iter'
        scores = state.Score;
        [current_best, best_idx] = min(scores(:,1)); 
        
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

%% NSGA-II save_population
function [state,options,optchanged] = save_population(options,state,flag)
    global ALL_SCORE;
    global EVOLUTION_NFEATURE; 
    optchanged = false;
    if strcmp(flag,'init') 
        ALL_SCORE = [];
        EVOLUTION_NFEATURE = [];
    elseif strcmp(flag,'iter')
        ALL_SCORE = [ALL_SCORE; state.Score]; 
        nfeat = sum(state.Population>0.5, 2); 
        EVOLUTION_NFEATURE(end+1) = mean(nfeat);
    end
end