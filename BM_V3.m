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
    global EVOLUTION_NFEATURE;    
    EVOLUTION_NFEATURE = [];      
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
        keep_ratio = 0.5;   % 波长保留比例
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
        Kmin = 50;
        Kmax = 100;
        K = round(0.1*P);
        K = max(Kmin,min(Kmax,K));
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); 

        otherwise 
        [stab_sort,idx_sort] = sort(Stability,'descend');
        cum_stab = cumsum(stab_sort);
        cum_stab = cum_stab/cum_stab(end);
        N_keep = find(cum_stab>=0.9,1);
        N_keep = max(N_keep,30);
        candidate_idx = idx_sort(1:min(N_keep, length(idx_sort))); 
    end
    
    %% --- NSGA-II Feature Optimization ---
    X_candidate = X_train(:,candidate_idx);
    Stability_candidate = Stability(candidate_idx);
    nVar = length(candidate_idx);
    LB = zeros(1,nVar);
    UB = ones(1,nVar);
    
    % === === NSGA-II 寻优空间复杂度开关 === === 
    fprintf('NSGA-II Mode selected: %s\n', upper(nsga_mode));
    switch lower(nsga_mode)
        case 'light'
            opts = optimoptions('gamultiobj', 'PopulationSize',20, ...
                'MaxGenerations',10, 'Display','iter', 'OutputFcn',@save_population); 
        case 'medium'
            opts = optimoptions('gamultiobj', 'PopulationSize',50, ...
                'MaxGenerations',30, 'Display','iter', 'OutputFcn',@save_population);
        case 'heavy'
            opts = optimoptions('gamultiobj', 'PopulationSize',100, ...
                'MaxGenerations',80, 'Display','iter', 'OutputFcn',@save_population);
        case 'auto'
            opts = optimoptions('gamultiobj',...
                'PopulationSize',100,...
                'MaxGenerations',200,...
                'OutputFcn',{@NSGA_OutputFcn, @save_population},...
                'Display','iter');
        otherwise
            error('未知的 nsga_mode。请使用 light, medium, heavy 或 auto');
    end
    
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
    end
    fprintf('Selected wavelengths after NSGA-II: %d\n', length(selected_base_idx));
    
    Results.Pareto.X = xPareto;
    Results.Pareto.F = fPareto;
    
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
    fprintf('Phase 2: Operator Derivation (Diff, Ratio, NDI, Sum)...\n');
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
            
            % === === 算子自由组合开关 === === 
            if op_switch(1) == 1
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
    
    [~, ~, ~, ~, ~, PLS_MSE_fusion] = plsregress(X_fusion_train, y_train, min(maxLV, size(X_fusion_train,2)), 'CV', fold);
    [~, optLV_fusion] = min(PLS_MSE_fusion(2, 2:end));
    
    VIP_fusion = calculate_VIP(X_fusion_train, y_train, optLV_fusion);
    
    nRaw = length(selected_base_idx);
    nFusion = length(VIP_fusion);
    FeatureBank_Before = struct('Type',cell(nFusion,1), 'Band1',cell(nFusion,1), ...
        'Band2',cell(nFusion,1), 'Operator',cell(nFusion,1), 'VIP',cell(nFusion,1));
        
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
    
    %% === === 新增：包含 Raw 在内的各特征家族内部 Top 5% 独立竞争保留机制 === ===
    keep_mask = true(nFusion, 1);
    
    % 将 'Raw' 也加入了特征类型的公平竞争池
    feature_types = {'Raw', 'Diff', 'Sum', 'Ratio', 'NDI'};
    
    for op_idx = 1:length(feature_types)
        op_name = feature_types{op_idx};
        
        % 寻找属于该类型的所有特征索引 (从 1 扫到尾)
        curr_indices = [];
        for i = 1 : nFusion
            if strcmp(FeatureBank_Before(i).Operator, op_name)
                curr_indices(end+1) = i;
            end
        end
        
        if ~isempty(curr_indices)
            curr_vips = VIP_fusion(curr_indices);
            [~, sort_pos] = sort(curr_vips, 'descend');
            
            % 强制保留 Top 5%，且至少保留1个避免该类型绝嗣
            keep_count = max(1, round(1 * length(curr_indices)));
            
            % 标记该类型中被淘汰的特征
            discard_pos = sort_pos(keep_count+1 : end);
            discard_indices = curr_indices(discard_pos);
            keep_mask(discard_indices) = false;
        end
    end
    
    % 实施物理剔除
    valid_idx = find(keep_mask);
    X_fusion_train = X_fusion_train(:, valid_idx);
    X_fusion_test = X_fusion_test(:, valid_idx);
    VIP_fusion = VIP_fusion(valid_idx);
    FeatureBank_Before = FeatureBank_Before(valid_idx);
    nFusion = length(valid_idx); % 更新剩余总数
    
    nRaw_retained = sum(strcmp({FeatureBank_Before.Type}, 'Raw'));
    nOper_retained = nFusion - nRaw_retained;
    fprintf('Top 5%% Retention completed: %d raw features and %d operator features retained.\n', ...
        nRaw_retained, nOper_retained);
    %% === === ============================================== === ===
    
    % VIP 二次蒸馏：基于保留下来的精英池再次进行组合择优
    min_feat = min(5, nFusion);
    num_steps = min(30, nFusion);
    if num_steps > 0
        candidate_nums = unique(round(linspace(min_feat, nFusion, num_steps)));
    else
        candidate_nums = [];
    end

    best_rmsecv = inf;
    best_idx = [];
    for k = 1:length(candidate_nums)
        keep_num = candidate_nums(k);
        [~,sort_idx] = sort(VIP_fusion,'descend');
        temp_idx = sort_idx(1:keep_num);
        Xtemp = X_fusion_train(:,temp_idx);
        maxLV_use = max(1, min(maxLV, size(Xtemp,2))); 
        try
            [~,~,~,~,~,~,MSE] = plsregress(Xtemp, y_train, maxLV_use, 'CV',fold);
            rmsecv = min(sqrt(MSE(2,2:end)));
            if rmsecv < best_rmsecv
                best_rmsecv = rmsecv;
                best_idx = temp_idx;
            end
        catch
            continue;
        end
    end
    distilled_idx = best_idx;
    if isempty(distilled_idx)
        distilled_idx = 1:nFusion; % 极限兜底
    end

    %% === ===  局部冗余压缩层 (Local Wavelength Competition, LWC) === === 
    keep = true(length(distilled_idx),1);
    gap_nm = 3; % 比较12nm间隔波长的VIP值
    
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

    % === === 最终保留波长 === === 
    FeatureBank_After = FeatureBank_Before(distilled_idx);
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

    %% --- Phase 4.1: VIP 二次蒸馏后的特征保留 结构体
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

    for i = 1:length(distilled_idx) 
        DistillationReport.KeepNames{i} = featureName(FeatureBank_Before(distilled_idx(i)));
    end
    for i = 1:length(remove_idx) 
        DistillationReport.RemoveNames{i} = featureName(FeatureBank_Before(remove_idx(i)));
    end
    
    Results.DistillationReport = DistillationReport;
    
    VIP_sort = sort(VIP_fusion,'descend'); 
    VIP_after = VIP_fusion(distilled_idx);
    VIP_after = sort(VIP_after,'descend');
    
    %% --- --- 二次蒸馏结果可视化绘图区 --- --- 
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

%% --- NSGA_OutputFcn ---  
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

%% Output Function （OutputFcn）
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