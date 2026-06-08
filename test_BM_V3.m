clc;
clear;
close all;
addpath(genpath(pwd));
%% 玉米数据
% load corn.mat
% X_raw = mp6spec.data;
% y = propvals.data(:,1);
% wl_NIR = 1100:2:2498;
% datatype = 'corn1';

%% SIMUIN 数据
load CAT3_SIMUI_DATA.mat;
X_raw = X_Type1_Additive;
wl_NIR = 1:size(X_raw, 2);
datatype = 'SIMUIN_ADD';

%% 汽油数据
% load viscga.mat;
% load density4052.mat;
% load freeze.mat;
% load tot_aromatics.mat;
% X_raw = X;
% wl_NIR = wl;
% datatype = 'density4052';

%% 参数
maxLV = 15;
fold  = 5;
nRuns = 10;     % 自定义运行次数

%% 循环测试
for run = 1:nRuns

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('Run %d / %d\n',run,nRuns);
    fprintf('========================================\n');

    rng(run);

    cv = cvpartition(length(y),'HoldOut',0.25);

    trainIdx = training(cv);
    testIdx  = test(cv);

    X_train = X_raw(trainIdx,:);
    y_train = y(trainIdx);

    X_test = X_raw(testIdx,:);
    y_test = y(testIdx);

    Results = BM_V3(...
        X_train,...
        y_train,...
        X_test,...
        y_test,...
        wl_NIR,...
        maxLV,...
        fold);

    fprintf('\nFinal Metrics:\n');

    fprintf('Rp2    = %.4f\n',Results.Metrics.Rp2);
    fprintf('RMSEP  = %.4f\n',Results.Metrics.RMSEP);
    fprintf('RPD    = %.4f\n',Results.Metrics.RPD);

    if isfield(Results,'Selected_Wavelengths')

        fprintf('\nSelected Wavelengths:\n');

        disp(Results.Selected_Wavelengths);

    end

end