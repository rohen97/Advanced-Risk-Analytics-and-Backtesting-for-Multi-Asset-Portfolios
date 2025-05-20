% ======================================================
% PORTFOLIO ANALYSIS: EQUALLY-WEIGHTED, RISK PARITY, MAX DIVERSIFICATION
% ======================================================
clear; close all; clc;

%% Load Data from CSV
filename = 'MyStock_merge.csv'; 
dataTable = readtable(filename); 

% Extract dates, prices, and tickers
dates = dataTable{:, 1};           
prices = table2array(dataTable(:, 2:7)); 
tickers = dataTable.Properties.VariableNames(2:7); 

%% Split Data into Training/Testing
splitIdx = floor(size(prices, 1)/2);
trainPrices = prices(1:splitIdx, :);
testPrices = prices(splitIdx+1:end, :);

%% Compute Log-Returns
trainLogRet = diff(log(trainPrices));
testLogRet = diff(log(testPrices));

NAsset = size(prices, 2); 

% Estimate Mean Vector and Covariance Matrix
MeanV = mean(trainLogRet)';
Sigma = cov(trainLogRet);

% Handle functions 
sg2p = @(x, Sigma) x' * Sigma * x;
MVaR = @(x, Sigma) Sigma * x / sqrt(sg2p(x, Sigma));
CVaR = @(x, Sigma) x .* MVaR(x, Sigma);
Conv = @(x, Sigma) sqrt(x.^2 .* diag(Sigma) - (x .* CVaR(x, Sigma)).^2);
Divers = @(x, Sigma) x'*diag(Sigma).^0.5/sqrt(sg2p(x, Sigma));

%% Portfolio Construction
% 1. Equally-Weighted Portfolio
w_eq = ones(NAsset, 1) / NAsset;

% 2. Risk Parity Portfolio
x0 = w_eq;
w_RP = fmincon(@(x) std(CVaR(x, Sigma)), x0, [], [], ones(1, NAsset), 1, zeros(NAsset, 1), ones(NAsset, 1));

% 3. Maximum Diversification Portfolio 
x0 = w_RP;
w_MD = fmincon(@(x) -Divers(x, Sigma), x0, [], [], ones(1, NAsset), 1, zeros(NAsset, 1), ones(NAsset, 1));

% Backtest Portfolios
portRet_EQ = testLogRet * w_eq;
portRet_RP = testLogRet * w_RP;
portRet_MD = testLogRet * w_MD;

%% Compute Performance Metrics
% Sharpe Ratio (assuming zero risk-free rate)
sharpe_EQ = mean(portRet_EQ) / std(portRet_EQ);
sharpe_RP = mean(portRet_RP) / std(portRet_RP);
sharpe_MD = mean(portRet_MD) / std(portRet_MD);

% Maximum Drawdown
cumRet_EQ = cumprod(1 + portRet_EQ);
cumRet_RP = cumprod(1 + portRet_RP);
cumRet_MD = cumprod(1 + portRet_MD);

drawdown_EQ = maxdrawdown(cumRet_EQ);
drawdown_RP = maxdrawdown(cumRet_RP);
drawdown_MD = maxdrawdown(cumRet_MD);

% VaR Violations (95% confidence)
VaR_95_EQ = norminv(0.95) * std(portRet_EQ);
violations_EQ = sum(portRet_EQ < -VaR_95_EQ);

VaR_95_RP = norminv(0.95) * std(portRet_RP);
violations_RP = sum(portRet_RP < -VaR_95_RP);

VaR_95_MD = norminv(0.95) * std(portRet_MD);
violations_MD = sum(portRet_MD < -VaR_95_MD);

%% Display Results
fprintf('===== Performance Metrics =====\n');
fprintf('Equally-Weighted Portfolio:\n');
fprintf('  Sharpe Ratio: %.4f\n', sharpe_EQ);
fprintf('  Max Drawdown: %.4f\n', drawdown_EQ);
fprintf('  VaR Violations: %d\n\n', violations_EQ);

fprintf('Risk Parity Portfolio:\n');
fprintf('  Sharpe Ratio: %.4f\n', sharpe_RP);
fprintf('  Max Drawdown: %.4f\n', drawdown_RP);
fprintf('  VaR Violations: %d\n\n', violations_RP);

fprintf('Maximum Diversification Portfolio:\n');
fprintf('  Sharpe Ratio: %.4f\n', sharpe_MD);
fprintf('  Max Drawdown: %.4f\n', drawdown_MD);
fprintf('  VaR Violations: %d\n\n', violations_MD);

%% Plot Portfolio Weights
figure('Color', [1 1 1]);
bar([w_eq, w_RP, w_MD]);
legend('Equally-Weighted', 'Risk Parity', 'Max Diversification', 'Location', 'best');
title('Portfolio Weights');
xticklabels(tickers);
xlabel('Assets');
ylabel('Weight');
grid on;

% Save figure
print('-dpng', 'Portfolio_Weights.png');
