%% Portfolio Returns and Descriptive Stats


clear
close all
clc


% ******* Initializing parameters
rng(0)
rollWin = 120;
strt_date = datetime('01/07/2014','InputFormat','dd/MM/uuuu');
M = 1E4;
p = [90 99];
 

% ******* Importing stock data to prices matrix
filename = 'MyStock_merge.csv';
stockData = readtable(filename);
prices = stockData{:,2:end};
Dates = stockData{2:end,1};
Dates = datetime(Dates, 'InputFormat', 'dd/MM/yyyy');

% ******* Calculating log returns and forming the return matrix along with dates
logRet = log(prices(2:end,:) ./ prices(1:end-1,:));


% ******* Assigning the weights; Equally weighted
NObs = size(logRet,1);
NAsset = size(logRet,2);
w = ones(NAsset,1) / NAsset;

% ******* Portfolio return
ret_Port = logRet*w;

% ******* Descriptive Stats
min_val = min(ret_Port)
max_val = max(ret_Port)
mean_val = mean(ret_Port);
std_dev = std(ret_Port);
skew_val = skewness(ret_Port);
kurt_val = kurtosis(ret_Port) - 3;
n = length(ret_Port);
jb_stat = (n/6) * (skew_val^2 + (kurt_val^2)/4);
crit_value = chi2inv(0.95, 2);

if jb_stat > crit_value
    jb_inference = 'Reject Normality (Non-Normal)';
else
    jb_inference = 'Fail to Reject Normality';
end


fprintf('-------------------------------------\n');
fprintf('Descriptive Analysis of Log Returns\n');
fprintf('-------------------------------------\n');
fprintf('Number of Observations:  %.4f\n', NObs);
fprintf('Mean:                    %.4f\n', mean_val);
fprintf('Standard Deviation:      %.4f\n', std_dev);
fprintf('Skewness:                %.4f\n', skew_val);
fprintf('Excess Kurtosis:         %.4f\n', kurt_val);
fprintf('Jarque-Bera Stat:        %.4f\n', jb_stat);
fprintf('Critical Value:          %.4f\n', crit_value);
fprintf('Jarque-Bera Test:        %s\n', jb_inference);



figure;

% 1. Time Series Plot
%subplot(2,1,1); % First subplot

plot(Dates, ret_Port, 'b-', 'LineWidth', 1.5);
datetick('x', 'yyyy'); % Format x-axis as years
xlabel('Time');
ylabel('Log Returns');
title('Portfolio Log Returns');
grid on;

plot(Dates, ret_Port, 'b-', 'LineWidth', 1.5);
datetick('x', 'yyyy'); % Format x-axis as years
xlabel('Time');
ylabel('Log Returns');
title('Portfolio Log Returns');
grid on;


figure;
% 2. QQ Plot
%subplot(2,1,2); % Second subplot
qqplot(ret_Port);
xlabel('Theoretical Quantiles'); % X-axis label
ylabel('Log Return Empirical Quantiles'); 
title('QQ Plot (Normality Check)');
grid on;

nobs = length(ret_Port)
symbol = "Portfolio"
Z = (ret_Port - mean(ret_Port)) / std(ret_Port); % Standardized Returns

%% ============================
%  Test for Autocorrelation
% ============================
maxlags = 20;
[acfValues,  ~, ~] = autocorr(Z, maxlags )
[acfSqValues,  ~, ~] = autocorr(Z.^2, maxlags )
[acfAbsValue,  ~, ~] = autocorr(abs(Z), maxlags )

% Create Summary Table
Synthesis = table(acfValues, acfSqValues, acfAbsValue);
Synthesis.Properties.VariableNames = {'ACF Ret', 'ACF SQ. Ret', 'ACF Abs. Ret'};
disp(Synthesis);

h = figure('Color',[1 1 1]);
autocorr(Z, maxlags);
xlim([0.5, maxlags+0.5])
xlabel('Lag','interpreter','latex')
ylabel('Sample autocorrelation','interpreter','latex')
title([symbol ': Autocorrelation of Log-Returns'], 'interpreter', 'latex');

h = figure('Color',[1 1 1]);
autocorr(Z.^2, maxlags);
xlim([0.5, maxlags+0.5])
xlabel('Lag','interpreter','latex')
ylabel('Sample autocorrelation','interpreter','latex')
title([symbol ': Autocorrelation of Squared Log-Returns'], 'interpreter', 'latex');

h = figure('Color',[1 1 1]);
autocorr(abs(Z), maxlags);
xlim([0.5, maxlags+0.5])
xlabel('Lag','interpreter','latex')
ylabel('Sample autocorrelation','interpreter','latex')
title([symbol ': Autocorrelation of Absolute Log-Returns'], 'interpreter', 'latex');

%% VaR Estimation Parametric

% ******* Checking the row number with start date = 30/06/2014
dates = datetime(stockData{:,1},'Inputformat', 'dd/MM/uuuu');
rowNum = find(dates == strt_date) - 1;


% ******* Calculating rolling portfolio means, std dev
RpG = zeros(NObs-rowNum+1,2);
%vAr_G = zeros(NObs-rowNum+1, length(p));

for i=0:NObs-rowNum
    
    RpG(i+1,1) = mean(logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end)*w);     % row 4-123, 5-124,....
    RpG(i+1,2) = std(logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end)*w);

end

% ******* Calc VaRs at 90% and 99%
p1 = 1 - p/100; 
vAr_G = -(RpG(:,1) + norminv(p1).*RpG(:,2));

%table_data = table(Dates(rowNum:end,1), vAr_G(:,1),vAr_G(:,2), 'VariableNames', {'Date', 'VaR_G@90%', 'VaR_G@99%'})

% ******* Actual portfolio returns
RpE = logRet(rowNum:end,:)*w;


% ******* Calculating VaR Violations
viol_G = [(RpE <-vAr_G(:,1))*1 (RpE <-vAr_G(:,2))*1 ];
Tviol_G = sum([(RpE <-vAr_G(:,1))*1 (RpE <-vAr_G(:,2))*1 ]);
vArObs = size(vAr_G,1);


fprintf('\n\n--------------------------------------------\n');
fprintf('VaR Estimations for 6 month Rolling Window\n');
fprintf('--------------------------------------------\n');
fprintf('\n*** VaR Estimation via Gaussian (Top-Down) *** \n');
fprintf('------------------------------------------------\n');
fprintf('The number of total obs: %d\n', vArObs);
fprintf('VaR violations at 90 percent: %d\n', Tviol_G(1));
fprintf('VaR violations at 99 percent: %d\n', Tviol_G(2));

%% Var Estimation by Historical Estimation (Non Parametric)

% ******* Calc VaRs at 90% and 99%

vAr_HS = zeros(NObs-rowNum+1,2);

for i=0:NObs-rowNum
    
    vAr_HS(i+1,:) = -prctile(logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end)*w, (100-p));   % row 4-123, 5-124,....
    
end

%table_data = table(Dates(rowNum:end,1), vAr_HS(:,1),vAr_HS(:,2), 'VariableNames', {'Date', 'VaR_HS@90%', 'VaR_HS@99%'})

% ******* Calculating VaR Violations
viol_HS = [(RpE <-vAr_HS(:,1))*1 (RpE <-vAr_HS(:,2))*1 ];
Tviol_HS = sum([(RpE <-vAr_HS(:,1))*1 (RpE <-vAr_HS(:,2))*1 ]);
vArObs = size(vAr_HS,1);

fprintf('\n*** VaR Estimation via Historical (Non-Paramteric) *** \n');
fprintf('---------------------------------------------------------\n');
fprintf('The number of total obs: %d\n', vArObs);
fprintf('VaR violations at 90 percent: %d\n', Tviol_HS(1));
fprintf('VaR violations at 99 percent: %d\n', Tviol_HS(2));




%% Var Estimation by Monte Carlo Simulation


% ******* Calculating rolling means, Portfolio means, and generating VaRs via MC simulation

meanRet = zeros(NObs-rowNum+1,NAsset);
vAr_MC = zeros(NObs-rowNum+1, length(p));

for i=0:NObs-rowNum
    rng(1234)
    meanRet(i+1,:) = mean(logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end));     % row 4-123, 5-124,....
    covarRoll = cov(logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end));
    simRi = mvnrnd(meanRet(i+1,:), covarRoll,M);
    simRP = log(exp(simRi)*w);
    vAr_MC(i+1,:) = -prctile(simRP, 100 - p);
    
     
end

%table_data = table(Dates(rowNum:end,1), vAr_MC(:,1),vAr_MC(:,2), 'VariableNames', {'Date', 'VaR_MC@90%', 'VaR_MC@99%'})

% ******* Actual portfolio returns
% RpE = log(exp(logRet(rowNum:end,:))*w);
% RpA = logRet(rowNum:end,:)*w;


% ******* Calculating VaR Violations
viol_MCE = [(RpE <-vAr_MC(:,1))*1 (RpE <-vAr_MC(:,2))*1 ];
Tviol_MCE = sum([(RpE <-vAr_MC(:,1))*1 (RpE <-vAr_MC(:,2))*1 ]);
%viol_MCA = sum([(RpA <-vAr_MC(:,1))*1 (RpA <-vAr_MC(:,2))*1 ]);
vArObs = size(vAr_MC,1);

fprintf('\n *** VaR Estimation via Monte Carlo Simulation (Bottom-Up) *** \n');
fprintf('-----------------------------------------------------------------\n');
fprintf('The number of total obs: %d\n', vArObs);
fprintf('VaR violations at 90 percent: %d\n', Tviol_MCE(1));
fprintf('VaR violations at 99 percent: %d\n', Tviol_MCE(2));


%% VaR estimation via Bootstrapping


vAr_BS = zeros(NObs-rowNum+1,size(p,2));
sRp = zeros(rollWin,1);


% ******* Generating bootstrapped VaRs

for i=0:NObs-rowNum
    
    sRp = logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end)*w;   % row 4-123, 5-124,....
    Nb = 1000;
    bsVar = zeros(Nb,2);
    for j=1:Nb
        indices = randi([1, rollWin], 1, rollWin);
        %indices = generate_numbers_with_repeats(1,rollWin,rollWin) ;
        bsVar(j,:) = -prctile(sRp(indices), 100 - p);
    end
    vAr_BS(i+1,:) = mean(bsVar);
end

%table_data = table(Dates(rowNum:end,1), vAr_BS(:,1),vAr_BS(:,2), 'VariableNames', {'Date', 'VaR_BS@90%', 'VaR_BS@99%'})

% ******* Calculating VaR Violations
viol_BS = [(RpE <-vAr_BS(:,1))*1 (RpE <-vAr_BS(:,2))*1 ];
Tviol_BS = sum([(RpE <-vAr_BS(:,1))*1 (RpE <-vAr_BS(:,2))*1 ]);
%viol_MCA = sum([(RpA <-vAr_MC(:,1))*1 (RpA <-vAr_MC(:,2))*1 ]);
vArObs = size(vAr_BS,1);

fprintf('\n *** VaR Estimation via Bootstrapping (Non-Paramteric) *** \n');
fprintf('-------------------------------------------------------------\n');
fprintf('The number of total obs: %d\n', vArObs);
fprintf('VaR violations at 90 percent: %d\n', Tviol_BS(1));
fprintf('VaR violations at 99 percent: %d\n', Tviol_BS(2));

%% VaR estimation via Block Bootstrapping
vAr_BBS = zeros(NObs-rowNum+1,size(p,2));
sRp = zeros(rollWin,1);
Bsz = 2;
Blockn = length(sRp) / Bsz;

for i=0:NObs-rowNum
    Nb = 1000;   
    sRp = logRet(rowNum-rollWin+i:(rowNum-1)+i,1:end)*w;   % row 4-123, 5-124,....
    blocks = reshape(sRp, [Blockn,Bsz])';
    bbsVar = zeros(Nb,Bsz);
    tmpOpt = bootstrp(Nb, @(x)x', blocks);
    bbsVar = -prctile(tmpOpt, 100 - p,2);
    vAr_BBS(i+1,:) = mean(bbsVar);
end

%table_data = table(Dates(rowNum:end,1), vAr_BS(:,1),vAr_BS(:,2), 'VariableNames', {'Date', 'VaR_BS@90%', 'VaR_BS@99%'})

% ******* Calculating VaR Violations
viol_BBS = [(RpE <-vAr_BBS(:,1))*1 (RpE <-vAr_BBS(:,2))*1 ];
Tviol_BBS = sum([(RpE <-vAr_BBS(:,1))*1 (RpE <-vAr_BBS(:,2))*1 ]);
vArObs = size(vAr_BBS,1);

fprintf('\n *** VaR Estimation via Block Bootstrapping (Non-Paramteric) *** \n');
fprintf('-------------------------------------------------------------\n');
fprintf('The number of total obs: %d\n', vArObs);
fprintf('VaR violations at 90 percent: %d\n', Tviol_BBS(1));
fprintf('VaR violations at 99 percent: %d\n', Tviol_BBS(2));

%% Time Series of various VaR estimates

% ************************************************************************
% VaR Estimation via Gaussian (Top-Down)

table_data = table(Dates(rowNum:end,1), vAr_G(:,1), vAr_G(:,2), ...
    'VariableNames', {'Date', 'VaR_G@90%', 'VaR_G@99%'});

% Plot the time series
figure;
plot(table_data.Date, table_data.("VaR_G@90%"), 'b', 'LineWidth', 1.5); % Blue line for 90% VaR
hold on;
plot(table_data.Date, table_data.("VaR_G@99%"), 'r', 'LineWidth', 1.5); % Red line for 99% VaR

% Formatting
xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Parametric Gaussian VaR  Estimations (90% & 99%)');
legend('VaR @ 90%', 'VaR @ 99%', 'Location', 'Best');
grid on;
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis with years

% Release hold
hold off;



% ************************************************************************
% VaR Estimation via Historical (Non-Paramteric)

table_data = table(Dates(rowNum:end,1), vAr_HS(:,1), vAr_HS(:,2), ...
    'VariableNames', {'Date', 'VaR_HS@90%', 'VaR_HS@99%'});

% Plot the time series
figure;
plot(table_data.Date, table_data.("VaR_HS@90%"), 'b', 'LineWidth', 1.5); % Blue line for 90% VaR
hold on;
plot(table_data.Date, table_data.("VaR_HS@99%"), 'r', 'LineWidth', 1.5); % Red line for 99% VaR

% Formatting
xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Historical Simulation VaR (90% & 99%)');
legend('VaR @ 90%', 'VaR @ 99%', 'Location', 'Best');
grid on;
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis with years

% Release hold
hold off;



% ************************************************************************
%VaR Estimation via Monte Carlo Simulation (Bottom-Up)

table_data = table(Dates(rowNum:end,1), vAr_MC(:,1), vAr_MC(:,2), ...
    'VariableNames', {'Date', 'VaR_MC@90%', 'VaR_MC@99%'});

% Plot the time series
figure;
plot(table_data.Date, table_data.("VaR_MC@90%"), 'b', 'LineWidth', 1.5); % Blue line for 90% VaR
hold on;
plot(table_data.Date, table_data.("VaR_MC@99%"), 'r', 'LineWidth', 1.5); % Red line for 99% VaR

% Formatting
xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Monte Carlo Simulated VaR (90% & 99%)');
legend('VaR @ 90%', 'VaR @ 99%', 'Location', 'Best');
grid on;
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis with years

% Release hold
hold off;


% ************************************************************************
% VaR Estimation via Bootstrapping (Non-Paramteric)
table_data = table(Dates(rowNum:end,1), vAr_BS(:,1), vAr_BS(:,2), ...
    'VariableNames', {'Date', 'VaR_BS@90%', 'VaR_BS@99%'});

% Plot the time series
figure;
plot(table_data.Date, table_data.("VaR_BS@90%"), 'b', 'LineWidth', 1.5); % Blue line for 90% VaR
hold on;
plot(table_data.Date, table_data.("VaR_BS@99%"), 'r', 'LineWidth', 1.5); % Red line for 99% VaR

% Formatting
xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Bootstrapped VaR (90% & 99%)');
legend('VaR @ 90%', 'VaR @ 99%', 'Location', 'Best');
grid on;
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis with years

% Release hold
hold off;

% ************************************************************************
% VaR Estimation via Block Bootstrapping (Non-Paramteric)
table_data = table(Dates(rowNum:end,1), vAr_BBS(:,1), vAr_BBS(:,2), ...
    'VariableNames', {'Date', 'VaR_BBS@90%', 'VaR_BBS@99%'});

% Plot the time series
figure;
plot(table_data.Date, table_data.("VaR_BBS@90%"), 'b', 'LineWidth', 1.5); % Blue line for 90% VaR
hold on;
plot(table_data.Date, table_data.("VaR_BBS@99%"), 'r', 'LineWidth', 1.5); % Red line for 99% VaR

% Formatting
xlabel('Date');
ylabel('Value at Risk (VaR)');
title('Block Bootstrapped VaR (90% & 99%)');
legend('VaR @ 90%', 'VaR @ 99%', 'Location', 'Best');
grid on;
datetick('x', 'yyyy', 'keeplimits'); % Format x-axis with years

% Release hold
hold off;


%% Analysis

% ******* Analysis of Estimated VaRs

varNames = {'Gaussian', 'Historical', 'Monte Carlo', 'Bootstrapping', 'Block Bootstrapping'};

Var90pct = [vAr_G(:,1) vAr_HS(:,1) vAr_MC(:,1) vAr_BS(:,1) vAr_BBS(:,1)];
Var99pct = [vAr_G(:,2) vAr_HS(:,2) vAr_MC(:,2) vAr_BS(:,2) vAr_BBS(:,2)];

corr_mat90 = corr(Var90pct);
corr_mat99 = corr(Var99pct);

figure;
imagesc(corr_mat90); % Display as an image
colorbar; % Add color legend
title('Correlation Matrix of 90% VaR');
xticks(1:5); yticks(1:5); % Label axes
xticklabels(varNames); yticklabels(varNames);
colormap(jet); % Set color map

figure;
imagesc(corr_mat99); % Display as an image
colorbar; % Add color legend
title('Correlation Matrix of 99% VaR');
xticks(1:5); yticks(1:5); % Label axes
xticklabels(varNames); yticklabels(varNames);
colormap(jet); % Set color map

%% Kupiec, Conditional Convergence and Distributional Tests


viol90pct = [viol_G(:,1) viol_HS(:,1) viol_MCE(:,1) viol_BS(:,1) viol_BBS(:,1)];
viol99pct = [viol_G(:,2) viol_HS(:,2) viol_MCE(:,2) viol_BS(:,2) viol_BBS(:,2)];

fprintf('\n\n-------------------------------------\n');
fprintf('90pct VaR Violations - Backtest Results\n');
fprintf('-------------------------------------\n');
backtest_var(viol90pct, .9)

fprintf('-------------------------------------\n');
fprintf('99pct VaR Violations - Backtest Results\n');
fprintf('-------------------------------------\n');
backtest_var(viol99pct, .99)

