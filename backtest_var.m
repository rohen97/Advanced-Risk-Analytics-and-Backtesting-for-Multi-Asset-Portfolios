function backtest_var(violations, confidence_level)
    % Number of observations and models
    [n, m] = size(violations); 

    % Define expected violation rate
    expected_violations_rate = 1 - confidence_level; 

    % Store results
    kupiec_p_values = zeros(1, m);
    conditional_coverage_p_values = zeros(1, m);
    ks_p_values = zeros(1, m);

    for i = 1:m
        % Extract violations for the current model
        model_violations = violations(:,i); 

        % --- Kupiec Test (Proportion of Failures) ---
        % Kupiec test compares the number of violations against the expected violations
        num_violations = sum(model_violations);
        kupiec_p_values(i) = kupiec_test(num_violations, n, expected_violations_rate);

        % --- Conditional Coverage Test ---
        % This test checks if violations are independent over time (Markov property)
        conditional_coverage_p_values(i) = conditional_coverage_test(model_violations, num_violations, n, expected_violations_rate);

        % --- KS Test ---
        % Kolmogorov-Smirnov test checks if the empirical distribution matches uniform
        ks_p_values(i) = ks_test(model_violations);
    end

    % Display results
    fprintf('--- Backtesting VaR Models ---\n');
    for i = 1:m
        fprintf('Model %d:\n', i);
        fprintf('  Kupiec Test p-value: %.4f\n', kupiec_p_values(i));
        fprintf('  Conditional Coverage Test p-value: %.4f\n', conditional_coverage_p_values(i));
        fprintf('  KS Test p-value: %.4f\n', ks_p_values(i));
    end
end

% Kupiec Test (Proportion of Failures)
function p_value = kupiec_test(num_violations, n, expected_rate)
    observed_rate = num_violations / n;
    T = n;
    T0 = n - num_violations;
    T1 = num_violations;
    
    %The next code does not work due to small values approximated:
    %lr = -2 * log((1 - expected_rate)^T0 * expected_rate^T1 / (1-T1/T)^T0 * (T1/T)^T1 );

    %modified version:
    lr = -2 * log((1 - expected_rate)^T0) + -2 * log(expected_rate^T1) + 2 * (log((1-T1/T)^T0) + log((T1/T)^T1));
    p_value = 1 - chi2cdf(lr, 1);  % Chi-squared distribution with 1 degree of freedom
end

% Conditional Coverage Test (Markov Transition Matrix)
function p_value = conditional_coverage_test(model_violations, num_violations, n, expected_rate)
    % We are checking if there are violations at time t and t-1
    violations_lag = model_violations(2:end);  % Shift by 1 for lag
    transitions = [model_violations(1:end-1), violations_lag];  % Create the transition matrix
    % Count transitions from 0 to 0, 0 to 1, 1 to 1, and 1 to 0
    transition_counts = [sum(transitions(:,1) == 0 & transitions(:,2) == 0), ...
                         sum(transitions(:,1) == 0 & transitions(:,2) == 1), ...
                         sum(transitions(:,1) == 1 & transitions(:,2) == 1), ...
                         sum(transitions(:,1) == 1 & transitions(:,2) == 0)];

    observed_rate = num_violations / n;
    N0 = n - num_violations;
    N1 = num_violations;
    LogL_p = log((1 - expected_rate)^N0) + log(expected_rate^N1);
    
    T00 = transition_counts(1);
    T01 = transition_counts(2);
    T11 = transition_counts(3);
    T10 = transition_counts(4);
    pi01 = T01/(T00+T01);
    pi11 = T11/(T10+T11);
    LogL_Pi = log((1-pi01)^T00) + log(pi01^T01) + log((1-pi11)^T10) + log(pi11^T11);

    lr = -2*LogL_p + 2*LogL_Pi;

    p_value = 1 - chi2cdf(lr, 1);
end

% KS Test (Kolmogorov-Smirnov)
function p_value = ks_test(model_violations)
    % Kolmogorov-Smirnov test compares empirical distribution with uniform
    [f, x] = ecdf(model_violations);  % Empirical cumulative distribution
    [~, p_value] = kstest(f, 'CDF', makedist('Uniform'));  % Compare to uniform distribution
end
