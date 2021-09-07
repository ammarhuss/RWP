

%% Author
%........................................................................
% @Author: Hussein A. Ammar,
% @Email: hussein.ammar@mail.utoronto.ca, hussein.ammar@live.com                       
% @Rights: All rights reserved.
% @Related_paper:
% [1] Hussein A. Ammar, Raviraj Adve, Shahram Shahbazpanahiy, Gary Boudreauz,
% and Kothapalli Venkata Srinivas, "RWP+: A New Random Waypoint Model
% for High-Speed Mobility", IEEE Communications Letters.
%........................................................................

%% About
%........................................................................
% @About: 
% This simulation generates Figs. 2 and 3 in our paper [1] mentioned at the
% beginning of this file.
% In this simulation, we will compare the handover rate
% in conventional network vs Monte Carlo simulation for different scenarios:
% - Scenario 1: PPP-based network  scenario from paper:
%               X. Lin, R. K. Ganti, P. J. Fleming and J. G. Andrews,
%               "Towards Understanding the Fundamentals of Mobility in
%               Cellular Networks," in IEEE TWC, vol. 12, no. 4, pp.
%               1686-1698, April 2013.
% - Scenario 2: Cancelled
% - Scenario 3: our study [1], theta is chosen from a uniform distribution.
% - Scenario 4: our study [1], theta is chosen from a normal distribution.
%........................................................................


clear all

tic

%% Input common to all scenarios

velocityPattern = 0; %0; % 0 -> resembles the statistics from Manhattan
%                      1 -> resembles the statistics from Rome
%                      2 -> resembles the statistics from Toronto
%                      3 -> resembles the statistics from Shanghai

sim_s1_flag = 1; %1 -> simulate scenario 1, 0 -> do not simulate
sim_s3_flag = 1; %1 -> simulate scenario 3, 0 -> do not simulate
sim_s4_flag = 1; %1 -> simulate scenario 4, 0 -> do not simulate

nOfTrials = 400;
nOfTransitionsPerTrial = 10; % larger transitions per trial requires larger
%                              network area -> slower simulation, so it is
%                              preferred to keep this value small and
%                              increase the number of trials nOfTrials
%                              instead (has same effect but it does not
%                              require larger network area).

% density of base stations
lambda_DU = [10, 20, 40, 80]; % density of Base stations per km^2

pauseTime_deterministic = 0;%5; % pause time in sec, use a deterministic pause time

loc_user = [0, 0]; % typical user found at origin, coordinates are in meters

Area_x_dimension_km = 40; % width of considered area in km
Area_y_dimension_km = Area_x_dimension_km; % height of considered area in km

%% Scenario 3, my study, v is not uniformly distributed, theta is still uniform

if( velocityPattern == 0 )
mu_v_input = [4.5; 7; 8.9; 11.8; 12.5; 14.5; 15.5; 16.5; 18; 20; 25]; % mean of each Gaussian distribution, in m/s
weights_v_input = [6.5; 8.5; 2.5; 5; 4; 6; 10; 6; 10; 1; 7]; % weight for each Gaussian distribution
sigma_v_input = (1/4) .* ones(length(mu_v_input), 1);% standard deviation, 4 sigma consists 95% of PDF of Gaussian

% for transition length which have a lognormal distribution
sigma_l = 1.01;
% mean of the logarithm of the random variables
mu_l = 5.98;

elseif( velocityPattern == 1 )
    mu_v_input = [3; 4.2; 7; 9; 12; 16; 20; 29];
    weights_v_input = [0.5; 0.5; 1; 1; 10 ; 1; 0.5; 2];
    sigma_v_input = (1/4) .* ones(length(mu_v_input), 1);
    
    sigma_l = 1.06;
    mu_l = 5.78;
elseif( velocityPattern == 2 )
    mu_v_input = [4.2; 7; 9; 11.2; 12.5; 13.4; 15.3; 15.6; 17.8; 20; 23];
    weights_v_input = [4; 7; 4; 10; 4; 9; 3; 3; 2; 1.5; 9];
    sigma_v_input = (1/4) .* ones(length(mu_v_input), 1);
    
    sigma_l = 1.13;
    mu_l = 6.13;
elseif( velocityPattern == 3 )
    mu_v_input = [4; 6.5; 8.5; 11; 12.5; 15; 17.8; 23.5; 25];
    weights_v_input = [1; 7; 3; 7; 5; 7; 10; 6; 6];
    sigma_v_input = (1/4) .* ones(length(mu_v_input), 1);
    
    sigma_l = 1;
    mu_l = 7.11;
else
    error('Please set "velocityPattern" to a supported value.')
end

weights_v_sum = sum(weights_v_input);

if(length(weights_v_input) ~= length(mu_v_input))
    error('Please choose weights for each mu_v_input.')
end

% Generate the PDF of the velocity
mysinglePDFTerm = @(v, mu_v, sigma_v) (1./ (sigma_v .* sqrt(2*pi)) ) .* exp( -power(v - mu_v, 2)./(2 .* sigma_v.^2) );
for i = 1 : length(mu_v_input)
    if(i == 1)
    PDF_v = @(v) (weights_v_input(i)/weights_v_sum) .* mysinglePDFTerm(v, mu_v_input(i), sigma_v_input(i));
    else
        PDF_v = @(v) PDF_v(v) + (weights_v_input(i)/weights_v_sum) .* mysinglePDFTerm(v, mu_v_input(i), sigma_v_input(i));
    end
end
PDF_v_prepared = @(v, v2) PDF_v(v) + v2;

if( max(mu_v_input) > 40)
    error('Max mean for the velocity set being generated is %d. Please change v_x_lim to a value greater than 50 to be able to generate velocities that can span such max velocity.', max(mu_v_input))
end
v_x_lim = 0:0.1:50; % put a resolution of 0.1 m/s for generating the velocities
v_y_lim = 0; % keep this parameter zero

x1_mat = ones(length(v_y_lim),1)*v_x_lim;
y1_mat = v_y_lim'*ones(1,length(v_x_lim));

dataset_v = PDF_v_prepared(x1_mat,y1_mat);

% % to generate a random velocity according to the predefined PDF use:
% [v_rand, ~] = pinky(v_x_lim,v_y_lim,dataset_v);%,res);
% % A test to verify the correctness (you can plot the data) 
% % for i = 1 : 1000
% %     [v_rand(i), v_rand2(i)] = pinky(v_x_lim,v_y_lim,dataset_v);%,res);
% % end

% transition duration
myIntegral_input = @(v, t, mu_m, sigma_m) exp(- power( log(v .* t) - mu_l, 2)./(2 * sigma_l^2) ) ...
    .* exp(- power( v - mu_m, 2)./(2 * sigma_m^2));
for i = 1 : length(mu_v_input)
    if(i == 1)
    PDF_v_input = @(v, t) (weights_v_input(i)./ (sigma_v_input(i) .* weights_v_sum) ) .* myIntegral_input(v, t, mu_v_input(i), sigma_v_input(i));
    else
        PDF_v_input = @(v, t) PDF_v_input(v, t) + (weights_v_input(i)/(sigma_v_input(i) .* weights_v_sum)) .* myIntegral_input(v, t, mu_v_input(i), sigma_v_input(i));
    end
end
PDF_t = @(t) (1./(2 * pi * sigma_l .* t)) .* integral(@(v) PDF_v_input(v, t), 0, inf, 'ArrayValued',true);

PDF_t_prepared = @(t, t2) PDF_t(t) + t2;

t_x_lim = 0.1:0.5:1000; % put a resolution of 0.5 sec for generating the transition duration
t_y_lim = 0; % must be zero

x1_mat = ones(length(t_y_lim),1)*t_x_lim;
y1_mat = t_y_lim'*ones(1,length(t_x_lim));

dataset_t = PDF_t_prepared(x1_mat, y1_mat);

%% Scenario 4, same as scenario 3, however theta is normally distributed
% only need to define the standard deviation for theta
sigma_theta = pi/4;

%% Scenario 1, For mobility pattern from paper

% needed for scenario 1
v_min = min(mu_v_input); % in m/s, min velocity of movement for user, choose a number
% larger than 0 by some gap to avoid having users traveling long
% trips with small speed (will be stuck)
v_max = max(mu_v_input); % in m/s, max velocity of movement for user

% get same mean of transition as scenarios 3 and 4 for a fair comparison
mean_scen1 = exp(mu_l + (sigma_l^2) / 2);
lambda_waypoint = 10^6 *  (1 /  (4 * mean_scen1^2) );
if(lambda_waypoint < 0.06)
    error('You may want to make the network area larger (simulation time will increase).')
end
% lambda_waypoint_permSquared = lambda_waypoint ./ 10^6;
% mean: 1/(sqrt(lambda_waypoint_permSquared))


%% Simulation

NofHandoff = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);
t_duration = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);

NofHandoff_scen2 = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);
t_duration_scen2 = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);

NofHandoff_scen3 = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);
t_duration_scen3 = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);

NofHandoff_scen4 = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);
t_duration_scen4 = zeros(length(lambda_DU), nOfTrials, nOfTransitionsPerTrial);

for density_index = 1 : length(lambda_DU)

    for monte_index = 1 : nOfTrials

        % Generate DUs as 2D PPP
        N_DUs = poissrnd(lambda_DU(density_index)*Area_x_dimension_km*Area_y_dimension_km);
        loc_DUs = unifrnd(-Area_x_dimension_km/2, Area_x_dimension_km/2, N_DUs, 1);
        loc_DUs = 1000 .* horzcat(loc_DUs , unifrnd(-Area_y_dimension_km/2, Area_y_dimension_km/2, N_DUs, 1));
        
        [vx,vy] = voronoi(loc_DUs(:, 1), loc_DUs(:, 2));
        nOfLines = size(vx, 2);
        v2 = [vx(:), vy(:)];
                
        if( sim_s1_flag == 1 )
            % Simulate Scenario 1
            
            loc_user_new = zeros(nOfTransitionsPerTrial+1, 2);
            loc_user_new(1, :) = loc_user;

            for t_index = 1 : nOfTransitionsPerTrial

                N_WP = poissrnd(lambda_waypoint*Area_x_dimension_km*Area_y_dimension_km);
                loc_WPs = unifrnd(-Area_x_dimension_km/2, Area_x_dimension_km/2, N_WP, 1);
                loc_WPs = 1000 .* horzcat(loc_WPs , unifrnd(-Area_y_dimension_km/2, Area_y_dimension_km/2, N_WP, 1));

                distance = sqrt(sum(bsxfun(@minus, loc_user_new(t_index, :), loc_WPs).^2,2));
                [min_dist, newLoc_index] = min(distance);

                loc_user_new(t_index+1, :) = loc_WPs(newLoc_index, :);

                user_v = v_min + (v_max - v_min)*rand;

                t_duration(density_index, monte_index, t_index) = min_dist ./ user_v;

                %check if the mobility intersects with the voronoi lines
                v1 = [loc_user_new(t_index, 1), loc_user_new(t_index, 2); ...
                    loc_user_new(t_index+1, 1), loc_user_new(t_index+1, 2)];

                temp_NAN = NaN .* zeros(3*nOfLines, 2);
                filled_indices = horzcat(1:3:3*nOfLines, 2:3:3*nOfLines);
                filled_indices = sort(filled_indices);
                temp_NAN(filled_indices, :) = v2;
                [xi, yi]= polyxpoly(v1(:, 1), v1(:, 2), temp_NAN(:, 1), temp_NAN(:, 2));

                % number of handoffs equal number of crossing between
                % mobility trajectory and cell boundary
                NofHandoff(density_index, monte_index, t_index) = length(xi);

            end
        
        end
        
        %%
        if( sim_s3_flag == 1 || sim_s4_flag == 1 )
            % Simulate Scenario 3 or 4
            
            if( sim_s3_flag == 1 )
                loc_user_new = zeros(nOfTransitionsPerTrial+1, 2);
                loc_user_new(1, :) = loc_user;
            end
            
            if( sim_s4_flag == 1 )
                loc_user_new_scen4 = zeros(nOfTransitionsPerTrial+1, 2);
                loc_user_new_scen4(1, :) = loc_user;
            end

            for t_index = 1 : nOfTransitionsPerTrial

                if( sim_s3_flag == 1 )
                    th_r = unifrnd(0, 2*pi); % in radians
                
                    % generate a velocity and time, which will determine the
                    % transition length
                    [user_v, ~] = pinky(v_x_lim, v_y_lim, dataset_v);%,res);
                    if(user_v < 0.5)
                        error('A velocity less than 2 m/s was generated, please use a larger velocity profile for the PDF.')
                    end
                    
                    [transition_time_scen3, ~] = pinky(t_x_lim, t_y_lim, dataset_t);%,res);

                    moved_distance_m = user_v * transition_time_scen3;

                    loc_user_new(t_index+1, 1) = loc_user_new(t_index, 1) + moved_distance_m .* cos(th_r);
                    loc_user_new(t_index+1, 2) = loc_user_new(t_index, 2) + moved_distance_m .* sin(th_r);

                    t_duration_scen3(density_index, monte_index, t_index) = transition_time_scen3;

                    %check if the mobility intersects with the voronoi lines
                    v1 = [loc_user_new(t_index, 1), loc_user_new(t_index, 2); ...
                        loc_user_new(t_index+1, 1), loc_user_new(t_index+1, 2)];

                    temp_NAN = NaN .* zeros(3*nOfLines, 2);
                    filled_indices = horzcat(1:3:3*nOfLines, 2:3:3*nOfLines);
                    filled_indices = sort(filled_indices);
                    temp_NAN(filled_indices, :) = v2;
                    [xi, yi]= polyxpoly(v1(:, 1), v1(:, 2), temp_NAN(:, 1), temp_NAN(:, 2));

                    NofHandoff_scen3(density_index, monte_index, t_index) = length(xi);

                end
                
                
                if( sim_s4_flag == 1 )
                    th_r = normrnd(0, sigma_theta);
                    
                    % generate a velocity and time, which will determine the
                    % transition length
                    [user_v, ~] = pinky(v_x_lim, v_y_lim, dataset_v);%,res);
                    if(user_v < 0.5)
                        error('A velocity less than 2 m/s was generated, please use a larger velocity profile for the PDF.')
                    end
                    [transition_time_scen4, ~] = pinky(t_x_lim, t_y_lim, dataset_t);%,res);

                    moved_distance_m = user_v * transition_time_scen4;

                    loc_user_new_scen4(t_index+1, 1) = loc_user_new_scen4(t_index, 1) + moved_distance_m .* cos(th_r);
                    loc_user_new_scen4(t_index+1, 2) = loc_user_new_scen4(t_index, 2) + moved_distance_m .* sin(th_r);

                    t_duration_scen4(density_index, monte_index, t_index) = transition_time_scen4;

                    %check if the mobility intersects with the voronoi lines
                    v1 = [loc_user_new_scen4(t_index, 1), loc_user_new_scen4(t_index, 2); ...
                        loc_user_new_scen4(t_index+1, 1), loc_user_new_scen4(t_index+1, 2)];

                    temp_NAN = NaN .* zeros(3*nOfLines, 2);
                    filled_indices = horzcat(1:3:3*nOfLines, 2:3:3*nOfLines);
                    filled_indices = sort(filled_indices);
                    temp_NAN(filled_indices, :) = v2;
                    [xi, yi]= polyxpoly(v1(:, 1), v1(:, 2), temp_NAN(:, 1), temp_NAN(:, 2));

                    NofHandoff_scen4(density_index, monte_index, t_index) = length(xi);

                end
                
            end
        end

        if(rem(monte_index, 10) == 0)
            fprintf('Finished simulation %d for density = %d.\n', monte_index, lambda_DU(density_index))
        end
    
    end

end

if( sim_s1_flag == 1 )
    NofHandoff_avg = mean( reshape( NofHandoff, ...
        length(lambda_DU), nOfTrials*nOfTransitionsPerTrial) ...
        , 2);
    t_duration_avg = mean( reshape( t_duration, ...
        length(lambda_DU), nOfTrials*nOfTransitionsPerTrial) ...
        , 2); % all the averages will be the same, because they are independent on the density of DUs 

    handOff_rate_avg = NofHandoff_avg ./ (t_duration_avg + pauseTime_deterministic);
end


if( sim_s3_flag == 1 )
    NofHandoff_scen3_avg = mean( reshape( NofHandoff_scen3, ...
        length(lambda_DU), nOfTrials*nOfTransitionsPerTrial) ...
        , 2);
    t_duration_scen3_avg = mean( reshape( t_duration_scen3, ...
        length(lambda_DU), nOfTrials*nOfTransitionsPerTrial) ...
        , 2); % all the averages will be the same, because they are independent on the density of DUs 

    handOff_rate_scen3_avg = NofHandoff_scen3_avg ./ (t_duration_scen3_avg + pauseTime_deterministic);
end

if( sim_s4_flag == 1 )
    NofHandoff_scen4_avg = mean( reshape( NofHandoff_scen4, ...
        length(lambda_DU), nOfTrials*nOfTransitionsPerTrial) ...
        , 2);
    t_duration_scen4_avg = mean( reshape( t_duration_scen4, ...
        length(lambda_DU), nOfTrials*nOfTransitionsPerTrial) ...
        , 2); % all the averages will be the same, because they are independent on the density of DUs 

    handOff_rate_scen4_avg = NofHandoff_scen4_avg ./ (t_duration_scen4_avg + pauseTime_deterministic);
end

lambda_waypoint_permSquared = lambda_waypoint ./ 10^6;
lambda_DU_permSquared = lambda_DU ./ 10^6;

%% Handover rate formula, scenario 1
if( sim_s1_flag == 1 )

    t_duration_mean = ( log(v_max) - log(v_min) ) ./ (2 * sqrt(lambda_waypoint_permSquared) * (v_max-v_min) ); % transition duration

    handOff_rate_form = (1. / ( t_duration_mean + pauseTime_deterministic ) ) .* (2/pi) .* sqrt( lambda_DU ./ lambda_waypoint );
end


%% Handover rate formula, scenario 3, my analysis, theta is uniformly distributed
if( sim_s3_flag == 1 )
    
    % mean of velocity
    mu_v = sum(weights_v_input .* mu_v_input ./ weights_v_sum);
    
    % mean of transition duration
    expect_t_formula = exp(mu_l + (sigma_l^2/2) ) ./ sqrt(2*pi);
    myIntegral_input = @(v, mu_m, sigma_m) (1./v) .* exp(- power( v - mu_m, 2)./(2 * sigma_m^2));
    for i = 1 : length(mu_v_input)
        if(i == 1)
        PDF_v_input = @(v) (weights_v_input(i)./ (sigma_v_input(i) .* weights_v_sum) ) .* myIntegral_input(v, mu_v_input(i), sigma_v_input(i));
        else
            PDF_v_input = @(v) PDF_v_input(v) + (weights_v_input(i)/(sigma_v_input(i) .* weights_v_sum)) .* myIntegral_input(v, mu_v_input(i), sigma_v_input(i));
        end
    end
    mu_t = expect_t_formula .* integral(PDF_v_input, 0, inf);
    % handoff rate
    handOff_rate_scen3_form = (4/pi) .* sqrt(lambda_DU_permSquared) .* mu_v .* (mu_t ./ (mu_t + pauseTime_deterministic));
    
end

%% Handover rate formula, scenario 4, my analysis, theta is normaly distributed
if( sim_s4_flag == 1 )
    
    % mean of velocity
    mu_v = sum(weights_v_input .* mu_v_input ./ weights_v_sum);
    
    % mean of transition duration
    expect_t_formula = exp(mu_l + (sigma_l^2/2) ) ./ sqrt(2*pi);
    myIntegral_input = @(v, mu_m, sigma_m) (1./v) .* exp(- power( v - mu_m, 2)./(2 * sigma_m^2));
    for i = 1 : length(mu_v_input)
        if(i == 1)
        PDF_v_input = @(v) (weights_v_input(i)./ (sigma_v_input(i) .* weights_v_sum) ) .* myIntegral_input(v, mu_v_input(i), sigma_v_input(i));
        else
            PDF_v_input = @(v) PDF_v_input(v) + (weights_v_input(i)/(sigma_v_input(i) .* weights_v_sum)) .* myIntegral_input(v, mu_v_input(i), sigma_v_input(i));
        end
    end
    mu_t = expect_t_formula .* integral(PDF_v_input, 0, inf);

    sinTheta_exp = 2/pi;
    
    % handoff rate
    handOff_rate_scen4_form = sinTheta_exp .* 2 .* sqrt(lambda_DU_permSquared) .* mu_v .* (mu_t ./ (mu_t + pauseTime_deterministic));
    
end

%% Plot
figure, 
hold on
    
if( sim_s3_flag == 1 )
    plot(lambda_DU, handOff_rate_scen3_form, '-', 'LineWidth', 1.5, 'DisplayName', 'Proposed, theor.', 'Color', [0 0.4470 0.7410])
    plot(lambda_DU, handOff_rate_scen3_avg, '+', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', 'Proposed, sim., \theta uniformly dist.', 'Color', [0 0.4470 0.7410])
end

if( sim_s4_flag == 1 )
    plot(lambda_DU, handOff_rate_scen4_form, '-', 'LineWidth', 1.5, 'DisplayName', 'Proposed, theor.', 'Color', [0.8500 0.3250 0.0980])
    plot(lambda_DU, handOff_rate_scen4_avg, 'o', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', 'Proposed, sim., \theta normally dist.', 'Color', [0.8500 0.3250 0.0980])
end

if( sim_s1_flag == 1 )
    plot(lambda_DU, handOff_rate_form, '-', 'LineWidth', 1.5, 'DisplayName', 'Literature profile, theor.', 'Color', 'k')
    plot(lambda_DU, handOff_rate_avg, 'x', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', 'Literature profile, sim.', 'Color', 'k')
end

xlabel('Density of BSs per km^2', 'FontName', 'Times New Roman','FontSize',16)
ylabel('Handoff rate', 'FontName', 'Times New Roman','FontSize',16)
grid on
box on
myLgd = legend;
myLgd.FontName = 'Times New Roman';
myLgd.FontSize = 10;

%%
toc




