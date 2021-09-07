

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
% In this simulation, I will calculate the handover rate using the trips
% collected from OSRM. Still to calculate the HO rate we need to
% generate locations of the base stations (BSs) as a Poisson Point Process
% (PPP) with a density lambda_DU.
%........................................................................


clear all

tic

%% Load data
data_flag = 0; % 0    -> load "Data_Manhattan"
%                1    -> load "Data_Toronto"
%                2    -> load "Data_Shanghai"
%                else -> load "Data_Rome"


if(data_flag == 0)
    load('Data_Manhattan.mat');
elseif(data_flag == 1)
    load('Data_Toronto.mat');
elseif(data_flag == 2) 
    load('Data_Shanghai.mat');
else
    load('Data_Rome.mat');
end

concat_transitionLength = zeros(totalNofTransitions, 1);
increment = 1;
for trip_index = 1 : nOfTrips
    myLength = length( tripTransitionsDistance_m{trip_index, 1} );
    concat_transitionLength(increment:increment+myLength-1, 1) = tripTransitionsDistance_m{trip_index, 1};
    increment = increment + myLength;
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the average velocity per transition, this is the one needed for
% the RWP model.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We need to calculate the average velocity of each transition (the path
% before changing direction). To do this, we will accumulate the small
% distances of the small chunks (segments) of paths multiplied with the
% respective velocity inside the path. Example:
% distance1 | distance2 | distance3
% velocity1 | velocity2 | velocity3
% Then the average velocity is total_Length / (distance1/velocity1 +
% distance2/velocity2 + distance3/velocity3 + etc )

average_velocityPerTransition = [];%zeros( length(concat_transitionLength), 1);
average_velocityPerTransition_cells = cell(nOfTrips, 1);
a_v_index = 1;
    
for trip_index = 1 : nOfTrips
    accumulatedDistance_m = []; % used to map the small chunk of paths to the transition paths
    distance_velocity_temp = []; % used to get the average of velocity
    transition_index = 1; % used to track the index of each transition path (before changing direction) 
    
    for i = 1 : length( tripAnnotation_distance_m{trip_index, 1} )
    
        if( transition_index > length(tripTransitionsDistance_m{trip_index, 1}) )
            break; % we do not need the remaining chunks
        end
        
        if(tripAnnotation_speed_mPerSec{trip_index, 1}(i, 1) ~= 0 )
            % sometimes there is an entry of 0 for speed and distance, thus
            % I need to skip it
            accumulatedDistance_m = vertcat(accumulatedDistance_m, ...
                tripAnnotation_distance_m{trip_index, 1}(i, 1) );

            distance_velocity_temp = vertcat(distance_velocity_temp, ...
                tripAnnotation_distance_m{trip_index, 1}(i, 1) ./ tripAnnotation_speed_mPerSec{trip_index, 1}(i, 1) );

        end
        
        if( round(sum(accumulatedDistance_m) + 1) ...
                >= floor( tripTransitionsDistance_m{trip_index, 1}(transition_index) ) )
            
            distance_velocity = sum(distance_velocity_temp);
            
            average_velocityPerTransition(a_v_index, 1) = ...
                ( sum(accumulatedDistance_m) )...
                ./ distance_velocity;
            
            average_velocityPerTransition_cells{trip_index, 1}(transition_index, 1) = average_velocityPerTransition(a_v_index, 1);
            
            % update variables for next iteration
            transition_index = transition_index + 1;
            a_v_index = a_v_index + 1;
            
            accumulatedDistance_m = [];
            distance_velocity_temp = []; 
        end
        
        
    end
end


%% Some parameters for the PPP network

% density of base stations
lambda_DU = [10, 20, 40, 80]; % number of Base stations per km^2

pauseTime_deterministic = 0; %5; %0; % pause time is not that mportan, so use a deterministic pause time

loc_user = [0, 0]; % typical user found at origin, coordinates are in meters

Area_x_dimension_km = 40; % width of considered area
Area_y_dimension_km = Area_x_dimension_km; % height of considered area

nOFTrialPerTrip = 1; % setting it to more than 1 leads to the same answer
%                      so there is no need to simulate many network
%                      realizations for each trip.

%% Simulation

NofHandoff = cell(nOfTrips, nOFTrialPerTrip); 
t_duration = cell(nOfTrips, nOFTrialPerTrip);
for trial_index = 1 : nOFTrialPerTrip
    for trip_index = 1 : nOfTrips

            NofHandoff{trip_index, trial_index} = zeros(length(lambda_DU), length(average_velocityPerTransition_cells{trip_index, 1}) );
            t_duration{trip_index, trial_index} = zeros(length(lambda_DU), length(average_velocityPerTransition_cells{trip_index, 1}) );

    end
end


for density_index = 1 : length(lambda_DU)

    for trial_index = 1 : nOFTrialPerTrip
    
    v_index = 1; % global over all trips
    
    for trip_index = 1 : nOfTrips

        % Generate DUs as 2D PPP
        N_DUs = poissrnd(lambda_DU(density_index)*Area_x_dimension_km*Area_y_dimension_km);
        loc_DUs = unifrnd(-Area_x_dimension_km/2, Area_x_dimension_km/2, N_DUs, 1);
        loc_DUs = 1000 .* horzcat(loc_DUs , unifrnd(-Area_y_dimension_km/2, Area_y_dimension_km/2, N_DUs, 1));
        
        [vx,vy] = voronoi(loc_DUs(:, 1), loc_DUs(:, 2));
        nOfLines = size(vx, 2);
        v2 = [vx(:), vy(:)];
        
        %%
        tripTransitionsNumber = length( average_velocityPerTransition_cells{trip_index, 1} );
        
        loc_user_new = zeros(tripTransitionsNumber+1, 2);
        loc_user_new(1, :) = loc_user;
       
        for t_index = 1 : tripTransitionsNumber

            % angle measured from north clockwise
            % for some reasons the 'radians' option is giving wrong answer,
            % so use degrees (default) and convert later to radians
            az_angle = azimuth( ...
            latOut{trip_index, 1}(1, t_index), lonOut{trip_index, 1}(1, t_index), ...
            latOut{trip_index, 1}(1, t_index+1), lonOut{trip_index, 1}(1, t_index+1));

            % need to change it as angle measured using polar coordinates
            th_r_degrees = az_angle - 90;
            th_r = th_r_degrees * pi / 180; % convert to radians
            
            % generate a velocity and time, which will determine the
            % transition length
            user_v = average_velocityPerTransition(v_index);
            v_index = v_index + 1; % for next transition in same  trip or new trip
            
            moved_distance_m = tripTransitionsDistance_m{trip_index, 1}(t_index, 1); %user_v * transition_time;

            transition_time = moved_distance_m / user_v;

            loc_user_new(t_index+1, 1) = loc_user_new(t_index, 1) + moved_distance_m .* cos(th_r);
            loc_user_new(t_index+1, 2) = loc_user_new(t_index, 2) + moved_distance_m .* sin(th_r);

            t_duration{trip_index, trial_index}(density_index, t_index) = transition_time;

            %check if the mobility intersects with the voronoi lines
            v1 = [loc_user_new(t_index, 1), loc_user_new(t_index, 2); ...
                loc_user_new(t_index+1, 1), loc_user_new(t_index+1, 2)];

            temp_NAN = NaN .* zeros(3*nOfLines, 2);
            filled_indices = horzcat(1:3:3*nOfLines, 2:3:3*nOfLines);
            filled_indices = sort(filled_indices);
            temp_NAN(filled_indices, :) = v2;
            [xi, yi]= polyxpoly(v1(:, 1), v1(:, 2), temp_NAN(:, 1), temp_NAN(:, 2));
%             toc

            NofHandoff{trip_index, trial_index}(density_index, t_index) = length(xi);


        end
        
        if(rem(trip_index, 10) == 0)
            fprintf('Finished simulation %d for trip %d and density = %d.\n', trial_index, trip_index, lambda_DU(density_index))
        end
    
    end
    
    end

end

NofHandoff_concat = [];
t_duration_concat = [];

for trial_index = 1 : nOFTrialPerTrip
    for trip_index = 1 : nOfTrips

        NofHandoff_concat = horzcat(NofHandoff_concat, NofHandoff{trip_index, trial_index});

        t_duration_concat = horzcat(t_duration_concat, t_duration{trip_index, trial_index});

    end
end

NofHandoff_avg = mean( NofHandoff_concat, 2);
t_duration_avg = mean( t_duration_concat, 2); % all the average wil be the same, because they are independent on the density of DUs 

handOff_rate_avg = NofHandoff_avg ./ (t_duration_avg + pauseTime_deterministic);

%% Plot

figure, 
hold on
plot(lambda_DU, handOff_rate_avg, 'p', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', 'Empirical data', 'Color', [0.8500 0.3250 0.0980])
xlabel('Density of BSs per km^2', 'FontName', 'Times New Roman','FontSize',14)
ylabel('Handoff rate', 'FontName', 'Times New Roman','FontSize',14)
grid on
box on
myLgd = legend;
myLgd.FontName = 'Times New Roman';
myLgd.FontSize = 10;

%%
toc




