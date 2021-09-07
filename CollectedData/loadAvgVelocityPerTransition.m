

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
% @About: This script calculates and plots the PDF of:
%         - Velocity per each segment inside the transitions of the trip.
%         - Average velocity per transition.
%         The data is obtained from the saved data for the trips using the
%         open source routing machine (OSRM).
%........................................................................

clear all

% Choose which file to load
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

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the velocities per small segments inside the transitions
% These velocities are not need for the random waypoint model (RWP)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tripAnnotation_speed_mPerSec_concat = [];
for trip_index = 1 : nOfTrips
    tripAnnotation_speed_mPerSec_concat = vertcat(tripAnnotation_speed_mPerSec_concat, tripAnnotation_speed_mPerSec{trip_index, 1});
    
end

figure,
h = histogram(tripAnnotation_speed_mPerSec_concat, 500,'Normalization','pdf');
xlabel('Velocity (m/sec)', 'FontName', 'Times New Roman','FontSize',14)
ylabel('Pobability Density Function', 'FontName', 'Times New Roman','FontSize',14)
grid on


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


figure,
h = histogram(average_velocityPerTransition, 500,'Normalization','pdf'); %%,'Normalization','probability');
% plot(binCenters, counts, 'b-', 'LineWidth', 1.5)
xlabel('Average Velocity per Transition (m/s)', 'FontName', 'Times New Roman','FontSize',12)
ylabel('Pobability Density Function', 'FontName', 'Times New Roman','FontSize',14)
grid on





