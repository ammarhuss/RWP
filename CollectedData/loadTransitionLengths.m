

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
% @About: This script loads and plots the lengths of the transitions inside
%         each trip.
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

% Concatenate the transition lengths from the trips (the variables are
% already loaded from Data_[CITY_NAME].mat )
concat_transitionLength = zeros(totalNofTransitions, 1);
increment = 1;
for trip_index = 1 : nOfTrips
    myLength = length( tripTransitionsDistance_m{trip_index, 1} );
    concat_transitionLength(increment:increment+myLength-1, 1) = tripTransitionsDistance_m{trip_index, 1};
    increment = increment + myLength;
end

% Plot the CDF and PDF of the transition length L
[t_length_CDF, t_duration_x] = ecdf(concat_transitionLength);
figure,
plot(t_duration_x, t_length_CDF, 'b-', 'LineWidth', 1.5)
xlabel('Transition Length (m)', 'FontName', 'Times New Roman','FontSize',14)
ylabel('Cumulative Density Function', 'FontName', 'Times New Roman','FontSize',14)
grid on

figure,
h = histogram(concat_transitionLength, 200,'Normalization','pdf');
xlabel('Transition Length (m)', 'FontName', 'Times New Roman','FontSize',14)
ylabel('Pobability Density Function', 'FontName', 'Times New Roman','FontSize',14)
grid on








