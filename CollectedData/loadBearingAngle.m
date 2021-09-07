

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
% @About: This script loads and plots the Bearing angle for the transitons
%         which are normalized by the direction of the trip (bearing angle
%         between the start and ending point of each trip).
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

% Calculate the Bearing angle and noramlize it with respect to the
% direction of the trip

concat_headingAngle_d = zeros(totalNofTransitions, 1); % in degrees
increment = 0;

for trip_index = 1 : nOfTrips
    
    temp_length = length(latOut{trip_index, 1});
    
    
    for i = 1 : temp_length - 1
        increment = increment + 1;
        %sprintf('%f, %f, %f, %f \n ',latOut{trip_index, 1}(1, i), lonOut{trip_index, 1}(1, i),  latOut{trip_index, 1}(1, i+1), lonOut{trip_index, 1}(1, i+1))
        % The function measures azimuths clockwise from north and expresses
        % them in degrees or radians.
        % az = azimuth(lat1,lon1,lat2,lon2)
        % can also check using:
        % https://www.igismap.com/map-tool/bearing-angle
        
        concat_headingAngle_d(increment, 1) = azimuth( ...
            latOut{trip_index, 1}(1, i), lonOut{trip_index, 1}(1, i), ...
            latOut{trip_index, 1}(1, i+1), lonOut{trip_index, 1}(1, i+1));
        
    end
    
end


baseAngle = zeros(nOfTrips, 1);

concat_headingAngle_d_shifted = zeros(totalNofTransitions, 1); % in degrees

start_i = 1;
for trip_index = 1 : nOfTrips
    
    temp_length = length(latOut{trip_index, 1});
    end_i = start_i + (temp_length - 1) - 1;
    
    % determine base angle (the direction of the trip, i.e., bearing angle
    % between the start and the end point of the trip)
    baseAngle(trip_index, 1) = azimuth( ...
            latOut{trip_index, 1}(1, 1), lonOut{trip_index, 1}(1, 1), ...
            latOut{trip_index, 1}(1, temp_length), lonOut{trip_index, 1}(1, temp_length));
   
    % shift all other angles with this amount
    concat_headingAngle_d_shifted(start_i:end_i, 1) = deg2rad(...
        wrapTo180( concat_headingAngle_d(start_i:end_i, 1) - baseAngle(trip_index, 1) ) );
    
    % update the start index for the next trip
    start_i = start_i + (temp_length - 1);
        
end


figure,
h = histogram(concat_headingAngle_d_shifted, 200,'Normalization','pdf');
xlabel('Bearing angles shifted by $\bar{\theta}_b$ (radian)', 'FontName', 'Times New Roman','FontSize',12, 'Interpreter', 'Latex')
ylabel('Pobability Density Function', 'FontName', 'Times New Roman','FontSize',14)
grid on
set(gca,'XTick',-pi:pi/2:pi); 
set(gca,'XTickLabel',{'-\pi', '-\pi/2', '0','\pi/2','\pi'});






