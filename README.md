Simulation and data files will be posted once the paper gets accepted.

# RWP+ Repository
This is a supplementary repository for the scientific article:

[1] Hussein A. Ammar, Raviraj Adve, Shahram Shahbazpanahi, Gary Boudreau, and Kothapalli Venkata Srinivas. "RWP+: A New Random Waypoint Model for High-Speed Mobility".

# Abstract
In this letter, we emulate real-world statistics for mobility patterns on road systems. We then propose modifications to the random waypoint (RWP) model to better represent high-mobility profiles. In this regard, we show that the transition length is best represented by a lognormal distribution, velocity is best described by a linear combination of normal distributions centered on some set of velocities, and transition direction has a normal distribution. Compared to the assumptions used in the literature for mobile cellular networks, our modeling provides mobility metrics, such as handoff rates, that better characterize actual emulated trips from the collected statistics.

# Collected Data Folder
This folder contains the data files collected from the open source routing machine (OSRM) in four studied areas:
- Data_Manhattan.mat: 200 trips collected in Manhattan area.
- Data_Toronto.mat: 200 trips collected in Toronto area.
- Data_Shanghai.mat: 200 trips collected in Shanghai area.
- Data_Rome.mat: 200 trips collected in Rome area.

To view useful metrics related to the paper please execute any of the following MATLAB scripts.
- To plot the cumulative distribution function (CDF) and probability density function (PDF) of the transition length L, please execute the script "loadTransitionLengths.m".
- To plot the PDF of the average velocity per transition V, please execute the following script "loadAvgVelocityPerTransition.m".
- To plot the PDF of the bearing angle around the base angle \bar{\theta}b (the direction of the trip), please execute the following script "loadBearingAngle.m".

Each script file contains a flag "data_flag" which allows to choose which data to load.

# Theoretical and simulation results
To simulate the Handoff (HO) rate for a specific area (Manhattan, Toronto, Shanghai, or Rome), please use the script file "HORate_theor_sim.m". The file also contains a flag "velocityPattern" which determines which area to simulate. The results will correspond to curves found in Fig.5 in [1].

- Theroretical Results: are calculated using [1, equation (12)]; can be simulated using "HORate_theor_sim.m". The results will correspond to the "Proposed, theor.", and "Literature profile, theor." curves found in Fig.5 in [1].

- Simulation results: are calculated using Monte Carlo simulation; can be simulated using "HORate_theor_sim.m". The results will correspond to the "Proposed, sim.", and "Literature profile, sim." curves found in Fig.5 in [1].
- Details of simulation results: Setup: We simulate 400 network realizations with areas of 40 by 40 km^2. The network area is chosen large enough to guarantee that the user does not reach the network boundary within the number of transitions per trip. User movement: In each \textcolor{cyan}{network} realization, a typical user is generated at the center of the network, then the user makes a trip of 10 transitions. For each transition we generate a random velocity using equation [1, equation (3)] and a random transition time using equation [1, equation (7)], and we obtain the transition length using L_m = V_m T_m. Depending on which assumption we wish to use, the bearing angle, \theta_m, is chosen as either normally or uniformly distributed, and the user location is updated. To simulate a specific area, Manhattan for example, the input parameters for [1, equation (3)] and [1, equation (7)] are chosen as those obtained from fitting and they are shown in [1, Table II]. Number of handoffs: The number of handoffs can be calculated as discussed previously. It is simply the number of intersections between the transition (a straight line between the start and end of transition) and the Voronoi Tessellation constructed by \Phi which represents the locations of the BSs. After that, handoff rate can be calculated using [1, equation (5)].


# Empirical results
Empirical results correspond to results obtained when using the trips collected from OSRM, these trips will represent the movemenet of the user. Still to calculate the empirical HO rate we need to generate locations for the base stations (BSs) as a Poisson Point Process (PPP) with a density \lambda. To perform this simulation, please use the script "HORate_empirical.m" found in the collected data folder. The results will correspond to the "Empirial data" curve found in Fig.5 in [1].


