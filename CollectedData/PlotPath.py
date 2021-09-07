# -*- coding: utf-8 -*-
"""
#........................................................................
@author: Hussein A. Ammar
@Email: hussein.ammar@mail.utoronto.ca, hussein.ammar@live.com                       
@Rights: All rights reserved.
@Related_paper:
[1] Hussein A. Ammar, Raviraj Adve, Shahram Shahbazpanahiy, Gary Boudreauz,
and Kothapalli Venkata Srinivas, "RWP+: A New Random Waypoint Model
for High-Speed Mobility", IEEE Communications Letters.
........................................................................
@About: This script plots a trip on an interactive map, it will be saved as
an HTML document: mymapOnePath.html
You need to input the waypoints of the trip represented through the longitude
and latitude as shown below in the example.
........................................................................
"""

import folium
import polyline

def merge(list1, list2): 
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 

def get_route(lat, lon):
    
    #loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    
    pickup_lon = lon[0]
    pickup_lat = lat[0]
    dropoff_lon = lon[len(lon) - 1]
    dropoff_lat = lat[len(lat) - 1]
    
    routes = merge(lat, lon)
    
    start_point = [pickup_lat, pickup_lon]
    end_point = [dropoff_lat, dropoff_lon]
    
    out = {'route':routes,
           'start_point':start_point,
           'end_point':end_point
          }

    m = get_map(out)

    m.save("mymapOnePath.html")
    
    return m


def get_map(route):
    
    m = folium.Map(location=[(route['start_point'][0] + route['end_point'][0])/2, 
                             (route['start_point'][1] + route['end_point'][1])/2], 
                   zoom_start=13)

    folium.PolyLine(
        route['route'],
        weight=8,
        color='blue',
        opacity=0.6
    ).add_to(m)

    folium.Marker(
        location=route['start_point'],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    folium.Marker(
        location=route['end_point'],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)

    return m

# Example:
lat = [41.884250,41.886390,41.887850,41.889430,41.888690,41.888460,41.886070,41.885320,41.877060,41.876890,41.878080,41.876920,41.872720,41.870640,41.870030,41.871510,41.865550,41.864280,41.857270,41.855690,41.854500,41.855490,41.851110,41.852310,41.854060,41.854180,41.853360,41.850790,41.846820,41.845480,41.843120,41.837650,41.835920,41.833610,41.832030,41.832540,41.831680,41.832160,41.835610,41.837650,41.838980]
lon = [12.398810,12.403570,12.404850,12.409420,12.412540,12.421440,12.421880,12.423030,12.428730,12.431300,12.433400,12.443210,12.449000,12.450400,12.451750,12.459580,12.464880,12.469160,12.471270,12.485470,12.487940,12.489270,12.494720,12.502850,12.508610,12.510150,12.513310,12.514880,12.516150,12.516370,12.515570,12.518750,12.518830,12.517770,12.520870,12.521260,12.524000,12.524790,12.523350,12.530210,12.529590]
get_route(lat, lon)




