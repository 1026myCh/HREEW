from math import sin
from math import cos
from math import pi
from math import acos

def distance(epiLat,epiLon,Lat,Lon):
    epi_dist = 6371.004*acos((sin(Lat/180*pi)*sin(epiLat/180*pi)+cos(Lat/180*pi)*cos(epiLat/180*pi)*cos((epiLon-Lon)/180*pi)))
    return epi_dist