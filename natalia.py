import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict
import math

def calculateDistance(instance, hubID, locationID):
    # Calculate the distance between a hub and a location
    hub = instance.Locations[hubID + 1] # +1 as the first locationID is the depot 
    # and then we have the hubs in order they appear in the file
    location = instance.Locations[locationID]
    return math.ceil(math.sqrt(pow(hub.X - location.X, 2) + pow(hub.Y - location.Y, 2)))

def groupRequestsToHubs(instance, formatted_requests):
    # groups the requests to the hubs, the closest for now
    # returns a dictionary with the requestID as key and hubID as value
    grouped = {}
    hubs = instance.Hubs
    for request in formatted_requests:
        ID_request = request[0]
        locationID = request[2]

        closest = None
        closestID = None
        # find the closest hub to the locationID
        for hub in hubs:
            if ID_request in hub.allowedRequests:
                distance = calculateDistance(instance, hub.ID, locationID)
                # if first considered hub, assign it as closest
                if closest is None:
                    closestID = hub.ID
                    closest = distance
                # if the distance is smaller than the closest, assign it as closest
                elif distance < closest:
                    closestID = hub.ID
                    closest = distance
        
        grouped[ID_request] = closestID
    return grouped