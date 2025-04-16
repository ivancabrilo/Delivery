import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict
import math

def ReadInstance():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="Path to the instance file")
    parser.add_argument("--filetype", help="File type (default: txt)", default="txt")

    args = parser.parse_args()
    instance = InstanceCO22(args.instance, filetype=args.filetype)

    if not instance.isValid():
        print("Invalid instance.")
        return
    
    return instance

def calculateDistance(instance, hubID, locationID):
    # Calculate the distance between a hub and a location
    hub = instance.Locations[hubID] # first locationID is the depot but we are dealing with a list so hub 1 is at index 1
    # and then we have the hubs in order they appear in the file
    location = instance.Locations[locationID - 1] # -1 as we have a list and index starts at 0
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

def routeVan(instance, groupRequestsToHubs, formatted_requests):
    # returns the routes for the vans
    # for now one van for one request
    routes = []
    for request in formatted_requests:
        ID_request = request[0]
        amounts = request[3]
        hubID = groupRequestsToHubs[ID_request]
        if sum(amounts) <= instance.VanCapacity:
            # if the request can be delivered by one van, return the route
            route = [hubID, [ID_request]]
            routes.append(route)
    numberOfVans = len(routes)

    return numberOfVans, routes

def printVanRoutes(numberOfVans, routes):
    # prints the routes for the vans
    print("NUMBER_OF_VANS = ", numberOfVans)
    for idx, route in enumerate(routes):
        hubID = route[0]
        ID_requests = route[1]  # this is a list
        requests_str = ", ".join(str(r) for r in ID_requests)
        print(f"{idx+1} H{hubID} {requests_str}")

def formatRequest(instance): 
        # returns the request in a format that is easier to work with
    formatted = []

    for request in instance.Requests:
        # assign the parts to the variables
        ID_request = request.ID
        day = request.desiredDay
        locationID_request = request.customerLocID
        amounts = request.amounts

        formatted.append([ID_request, day, locationID_request, amounts])

    return formatted

def hubProducts(groupCustomersToHubs, instance, formatted_requests): # assuming that the customer is equal to the request 
     # returns how many products must be delivered to given hub from the depot

    products = defaultdict(lambda: np.zeros(len(instance.Products)))

    for ID_request, day, locationID, amounts in formatted_requests:
        hubID = groupCustomersToHubs[locationID]
        locationID_hub = hubID # locationID_hub is the index of the hub in the list of locations

        products[(hubID, day, locationID_hub)] += np.array(amounts)

    result = []

    for (hubID, day, locationID_hub), amounts in products.items():
        result.append([hubID, day, locationID_hub, amounts])
        
    return result # formatting is the same as the requests
        
def routeTruck(instance, hubProductsgrouped):
    # returns the routes for the vans
    # for now one van for one request (hub) if possible
    # works for day by day due to formatting of the answer
    # can also do all days together and then edit only the formatting of answers, might save time and make the program faster
    routes = []
    # arraySumAllHubs = np.sum([item[3] for item in hubProductsgrouped], axis=0)
    # sumAllHubs = np.sum(arraySumAllHubs)
    # print("sumAllHubs:", sumAllHubs)
    for hub in hubProductsgrouped:
        hub_ID = hub[0]
        hub_locationID = hub[2]
        amounts = hub[3]
        # have to sum each product separately so the truck knows how much of each to take
        sum_amounts = np.sum(amounts)
       # print("hub_ID", hub_ID, "amounts", sum_amounts)
        if sum_amounts <= instance.TruckCapacity:
            # if the request can be delivered by one truck, return the route
            # add the amount of each product to the route 
            route = [hub_ID, [hub_locationID], amounts]
            #print("route", route)
            routes.append(route)
        # else:
        #     for i in math.ceil(sum_amounts/instance.TruckCapacity):     # round up to the nearest integer
        #         # has to divide the products among the necessary trucks
        #         for j in range(len(amounts)):
        #             if amounts[j] < instance.TruckCapacity:
        #                 # if the amount of product is less than the truck capacity, add it to the route
        #                 route = [hub_ID, [hub_locationID], amounts[j]]
        #                 print("route", route)
        #                 routes.append(route)
        #             else:
        #                 # if the amount of product is greater than the truck capacity, divide it

                        
        #         route = [hub_ID, [hub_locationID], ]
        #         routes.append(route)
    
    numberOfTrucks = len(routes)

    return numberOfTrucks, routes

def printTruckRoutes(numberOfTrucks, routes):
    # prints the routes for the vans
    print("NUMBER_OF_TRUCKS = ", numberOfTrucks)
    for idx, route in enumerate(routes):
        hubID = route[0]
        amounts = route[2]  # this is a list
        amounts_str = ", ".join(str(int(r)) for r in amounts)
        print(f"{idx+1} H{hubID} {amounts_str}")

def Optimize(instance):
    formatted = formatRequest(instance) # put here such that we only need to run it once
    grouped = groupRequestsToHubs(instance, formatted)
    result = hubProducts(grouped, instance, formatted)
    print("DATASET = ", instance.Dataset)
    for day in range(1, instance.Days + 1):
        print("DAY =", day)
        numberOfVans, routes = routeVan(instance, grouped, formatted)
        numberOfTrucks, routesTrucks = routeTruck(instance, result)
        printTruckRoutes(numberOfTrucks, routesTrucks)
        printVanRoutes(numberOfVans, routes)
        
    
    # for res in result:
    #     print(res, "\n")
   

if __name__ == "__main__":
    instance = ReadInstance()
    #print(type(instance.VanCapacity))
    Optimize(instance)
    print("Done.")