import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict
import math

class Vehicle:
    def __init__(self, vehicle_type, vehicle_id, capacity, milage, visits, products):
        self.vehicle_type = vehicle_type
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.milage = milage
        self.visits = visits
        self.products = products
        #self.list_of_products = defaultdict(int)  # Default dictionary to store product quantities


    def load(self, amount, distance):
        total_amount = np.sum(amount)
        self.products += amount
        self.capacity -= total_amount
        self.milage -= distance
    
    def return_carried_products(self):
        return self.products

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

def extramileage(instance, i, h, j):
    m = calculateDistance(instance, i, h) + calculateDistance(instance, h, j) - calculateDistance(instance, i, j)
    return m

def routeVan(instance, groupRequestsToHubs, formatted_requests):
    alpha = 2
    beta = 1
    gamma = 0.5
    
    hubs_to_requests = defaultdict()


    for request in formatted_requests:
        ID_request = request[0]
        hubID = groupRequestsToHubs[ID_request]
        hubs_to_requests[hubID].append(request) 


    for hub in instance.Hubs:
        #requests_for_hub = [request_id for request_id, hub_id in groupRequestsToHubs.items() if hub_id == hub.ID]
        requests_for_hub = hubs_to_requests[hub.ID]
        scores = defaultdict()
        for request in requests_for_hub:
            ID_request = request[0]
            location_ID_request = request[2]
            amounts = request[3]
            distance = calculateDistance(instance, hub.ID, location_ID_request)
            score = alpha * np.sum(amounts) + beta * distance # Can be made better with normalization but need matrix for that
            scores[request] = score
        
        request_with_max_score = max(scores, key=scores.get)
        van = Vehicle("van", 1, instance.VanCapacity, instance.VanMaxDistance, np.zeros(), np.zeros(len(instance.Products)))
        van.visits[0] = request_with_max_score[0]
        distance = calculateDistance(instance, hub.ID, request_with_max_score[2])
        van.load(request_with_max_score[3], distance)
        requests_for_hub.remove(request_with_max_score)

        best_m = float("inf")
        best_h = []
        for request in requests_for_hub:
            m = extramileage(instance, hub.ID, request[2], request_with_max_score[2])
            if m < best_m:
                best_m = m 
                best_h = request
        van.visits[1] = best_h[0]
        distance = calculateDistance(instance, hub.ID, best_h[2])
        van.load(best_h[3], distance)
        requests_for_hub.remove(best_h)

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
        hubID = groupCustomersToHubs[ID_request]
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
        #print("hub_ID", hub_ID, "amounts", sum_amounts)
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
    # prints the routes for the trucks
    print("NUMBER_OF_TRUCKS = ", numberOfTrucks)
    for idx, route in enumerate(routes):
        hubID = route[0]
        amounts = route[2]  # this is a list
        amounts_str = ", ".join(str(int(r)) for r in amounts)
        print(f"{idx+1} H{hubID} {amounts_str}")

def writeTruckRoutes(numberOfTrucks, routes):
    # prints the routes for the trucks
    lines = []
    lines.append(f"NUMBER_OF_TRUCKS = {numberOfTrucks}")
    for idx, route in enumerate(routes):
        hubID = route[0]
        amounts = route[2]  # this is a list
        amounts_str = ",".join(str(int(r)) for r in amounts)
        lines.append(f"{idx+1} H{hubID} {amounts_str}")
    return "\n".join(lines)

def writeVanRoutes(numberOfVans, routes):
    # prints the routes for the vans
    lines = []
    lines.append(f"NUMBER_OF_VANS = {numberOfVans}")
    for idx, route in enumerate(routes):
        hubID = route[0]
        ID_requests = route[1]  # this is a list
        requests_str = ",".join(str(r) for r in ID_requests)
        lines.append(f"{idx+1} H{hubID} {requests_str}")
    return "\n".join(lines)

def Optimize(instance):
    formatted = formatRequest(instance) # put here such that we only need to run it once
    grouped = groupRequestsToHubs(instance, formatted)
    #print("grouped = ", grouped)
    result_hubs = hubProducts(grouped, instance, formatted)
    #print("result_hubs = ", result_hubs)
    # print("DATASET = ", instance.Dataset)

    with open("solution_main.txt", "w") as file:
        file.write(f"\nDATASET =  {instance.Dataset}\n")
        for day in range(1, instance.Days + 1):
            # print("DAY =", day)
            file.write(f"\nDAY = {day}\n")
            file.write("\n")
            formatted_day = [request for request in formatted if request[1] == day]
            result_day = [hub_request for hub_request in result_hubs if hub_request[1] == day]
            numberOfVans, routes = routeVan(instance, grouped, formatted_day)
            numberOfTrucks, routesTrucks = routeTruck(instance, result_day)

            # printTruckRoutes(numberOfTrucks, routesTrucks)
            # printVanRoutes(numberOfVans, routes)

            file.write(writeTruckRoutes(numberOfTrucks, routesTrucks))
            file.write("\n")
            file.write(writeVanRoutes(numberOfVans, routes))
            file.write("\n")

    # with open("solution_main.txt", "w") as file:
    #     file.write(f"\nDAY = {day}\n")
    #     file.write("\n")
    #     file.write(writeTruckRoutes(numberOfTrucks, routesTrucks))
    #     file.write("\n")
    #     file.write(writeVanRoutes(numberOfVans, routes))
    #     file.write("\n")
        
    
    # for res in result:
    #     print(res, "\n")
   
if __name__ == "__main__":
    instance = ReadInstance()
    #print(type(instance))
    Optimize(instance)
    print("Done.")