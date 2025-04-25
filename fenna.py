import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict
import math
import pandas as pd

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
        total_amount = sum(amount)
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

def calculateDistance(loc1, loc2):
    # Calculate the distance between a hub and a location
    hub = instance.Locations[loc1 - 1] # first locationID is the depot but we are dealing with a list so hub 1 is at index 1
    # and then we have the hubs in order they appear in the file
    location = instance.Locations[loc2 - 1] # -1 as we have a list and index starts at 0
    return math.ceil(math.sqrt(pow(hub.X - location.X, 2) + pow(hub.Y - location.Y, 2)))

def dictionariesLocations():
    hubs = {}
    requests = {}

    for i in range(len(instance.Hubs) + 1):
        hub = instance.Hubs[i]
        hubs[hub.ID] = hub.ID + 1 #locationID 

    for i in range(len(instance.Requests) + 1):
        request = instance.Requests[i]
        requests[request.ID] = request.customerLocID

    return hubs, requests 

def distanceMatrix(listHubs, listRequests):
    # returns the distance matrix between the hubs and the requests
    # the distance is calculated using the Euclidean distance
    # the distance is rounded up to the nearest integer
    large_number = 999999999
    n = len(instance.Locations)
    distance_df = pd.DataFrame(-1, index=range(1, n + 1), columns=range(1, n + 1))
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            if i == j:
                distance_df[i][j] = large_number
            else:
                distance = calculateDistance(i, j)
                distance_df[i][j] = distance
                distance_df[j][i] = distance
                # instance.Locations[i], instance.Locations[j]

    return distance_df


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
                distance = calculateDistance(instance, hub.ID+1, locationID)
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
    # alpha, beta an gamma are weights for the score of the requests and can be changed to optimize the solution
    alpha = 2
    beta = 1
    gamma = 0.5 # not used yet but will be when we use another formula 
    
    hubs_to_requests = defaultdict(list)

    for request in formatted_requests:
        ID_request = request[0]
        hubID = groupRequestsToHubs[ID_request]
        hubs_to_requests[hubID].append(request) 

    van_number = 0
    routes = [] # to keep track of routes of vans

    for hub in instance.Hubs:
        #requests_for_hub = [request_id for request_id, hub_id in groupRequestsToHubs.items() if hub_id == hub.ID]
        requests_for_hub = hubs_to_requests[hub.ID]
        scores = defaultdict(float)
        for request in requests_for_hub:
            ID_request = request[0]
            location_ID_request = request[2]
            amounts = request[3]
            distance = calculateDistance(instance, hub.ID+1, location_ID_request)
            score = alpha * np.sum(amounts) + beta * distance # Can be made better with normalization but need matrix for that
            hashable_request = (request[0], request[1], request[2], tuple(request[3])) # to be able to have the request as a dictionairy ID
            scores[hashable_request] = score

        # Runs untill all requests are served and uses as many vans as needed
        while requests_for_hub:
            van_number += 1
            request_with_max_score = max(scores, key=scores.get) # this is the pivot
            van = Vehicle("van", van_number, instance.VanCapacity, instance.VanMaxDistance, np.zeros(len(requests_for_hub)), [])
            van.visits = np.append(van.visits, request_with_max_score[0])
            distance = calculateDistance(instance, hub.ID+1, request_with_max_score[2])
            van.load(request_with_max_score[3], distance)
            # Have to format key to the format of the requests to remove it from the requests_for_hub list
            request_with_max_score_formatted = [request_with_max_score[0], request_with_max_score[1], request_with_max_score[2], list(request_with_max_score[3])]
            requests_for_hub.remove(request_with_max_score_formatted)
            scores.pop(request_with_max_score)
    
            # Runs as long as still requests available but terminates the moment the van cannot serve more requests
            while requests_for_hub:
                best_m = float("inf")
                best_h = []
                best_after = -1
                for request in requests_for_hub:
                    # all visits inlcuding hubs at start and end
                    all_visit_locations = van.visits + len(instance.Hubs) + 1 # to make the request IDs into location IDs
                    # Adding the hub loction ID at frond and back since it is the start and end
                    all_visit_locations = np.insert(all_visit_locations, 0, hub.ID + 1)
                    all_visit_locations = np.append(all_visit_locations, hub.ID +1) 
                    for i in range(len(all_visit_locations)-1):
                        m = extramileage(instance, all_visit_locations[i], request[2], all_visit_locations[i+1])
                        if m < best_m:
                            best_m = m 
                            best_h = request
                            best_after = i

                # Do this before if statement, because best_h could also be inserted at the end and then its distance to the hub needs to be considered
                van.visits = np.insert(van.visits, best_after, best_h[0])
                location_ID_last_visit = van.visits[-1] + len(instance.Hubs) + 1 # Location ID of request is request ID plus the number of hubs
                if (van.capacity - np.sum(best_h[3]) >= 0) & (van.milage - best_m - calculateDistance(instance, hub.ID+1, location_ID_last_visit) >= 0): 
                    van.load(best_h[3], best_m) # extra distance to travel is extramileage m
                    requests_for_hub.remove(best_h)
                    hashable_best_h = (best_h[0], best_h[1], best_h[2], tuple(best_h[3])) # convert best_h into format of keys in scores
                    scores.pop(hashable_best_h)
                else:
                    van.visits = van.visits[van.visits != best_h[0]] # Van cannot serve this request so has to be removed
                    routes.append(van.visits)
                    break

    numberOfVans = van_number # last van number is total number of vans used

    return numberOfVans, routes

def printVanRoutes(numberOfVans, routes):
    # prints the routes for the vans
    print("NUMBER_OF_VANS = ", numberOfVans)
    for idx, route in enumerate(routes):
        hubID = route[0]
        ID_requests = route[1:-1]  # this is a list
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
        ID_requests = route[1:-1]  # this is a list
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