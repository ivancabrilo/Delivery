import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict
import math
import pandas as pd
import random

class Vehicle:
    def __init__(self, vehicle_type, vehicle_id, capacity, milage, visits, all_visit_locations, products):
        self.vehicle_type = vehicle_type
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.milage = milage
        self.visits = visits
        self.all_visit_locations = all_visit_locations
        self.products = products
        #self.list_of_products = defaultdict(int)  # Default dictionary to store product quantities


    def load(self, amount, distance):
        total_amount = sum(amount)
        self.products += amount
        self.capacity -= total_amount
        self.milage -= distance
    
    def return_carried_products(self):
        return self.products
    
class Solution:
    def __init__(self, day, num_of_vans, van_routes, num_of_trucks, truck_routes, cost):
        self.day = day
        self.num_of_vans = num_of_vans
        self.num_of_trucks = num_of_trucks
        self.van_routes = van_routes
        self.truck_routes = truck_routes
        self.cost = cost

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

def dictionariesLocations(instance):
    hubs = {}
    requests = {}

    for i in range(len(instance.Hubs)):
        hub = instance.Hubs[i]
        hubs[hub.ID] = hub.ID + 1 #locationID 

    for i in range(len(instance.Requests)):
        request = instance.Requests[i]
        requests[request.ID] = request.customerLocID

    return hubs, requests 

def distanceMatrix(hubs, requests, instance):
    # returns the distance matrix between the hubs and the requests
    # the distance is calculated using the Euclidean distance
    # the distance is rounded up to the nearest integer
    # large_number = 999999999
    n = len(instance.Locations)
    distance_df = pd.DataFrame(-1, index=range(1, n + 1), columns=range(1, n + 1))
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            if i == j:
                distance_df.loc[i,j] = 0
            else:
                distance = calculateDistance(i, j)
                distance_df.loc[i,j] = distance
                distance_df.loc[j,i] = distance
                # instance.Locations[i], instance.Locations[j]

    return distance_df


def groupRequestsToHubs(instance, formatted_requests, distance_df):
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
                distance = distance_df.loc[hub.ID+1, locationID]
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

def groupRequestsToHubsMulti(instance, formatted_requests, distance_df, iterations=5, offset_range=5):
    all_groupings = []

    original_grouping = groupRequestsToHubs(instance, formatted_requests, distance_df)
    all_groupings.append(original_grouping)

    for _ in range(iterations):
        # Randomly shift the coordinates of each request
        # This is done by adding a random integer between -offset_range and offset_range to the original coordinates
        shifted_locations = {}
        grouped = {}

        for request in formatted_requests:
            req_id = request[0]
            orig_loc_id = request[2]
            orig_location = instance.Locations[orig_loc_id - 1]

            # Alter location coordinates randomly
            shifted_x = orig_location.X + random.uniform(-offset_range, offset_range)
            shifted_y = orig_location.Y + random.uniform(-offset_range, offset_range)
            shifted_locations[req_id] = (shifted_x, shifted_y)

            # Assign each request to the closest hub
            closest_dist = float('inf')
            closest_hub_id = None

            for hub in instance.Hubs:
                if req_id in hub.allowedRequests:
                    hub_loc = instance.Locations[hub.ID]  # Hubs use ID as index
                    dist = math.sqrt((hub_loc.X - shifted_x) ** 2 + (hub_loc.Y - shifted_y) ** 2)

                    if dist < closest_dist:
                        closest_dist = dist
                        closest_hub_id = hub.ID

            grouped[req_id] = closest_hub_id

        all_groupings.append(grouped)

    return all_groupings

def extramileage(i, h, j, distance_df):
    m = distance_df.loc[i, h] + distance_df.loc[h, j] - distance_df.loc[i, j] 
    return m

def calculateScores(hub, requests_for_hub, distance_df, num_pivots):
    # alpha, beta an gamma are weights for the score of the requests and can be changed to optimize the solution
    alpha = 1
    beta = 3
    gamma = 2.5
    
    demand_hub = 0
    scores = defaultdict(float)

    # Compute the normalization factors
    if requests_for_hub:
        q_max = max([np.sum(request[3]) for request in requests_for_hub]) 
        c0_max = max([distance_df.loc[hub.ID+1, request[2]] for request in requests_for_hub]) 
    else:
        q_max = c0_max = 1
    
    c_max = distance_df.values.max() if not distance_df.empty else 1

    preliminary_scores = {}
    for request in requests_for_hub:
        demand = np.sum(request[3])
        norm_demand = demand / q_max

        location_ID_request = request[2]
        distance = distance_df.loc[hub.ID + 1, location_ID_request]
        norm_depot_distance = distance / c0_max 

        score = alpha * norm_demand + beta * norm_depot_distance
        hashable_request = (request[0], request[1], location_ID_request, tuple(request[3]))
        preliminary_scores[hashable_request] = score

    sorted_requests = sorted(preliminary_scores.items(), key=lambda x: x[1], reverse=True)
    pivot_locations = [req[0][2] for req in sorted_requests[:min(num_pivots, len(requests_for_hub))]]

    for request in requests_for_hub:
        location_ID_request = request[2]
        demand = np.sum(request[3])
        demand_hub += demand

        norm_demand = demand / q_max 
        distance = distance_df.loc[hub.ID+1, location_ID_request]
        norm_depot_distance = distance / c0_max 

        norm_pivot_distance = 0
        if gamma > 0 and pivot_locations:
            norm_pivot_distance = np.sum([distance_df.loc[location_ID_request, pivot] / c_max for pivot in pivot_locations]) 

        score = alpha * norm_demand + beta * norm_depot_distance + gamma * norm_pivot_distance
        hashable_request = (request[0], request[1], location_ID_request, tuple(request[3])) # to be able to have the request as a dictionairy ID
        scores[hashable_request] = score
    
    return (demand_hub, scores)

def findBestInsertion(available_vans, vans, requests_for_hub_copy, distance_df, scores):
    best_value = float("inf")
    best_h = []
    best_after = -1
    index_best_van = -1

    for j in available_vans:
        van = vans[j-1]
        for request in requests_for_hub_copy:
            for i in range(len(van.all_visit_locations)-1):
                m = extramileage(van.all_visit_locations[i], request[2], van.all_visit_locations[i+1], distance_df)
                hashable_request = (request[0], request[1], request[2], tuple(request[3]))
                score = scores.get(hashable_request, 0)

                weighted_cost = m - score * 1 # Adjust the weight as needed
                if weighted_cost < best_value:
                    best_value = weighted_cost
                    best_h = request
                    best_after = i
                    index_best_van = j-1
                    
    return (best_value, best_h, best_after, index_best_van)

def routeVan(instance, groupRequestsToHubs, formatted_requests, dict_requests, distance_df):
    hubs_to_requests = defaultdict(list)

    for request in formatted_requests:
        ID_request = request[0]
        hubID = groupRequestsToHubs[ID_request]
        hubs_to_requests[hubID].append(request) 

    totalNumberOfVans = 0
    cost = 0
    routes = [] # to keep track of routes of vans

    # Find the best routes for each hub 
    for hub in instance.Hubs:
        requests_for_hub = hubs_to_requests[hub.ID]
        num_pivots = 10
        demand_hub, scores = calculateScores(hub, requests_for_hub, distance_df, num_pivots)
        lowest_routes_cost = float("inf")
        best_routes = []
        best_number_of_vans = 0

        # To find the routes with the optimal number of vans we loop over different numbers of vans
        for number_of_vans in range(math.ceil(demand_hub/instance.VanCapacity), len(requests_for_hub)+1):
            vans = []
            option_routes = []
            routes_cost = 0
            available_vans = list(range(1,number_of_vans+1))
            scores_copy = scores.copy()
            requests_for_hub_copy = requests_for_hub.copy()

            for van_number in available_vans:
                request_with_max_score = max(scores_copy, key=scores_copy.get) # this is the pivot for this van_number
                van = Vehicle("van", van_number, instance.VanCapacity, instance.VanMaxDistance, [np.array([], dtype=int)], np.array([], dtype=int), [])
                van.visits = np.append(van.visits, request_with_max_score[0])
                # all visits inlcuding hubs at start and end
                van.all_visit_locations = np.array(list(map(dict_requests.get, van.visits)))# to make the request IDs into location IDs
                # Adding the hub loction ID at frond and back since it is the start and end
                van.all_visit_locations = np.insert(van.all_visit_locations, 0, hub.ID + 1)
                van.all_visit_locations = np.append(van.all_visit_locations, hub.ID +1) 

                distance = distance_df.loc[hub.ID+1, request_with_max_score[2]]
                van.load(request_with_max_score[3], distance)

                vans.append(van)

                routes_cost += (instance.VanDistanceCost * distance)

                # Have to format key to the format of the requests to remove it from the requests_for_hub_copy list
                request_with_max_score_formatted = [request_with_max_score[0], request_with_max_score[1], request_with_max_score[2], list(request_with_max_score[3])]
                requests_for_hub_copy.remove(request_with_max_score_formatted)
                scores_copy.pop(request_with_max_score)
        
            # Runs as long as still requests available 
            while requests_for_hub_copy:
                best_m, best_h, best_after, index_best_van = findBestInsertion(available_vans, vans, requests_for_hub_copy, distance_df, scores_copy)

                # Do this before if statement, because best_h could also be inserted at the end and then its distance to the hub needs to be considered
                vans[index_best_van].visits = np.insert(vans[index_best_van].visits, best_after-1, best_h[0]) # best_after-1 since this is an index for all_visit_locations where the first index is for the hub and this is not the case for the visits
                location_ID_last_visit = dict_requests[vans[index_best_van].visits[-1]] # Location ID of request is request ID plus the number of hubs
                if (vans[index_best_van].capacity - np.sum(best_h[3]) >= 0) & (vans[index_best_van].milage - best_m - distance_df.loc[hub.ID+1, location_ID_last_visit] >= 0): 
                    vans[index_best_van].load(best_h[3], best_m) # extra distance to travel is extramileage m
                    vans[index_best_van].all_visit_locations = np.insert(vans[index_best_van].all_visit_locations, best_after, dict_requests[best_h[0]])
                    requests_for_hub_copy.remove(best_h)
                    routes_cost += (best_m * instance.VanDistanceCost)
                else:
                    # Van cannot serve this request so has to be removed
                    vans[index_best_van].visits = vans[index_best_van].visits[vans[index_best_van].visits != best_h[0]] 
                    available_vans.remove(index_best_van+1) # remove van from list of available vans since cannot serve any more requests

                # If there are no available vans any more we leave the while loop
                if len(available_vans) == 0:
                    break

            if len(requests_for_hub_copy) == 0:
                # Then all requests for this hub are surved and it is a valid routing
                for van in vans:
                    route = [hub.ID, van.visits.tolist()]
                    location_ID_last_visit = dict_requests[van.visits[-1]]
                    routes_cost += instance.VanDistanceCost * distance_df.loc[hub.ID+1, location_ID_last_visit] 
                    option_routes.append(route)
                routes_cost += (instance.VanDayCost * number_of_vans)
            else: 
                # Then not all requests for this hub are surved and so it is not a valid routing
                routes_cost = float("inf")
            
            if routes_cost < lowest_routes_cost:
                lowest_routes_cost = routes_cost
                best_routes = option_routes
                best_number_of_vans = number_of_vans
        
        totalNumberOfVans += best_number_of_vans
        routes += best_routes
        cost += lowest_routes_cost
                

    return totalNumberOfVans, routes, cost

def printVanRoutes(numberOfVans, routes):
    # prints the routes for the vans
    print("NUMBER_OF_VANS = ", numberOfVans)
    for idx, route in enumerate(routes):
        hubID = route[0]
        ID_requests = route[1]  # this is a list
        requests_str = " ".join(str(r) for r in ID_requests)
        print(f"{idx+1} H{hubID} {requests_str}")

def formatRequest(instance): 
    results = []

    # When deliverEarlyPenalty is 0 we are not allowed to deliver early
    if instance.deliverEarlyPenalty == 0:
        max_days = 0
    else:
        max_days = instance.Days

    # Here we are looping over all options of number of days between deliveries. So say i is 2, vans only deliver to requests on days 1, 3, 5 etc. 
    for i in range(max_days+1):
        extra_cost = 0
        # returns the request in a format that is easier to work with
        formatted = []
        
        for request in instance.Requests:
            # assign the parts to the variables
            ID_request = request.ID
            day = request.desiredDay
            
            # We want to deliver every i days so the day of the request is changed into the ith day that came before 
            # it or it stays the same when the day is an ith day
            if i != 0:
                number_of_days_early = (day - 1) % i
                if number_of_days_early != 0:
                    extra_cost += instance.deliverEarlyPenalty**number_of_days_early
                day -= number_of_days_early

            locationID_request = request.customerLocID
            amounts = request.amounts

            formatted.append([ID_request, day, locationID_request, amounts])
        results.append((formatted, extra_cost))

    return results

def hubProducts(groupCustomersToHubs, instance, formatted_requests): # assuming that the customer is equal to the request 
     # returns how many products must be delivered to given hub from the depot
    results = []
    shortest_fresh_time = float("inf")
    
    for product in instance.Products:
        fresh_days = product.daysFresh
        if fresh_days < shortest_fresh_time:
            shortest_fresh_time = fresh_days

    # Here we are looping over all options of number of days between deliveries. So say i is 2, trucks only deliver to hubs on days 1, 3, 5 etc. 
    # The maximum number of days between deliveries is the min number of fresh days of all the products so we do not have expired products at the hubs
    for i in range(shortest_fresh_time+1):
        products = defaultdict(lambda: np.zeros(len(instance.Products)))
        for ID_request, day, locationID, amounts in formatted_requests:
            hubID = groupCustomersToHubs[ID_request]
            # We want to deliver every i days so the day of the request is changed into the ith day that came before 
            # it or it stays the same when the day is an ith day
            if i != 0:
                day -= (day - 1) % i
            locationID_hub = hubID + 1 # locationID_hub is the hub ID plus 1 considdering depot has location ID 1
            products[(hubID, day, locationID_hub)] += np.array(amounts)

        result = []

        for (hubID, day, locationID_hub), amounts in products.items():
            result.append([hubID, day, locationID_hub, list(amounts.astype(int))])
            
        results.append(result)
    return results # formatting is the same as the requests

def routeTruck2(instance, hubProductsGrouped, dict_hubs, distance_df):
    # alpha, beta an gamma are weights for the score of the requests and can be changed to optimize the solution
    alpha = 2
    beta = 1
    gamma = 0.5 # not used yet but will be when we use another formula 
    
    routes = []
    cost = 0
    location_depot = 1
    truck_number = 0
        
    scores = defaultdict(float)
    for hub in hubProductsGrouped:
        ID_hub = hub[0]
        location_ID_hub = hub[2]
        amounts = hub[3]
        distance = distance_df.loc[location_depot, location_ID_hub]
        score = alpha * np.sum(amounts) + beta * distance # Can be made better with normalization but need matrix for that
        hashable_hub = (hub[0], hub[1], hub[2], tuple(hub[3])) # to be able to have the request as a dictionairy ID
        scores[hashable_hub] = score

        # Runs untill all requests are served and uses as many vans as needed
    while hubProductsGrouped:
        truck_number += 1
        hub_with_max_score = max(scores, key=scores.get) # this is the pivot
        truck = Vehicle("truck", truck_number, instance.TruckCapacity, instance.TruckMaxDistance, [np.array([], dtype=int)], np.array([], dtype=int), [])
        truck.visits = np.append(truck.visits, hub_with_max_score[0])
        distance = distance_df.loc[location_depot, hub_with_max_score[2]]
        amounts = hub_with_max_score[3]

        if np.sum(amounts) <= instance.TruckCapacity:
            truck.load(amounts, distance)
            hub_with_max_score_formatted = [hub_with_max_score[0],hub_with_max_score[1], hub_with_max_score[2], list(amounts)]
            hubProductsGrouped.remove(hub_with_max_score_formatted)
            scores.pop(hub_with_max_score)

        #else:
            #truck.load(instance.TruckCapacity, distance)


        cost += instance.TruckDistanceCost * distance
    
        # Runs as long as still requests available but terminates the moment the van cannot serve more requests
        while hubProductsGrouped:
            best_m = float("inf")
            best_h = []
            best_after = -1
            for hub in hubProductsGrouped:
                # all visits inlcuding hubs at start and end
                truck.all_visit_locations = np.array(list(map(dict_hubs.get, truck.visits)))# to make the request IDs into location IDs
                # Adding the hub loction ID at frond and back since it is the start and end
                truck.all_visit_locations = np.insert(truck.all_visit_locations, 0, location_depot)
                truck.all_visit_locations = np.append(truck.all_visit_locations, location_depot) 
                for i in range(len(truck.all_visit_locations)-1):
                    m = extramileage(truck.all_visit_locations[i], hub[2], truck.all_visit_locations[i+1], distance_df)
                    if m < best_m:
                        best_m = m 
                        best_h = hub
                        best_after = i

            # Do this before if statement, because best_h could also be inserted at the end and then its distance to the hub needs to be considered
            truck.visits = np.insert(truck.visits, best_after - 1, best_h[0])
            location_ID_last_visit = dict_hubs[truck.visits[-1]] # Location ID of request is request ID plus the number of hubs
            if (truck.capacity - np.sum(best_h[3]) >= 0) & (truck.milage - best_m - distance_df.loc[location_depot, location_ID_last_visit] >= 0): 
                truck.load(best_h[3], best_m) # extra distance to travel is extramileage m
                hubProductsGrouped.remove(best_h)
                hashable_best_h = (best_h[0], best_h[1], best_h[2], tuple(best_h[3])) # convert best_h into format of keys in scores
                scores.pop(hashable_best_h)
                cost += best_m * instance.TruckDistanceCost
            else:
                truck.visits = truck.visits[truck.visits != best_h[0]] # Van cannot serve this request so has to be removed
                break

        route = [truck.visits.tolist(), truck.products]
        location_ID_last_visit = dict_hubs[truck.visits[-1]]
        cost += instance.TruckDistanceCost * distance_df.loc[location_depot, location_ID_last_visit]
        routes.append(route)

    numberOfVans = truck_number # last van number is total number of vans used
    cost += instance.VanDayCost * numberOfVans
    return numberOfVans, routes, cost
       
def routeTruck(instance, hubProductsgrouped, distance_df):
    # returns the routes for the vans
    # for now one van for one request (hub) if possible
    # works for day by day due to formatting of the answer
    # can also do all days together and then edit only the formatting of answers, might save time and make the program faster
    routes = []
    cost = 0
    location_depot = 1
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
            cost += (2 * instance.TruckDistanceCost * distance_df.loc[location_depot, hub_locationID])
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
    cost += (numberOfTrucks * instance.TruckDayCost)
    return numberOfTrucks, routes, cost

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
        amounts = route[1]  # this is a list
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
        requests_str = " ".join(str(r) for r in ID_requests)
        lines.append(f"{idx+1} H{hubID} {requests_str}")
    return "\n".join(lines)

def Optimize(instance):
    all_formatted = formatRequest(instance) # put here such that we only need to run it once
    iterations = 100
    offset_range = 3
    dict_hubs, dict_requests = dictionariesLocations(instance)
    distance_df = distanceMatrix(dict_hubs, dict_requests, instance) # distance matrix between the hubs and the requests
    BestSolution = None
    lowestCost = float("inf")
    for formatted, extra_cost in all_formatted:
        all_groupings = groupRequestsToHubsMulti(instance, formatted, distance_df, iterations, offset_range)
        for grouped in all_groupings:
            results_hubs = hubProducts(grouped, instance, formatted)
            for result_hubs in results_hubs:
                cost = extra_cost
                # Calculate cost of used hubs
                for hub in set(grouped.values()):
                    cost += instance.Hubs[hub-1].hubOpeningCost

                all_numbers_of_vans = []
                all_numbers_of_trucks = []
                new_solution = []
                for day in range(1, instance.Days + 1):
                    formatted_day = [request for request in formatted if request[1] == day]
                    result_day = [hub_request for hub_request in result_hubs if hub_request[1] == day]
                    numberOfVans, routesVans, costVans = routeVan(instance, grouped, formatted_day, dict_requests, distance_df)
                    numberOfTrucks, routesTrucks, costTrucks = routeTruck2(instance, result_day, dict_hubs, distance_df)
                    cost += (costVans + costTrucks)
                    all_numbers_of_vans.append(numberOfVans)
                    all_numbers_of_trucks.append(numberOfTrucks)
                    new_solution_day = Solution(day,numberOfVans, routesVans, numberOfTrucks, routesTrucks, cost)
                    new_solution.append(new_solution_day)
                    
                total_cost = (cost + max(all_numbers_of_vans) * instance.VanCost + max(all_numbers_of_trucks) * instance.TruckCost)
                if(total_cost < lowestCost):
                    lowestCost = total_cost
                    BestSolution = new_solution
    print(lowestCost)
    # Write solution in file
    with open("solution_fenna.txt", "w") as file:
        file.write(f"\nDATASET =  {instance.Dataset}\n")
        for solution_day in BestSolution:
            # print("DAY =", day)
            file.write(f"\nDAY = {solution_day.day}\n")
            file.write("\n")
            numberOfVans = solution_day.num_of_vans
            routesVans = solution_day.van_routes
            numberOfTrucks = solution_day.num_of_trucks
            routesTrucks = solution_day.truck_routes
            file.write(writeTruckRoutes(numberOfTrucks, routesTrucks))
            file.write("\n")
            file.write(writeVanRoutes(numberOfVans, routesVans))
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
    random.seed(1234)
    instance = ReadInstance()
    Optimize(instance)
    print("Done.")