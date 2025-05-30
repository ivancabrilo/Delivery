import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict
import math
import pandas as pd
import random

class Vehicle:
    def __init__(self, vehicle_type, vehicle_id, capacity, milage, visits, all_visit_locations):
        # Initializes a Vehicle instance
        self.vehicle_type = vehicle_type # Truck or van
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.milage = milage
        self.visits = visits # Ordered list of locations to visit
        self.all_visit_locations = all_visit_locations # All potential locations for delivery
        self.deliveries = [] # Stores tuples of (location, products delivered)


    def load(self, hub_or_request_id, products, extra_mileage, index):
         # Loads the vehicle with products and deducts capacity and mileage
        self.capacity -= np.sum(products)
        self.milage -= extra_mileage
        self.deliveries.insert(index, (hub_or_request_id, np.array(products, dtype = int)))
    
    def return_carried_products(self):
         # Returns a list of all deliveries made by the vehicle 
        return self.deliveries
    
class Solution:
     # represents a delivery solution for a specific day
    def __init__(self, day, num_of_vans, van_routes, num_of_trucks, truck_routes, cost):
        self.day = day
        self.num_of_vans = num_of_vans
        self.num_of_trucks = num_of_trucks
        self.van_routes = van_routes
        self.truck_routes = truck_routes
        self.cost = cost

def ReadInstance():
    # Reads in the file with InstanceCO22
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
    # -1 as we have a list and indexing starts at 0
    location1 = instance.Locations[loc1 - 1] 
    location2 = instance.Locations[loc2 - 1] 
    return math.ceil(math.sqrt(pow(location1.X - location2.X, 2) + pow(location1.Y - location2.Y, 2)))

def dictionariesLocations(instance):
    # Makes dictionaries for the hub and the requests 
    hubs = {}
    requests = {}

    for i in range(len(instance.Hubs)):
        hub = instance.Hubs[i]
        hubs[hub.ID] = hub.ID + 1 #locationID 

    for i in range(len(instance.Requests)):
        request = instance.Requests[i]
        requests[request.ID] = request.customerLocID

    return hubs, requests 

def distanceMatrix(instance):
    # returns the distance matrix between the hubs and the requests
    # the distance is calculated using the Euclidean distance
    # the distance is rounded up to the nearest integer
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
     # Groups requests to hubs using randomized perturbations for robustness and diversity
    all_groupings = []

    # Initial deterministic grouping
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
                    dist = math.ceil(math.sqrt((hub_loc.X - shifted_x) ** 2 + (hub_loc.Y - shifted_y) ** 2))

                    if dist < closest_dist:
                        closest_dist = dist
                        closest_hub_id = hub.ID

            grouped[req_id] = closest_hub_id

        all_groupings.append(grouped)

    return all_groupings

def extramileage(i, h, j, distance_df):
    # Computes the extra mileage incurred by routing via a hub\request
    m = distance_df.loc[i, h] + distance_df.loc[h, j] - distance_df.loc[i, j] 
    return m

def calculateScores(hub_ID, requests_for_hub, distance_df):
    # Calculates scores for each request to help with prioritization based on demand, distance to hub, and distance to selected pivot requests 
    # alpha, beta an gamma are weights for the score of the requests and can be changed to optimize the solution
    alpha = 0.2
    beta = 0.5
    gamma = 0.7
    
    demand_hub = 0
    scores = defaultdict(float)

    # Compute the normalization factors
    if requests_for_hub:
        q_max = max([np.sum(request[3]) for request in requests_for_hub]) 
        c0_max = max([distance_df.loc[hub_ID + 1, request[2]] for request in requests_for_hub]) 
        if q_max == 0:
            q_max = 1
        if c0_max == 0:
            c0_max = 1     
    else:
        q_max = c0_max = 1
    
    c_max = distance_df.values.max() if not distance_df.empty else 1

    # Select pivot locations based on demand and distance to depot
    preliminary_scores = {}
    for request in requests_for_hub:
        demand = np.sum(request[3])
        norm_demand = demand / q_max

        location_ID_request = request[2]
        distance = distance_df.loc[hub_ID + 1, location_ID_request]
        norm_depot_distance = distance / c0_max 

        score = alpha * norm_demand + beta * norm_depot_distance
        hashable_request = (request[0], request[1], location_ID_request, tuple(request[3]))
        preliminary_scores[hashable_request] = score

    # Select top pivot locations based on score 
    sorted_requests = sorted(preliminary_scores.items(), key=lambda x: x[1], reverse=True)
    pivot_locations = [req[0][2] for req in sorted_requests[:len(requests_for_hub)]]

    # Calculate scores for each request based on demand, distance to depot, and distance to pivot locations
    for request in requests_for_hub:
        location_ID_request = request[2]
        demand = np.sum(request[3])
        demand_hub += demand

        norm_demand = demand / q_max 
        distance = distance_df.loc[hub_ID+1, location_ID_request]
        norm_depot_distance = distance / c0_max 

        norm_pivot_distance = 0
        if gamma > 0 and pivot_locations:
            norm_pivot_distance = np.sum([distance_df.loc[location_ID_request, pivot] / c_max for pivot in pivot_locations]) 

        score = alpha * norm_demand + beta * norm_depot_distance + gamma * norm_pivot_distance
        hashable_request = (request[0], request[1], location_ID_request, tuple(request[3])) 
        scores[hashable_request] = score
    
    return (demand_hub, scores)

def findBestInsertion(available_vans, vans, requests_for_hub_copy, distance_df):
    # Find the best request to insert into a van or truck route to minimize extra mileage 
    best_m = float("inf")
    best_h = []
    best_after = -1
    index_best_van = -1

    for j in available_vans:
        van = vans[j-1]
        for request in requests_for_hub_copy:
            for i in range(len(van.all_visit_locations)-1):
                m = extramileage(van.all_visit_locations[i], request[2], van.all_visit_locations[i+1], distance_df)
                if m < best_m:
                    best_m = m 
                    best_h = request
                    best_after = i
                    index_best_van = j-1
    return (best_m, best_h, best_after, index_best_van)


def routeVan(instance, groupRequestsToHubs, formatted_requests, dict_requests, distance_df):
    # Constructs van delivery routes from hubs to customers 
    hubs_to_requests = defaultdict(list)

     # Group requests by their assigned hub
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
        demand_hub, scores = calculateScores(hub.ID, requests_for_hub, distance_df)
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
            
            # Assign hightest scoring requests as pivots 
            for van_number in available_vans:
                request_with_max_score = max(scores_copy, key=scores_copy.get) # this is the pivot for this van_number
                van = Vehicle("van", van_number, instance.VanCapacity, instance.VanMaxDistance, np.array([], dtype=int), np.array([], dtype=int))
                van.visits = np.append(van.visits, request_with_max_score[0])
                # all visits inlcuding hubs at start and end
                van.all_visit_locations = np.array(list(map(dict_requests.get, van.visits)))# to make the request IDs into location IDs
                # Adding the hub loction ID at frond and back since it is the start and end
                van.all_visit_locations = np.insert(van.all_visit_locations, 0, hub.ID + 1)
                van.all_visit_locations = np.append(van.all_visit_locations, hub.ID +1) 
                distance = distance_df.loc[hub.ID+1, request_with_max_score[2]]
                van.load(request_with_max_score[0], request_with_max_score[3], distance, 0)

                vans.append(van)

                # Have to format key to the format of the requests to remove it from the requests_for_hub_copy list
                requests_for_hub_copy.remove([request_with_max_score[0], request_with_max_score[1], request_with_max_score[2], list(request_with_max_score[3])])
                scores_copy.pop(request_with_max_score)
        
            # Runs as long as still requests available 
            while requests_for_hub_copy:
                best_m, best_h, best_after, index_best_van = findBestInsertion(available_vans, vans, requests_for_hub_copy, distance_df)
                
                # Do this before if statement, because best_h could also be inserted at the end and then its distance to the hub needs to be considered
                vans[index_best_van].visits = np.insert(vans[index_best_van].visits, best_after, best_h[0]) # best_after-1 since this is an index for all_visit_locations where the first index is for the hub and this is not the case for the visits
                location_ID_last_visit = dict_requests[vans[index_best_van].visits[-1]] # Location ID of request is request ID plus the number of hubs
        
                if (vans[index_best_van].capacity - np.sum(best_h[3]) >= 0) & (vans[index_best_van].milage - best_m - distance_df.loc[hub.ID+1, location_ID_last_visit] >= 0): 
                    vans[index_best_van].load(best_h[0], best_h[3], best_m, best_after) # extra distance to travel is extramileage m
                    vans[index_best_van].all_visit_locations = np.insert(vans[index_best_van].all_visit_locations, best_after+1, dict_requests[best_h[0]])
                    requests_for_hub_copy.remove(best_h)
                    
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
                    total_distance = 0 
                    for i in range(len(van.all_visit_locations)-1): 
                        total_distance += distance_df.loc[van.all_visit_locations[i], van.all_visit_locations[i+1]]
                    routes_cost += instance.VanDistanceCost * total_distance
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

def formatRequest(instance): 
    results = []
    # When deliverEarlyPenalty is 0 we are not allowed to deliver early
    if instance.deliverEarlyPenalty == 0:
        max_days = 0
    else:
        max_days = instance.Days # Can be adusted to reduce runtime

    # Here we are looping over all options of number of days between deliveries. So say i is 2, vans only deliver to requests on days 1, 3, 5 etc. 
    for i in range(max_days + 1): # have to look how many days
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

def routeTruck(instance, hubProductsGrouped, dict_hubs, distance_df):
    # Routes trucks from the depot to hubs to deliver required product quantities 
    routes = []
    trucks = []
    cost = 0
    location_depot = 1 # Assuming depot location id is 1
    numberOfTrucks = 0 
    
    # Runs untill all requests are served and uses as many vans as needed
    while hubProductsGrouped:
        # Score hubs by how efficient they are to deliver to 
        demand, scores = calculateScores((location_depot - 1), hubProductsGrouped, distance_df)

        numberOfTrucks += 1
        hub_with_max_score = max(scores, key=scores.get) # this is the pivot
        truck = Vehicle("truck", numberOfTrucks, instance.TruckCapacity, instance.TruckMaxDistance, np.array([], dtype=int), np.array([], dtype=int))
        
        truck.visits = np.append(truck.visits, hub_with_max_score[0])
        distance = distance_df.loc[location_depot, hub_with_max_score[2]]
        
        amounts = np.array(hub_with_max_score[3], dtype = int)

        total_amount = np.sum(amounts)
        # Check if full delivery fits in the truck 
        if total_amount <= truck.capacity:
            amount_to_load = amounts.copy()
        else:
            ratio = truck.capacity / total_amount
            amount_to_load = np.floor(amounts * ratio).astype(int)

        truck.load(hub_with_max_score[0], amount_to_load, distance, 0)
        remaining_amount = amounts - amount_to_load

        # all visits inlcuding depot at start and end
        truck.all_visit_locations = np.array(list(map(dict_hubs.get, truck.visits)))# to make the hub IDs into location IDs
        # Adding the depot loction ID at frond and back since it is the start and end
        truck.all_visit_locations = np.insert(truck.all_visit_locations, 0, location_depot)
        truck.all_visit_locations = np.append(truck.all_visit_locations, location_depot) 

        # Remove the hub from the pool
        hubProductsGrouped.remove([hub_with_max_score[0], hub_with_max_score[1], hub_with_max_score[2], list(amounts)])
        if np.any(remaining_amount > 0):
            # Re-add the hub with the remaining demand for future trucks 
            hubProductsGrouped.append([hub_with_max_score[0], hub_with_max_score[1], hub_with_max_score[2], list(remaining_amount)])

        demand, scores = calculateScores((location_depot - 1), hubProductsGrouped, distance_df)
        
         # If truck is full or out of mileage, return to depot and finalize route
        if np.sum(amount_to_load) >= truck.capacity or truck.milage <= 0:
            route = [[delivery[0], np.array(delivery[1], dtype = int)] for delivery in truck.deliveries]
            location_ID_last_visit = dict_hubs[truck.visits[-1]]
            routes.append(route)
            trucks.append(truck)
            continue
    
        # Try inserting more hubs 
        while hubProductsGrouped:
            best_m = float("inf")
            best_h = None
            best_after = -1

            # Find the best insertions for a remaining hub 
            for hub in hubProductsGrouped:
                for i in range(len(truck.all_visit_locations)-1):
                    m = extramileage(truck.all_visit_locations[i], hub[2], truck.all_visit_locations[i+1], distance_df)
                    if m < best_m:
                        best_m = m 
                        best_h = hub
                        best_after = i

            if best_h is None:
                break # No valid hub inserion found 
            
            # Do this before if statement, because best_h could also be inserted at the end and then its distance to the hub needs to be considered
            truck.visits = np.insert(truck.visits, best_after, best_h[0])
            location_ID_last_visit = dict_hubs[truck.visits[-1]] # Location ID of request is request ID plus the number of hubs

            amounts = np.array(best_h[3], dtype = int)
            available_distance = truck.milage - best_m - distance_df.loc[location_depot, location_ID_last_visit]

            if (np.sum(amounts) < truck.capacity) and (available_distance > 0): 
                truck.load(best_h[0], amounts, best_m, best_after) # extra distance to travel is extramileage m
                truck.all_visit_locations = np.insert(truck.all_visit_locations, best_after+1, dict_hubs[best_h[0]])
                hubProductsGrouped.remove(best_h)
                demand, scores = calculateScores((location_depot - 1), hubProductsGrouped, distance_df)
                continue

             # If full does not fit, try partial delivery 
            ratio = truck.capacity / np.sum(amounts) if np.sum(amounts) > 0 else 0
            amount_to_load = np.floor(amounts * ratio).astype(int)

            if np.any(amount_to_load > 0) and available_distance > 0:
                truck.load(best_h[0], amount_to_load, best_m, best_after)
                truck.all_visit_locations = np.insert(truck.all_visit_locations, best_after+1, dict_hubs[best_h[0]])
                remaining_amount = amounts - amount_to_load
                # Update remaining demand
                hubProductsGrouped.remove(best_h)

                if np.any(remaining_amount > 0):
                    hubProductsGrouped.append([best_h[0], best_h[1], best_h[2], list(remaining_amount)])
            else:
                break # No feasible partial delivery or mileage left

        route = [[delivery[0], np.array(delivery[1], dtype=int)] for delivery in truck.deliveries]
        
        location_ID_last_visit = dict_hubs[truck.visits[-1]]
        #cost += instance.TruckDistanceCost * distance_df.iloc[location_depot, location_ID_last_visit]
        routes.append(route)
        trucks.append(truck)
    numberOfTrucks = len(trucks)
    
    for truck in trucks:
        total_distance = 0 
        for i in range(len(truck.all_visit_locations)-1): 
            total_distance += distance_df.loc[truck.all_visit_locations[i], truck.all_visit_locations[i+1]]
        cost += instance.TruckDistanceCost * total_distance
    #print(cost)
    cost += instance.TruckDayCost * numberOfTrucks
    return numberOfTrucks, routes, cost
       
def writeTruckRoutes(numberOfTrucks, routes):
     # Prints the routes for the trucks 
    lines = []
    lines.append(f"NUMBER_OF_TRUCKS = {numberOfTrucks}")
    
    for idx, route in enumerate(routes):
        route_str = f"{idx + 1}"
        
        # Loop through each visit in the truck route
        for visit, products in route:
            # Convert products to a string of numbers, separated by commas
            products_str = ",".join(str(int(p)) for p in products)
            # Append the hub and products to the route_str for the truck
            route_str += f" H{visit} {products_str}"
        
        # Add the route for this truck to the list of lines
        lines.append(route_str)
    
    # Join all lines into a single string separated by newlines
    return "\n".join(lines)

def Optimize(instance):
    # Main function to optimize hub assignments and route planning 
    all_formatted = formatRequest(instance) # put here such that we only need to run it once
    iterations = 50 # can be changed to reduce runtime 
    offset_range = 3
    dict_hubs, dict_requests = dictionariesLocations(instance)
    distance_df = distanceMatrix(instance) # distance matrix between the hubs and the requests
    BestSolution = None
    lowestCost = float("inf")
    best_cost_van = 0
    best_cost_truck = 0
    for formatted, extra_cost in all_formatted:
         # Try many groupings for requests to hubs 
        all_groupings = groupRequestsToHubsMulti(instance, formatted, distance_df, iterations, offset_range)
        for grouped in all_groupings:
            results_hubs = hubProducts(grouped, instance, formatted)
            for result_hubs in results_hubs:
                cost = extra_cost
                total_van_cost = 0
                total_truck_cost = 0
                # Calculate cost of used hubs
                for hub in set(grouped.values()):
                    cost += instance.Hubs[hub-1].hubOpeningCost

                all_numbers_of_vans = []
                all_numbers_of_trucks = []
                new_solution = []
                # Solve routing day-by-day
                for day in range(1, instance.Days + 1):
                    formatted_day = [request for request in formatted if request[1] == day]
                    result_day = [hub_request for hub_request in result_hubs if hub_request[1] == day]
                    # Route vans and trucks 
                    numberOfVans, routesVans, costVans = routeVan(instance, grouped, formatted_day, dict_requests, distance_df)
                    numberOfTrucks, routesTrucks, costTrucks = routeTruck(instance, result_day, dict_hubs, distance_df)
                    total_van_cost += costVans
                    total_truck_cost += costTrucks
                    cost += (costVans + costTrucks)
                    all_numbers_of_vans.append(numberOfVans)
                    all_numbers_of_trucks.append(numberOfTrucks)
                     # Store this day's result 
                    new_solution_day = Solution(day,numberOfVans, routesVans, numberOfTrucks, routesTrucks, cost)
                    new_solution.append(new_solution_day)
                # Get the final costs by adding the van and truck costs
                total_cost = (cost + max(all_numbers_of_vans) * instance.VanCost + max(all_numbers_of_trucks) * instance.TruckCost)

                if(total_cost < lowestCost):
                    lowestCost = total_cost
                    BestSolution = new_solution
                    best_cost_van = total_van_cost
                    best_cost_truck = total_truck_cost
    print(lowestCost)
    print("van cost:", best_cost_van)
    print("truck cost:", best_cost_truck)
    # Write solution in file
    with open("solution_.txt", "w") as file:
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
   
if __name__ == "__main__":
    random.seed(1234)  # Ensure reproducibilty
    instance = ReadInstance()
    Optimize(instance)
    print("Done.")