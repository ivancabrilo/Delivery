import argparse
from InstanceCO25 import InstanceCO22
import numpy as np
from collections import defaultdict

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

def formatRequest(instance): 
        # returns the request in a format that is easier to work with
        formatted = []

        for request in instance.Requests:
            # split the requests to get the information
            parts = request.split("amounts =")
            part1 = parts[0].split()
            part2 = parts[1]

            # assign the parts to the variables
            ID_request = int(part1[0])
            day = int(part1[1])
            locationID_request = int(part1[2])
            amounts = [int(x) for x in part2.strip(',').split(',')]

            formatted.append([ID_request, day, locationID_request, amounts])

        return formatted

def hubProducts(groupCustomersToHubs, instance, formatted_requests): # assuming that the customer is equal to the request 
        # returns how many products must be delivered to given hub from the depot

        products = defaultdict(lambda: np.zeros(len(instance.Products)))

        for ID_request, day, locationID, amounts in formatted_requests:
            hubID = groupCustomersToHubs[locationID]
            locationID_hub = hubID + 1 # locationID_hub is the index of the hub in the list of locations

            products[(hubID, day, locationID_hub)] += np.array(amounts)

        result = []

        for (hubID, day, locationID_hub), amounts in products.items():
            result.append([hubID, day, locationID_hub, amounts])
        
        return result # formatting is the same as the requests

def Optimize(instance):
    print("Optimizing...")

    formatted_requests = formatRequest(instance) # put here such that we only need to run it once

    # Group the requests to their nearest hub
    groupRequestsToHubs = groupRequestsToHubs()

    # Find the total amount of products needed at each hub for each day
    hubProducts = hubProducts(groupRequestsToHubs, instance, formatted_requests)

    for day in instance.Days:
         #Find truck routes
         #Find van routes

    # return the result

def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    optimization = Optimize(instance)
    WriteResults()
    print("Done.")