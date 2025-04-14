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

def Optimize(instance):
    print("Optimizing...")

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


    def hubProducts(groupCustomersToHubs, instance): # assuming that the customer is equal to the request 
        # returns how many products must be delivered to given hub from the depot

        products = defaultdict(lambda: np.zeros(len(instance.Products)))
        formatted_requests = formatRequest(instance)

        for ID_request, day, locationID, amounts in formatted_requests:
            hubID = groupCustomersToHubs[locationID]
            locationID_hub = hubID + 1 # locationID_hub is the index of the hub in the list of locations

            products[(hubID, day, locationID_hub)] += np.array(amounts)

        return products # hubID day location amount
        

def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    optimization = Optimize(instance)
    WriteResults()
    print("Done.")