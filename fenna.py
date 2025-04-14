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

    def hubProducts(groupCustomersToHubs, instance): # assuming that the customer is equal to the request 
        # returns how many products must be delivered to given hub from the depot

        products = defaultdict(lambda: np.zeros(len(instance.Products)))
        
        for request in instance.Requests:
            # split the requests to get the information
            parts = request.split("amounts =")
            part1 = parts[0].split()
            part2 = parts[1]

            # assign the parts to the variables
            ID_request = int(part1[0])
            hubID = groupCustomersToHubs[ID_request] # if key is a string then adujust the key to be an int
            day = int(part1[1])
            locationID_hub = hubID + 1
            amounts = [int(x) for x in part2.strip(',').split(',')]

            products[(hubID, day, locationID_hub)] += amounts
         
        
        return products # hubID day location amount
        

def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    optimization = Optimize(instance)
    WriteResults()
    print("Done.")