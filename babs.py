import argparse
from InstanceCO25 import InstanceCO22

from collections import defaultdict 
import numpy as np



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
     def hubProducts(groupCustomersToHubs, instance, requests): # assuming that the customer is equal to the request 
        # returns how many products must be delivered to given hub from the depot

        products = defaultdict(lambda: np.zeros(len(instance.Products)))
        
        for request in requests:
            # assign the parts to the variables
            ID_request = request[0]
            hubID = groupCustomersToHubs[ID_request] # if key is a string then adujust the key to be an int
            day = request[1]
            locationID_hub = hubID + 1
            amounts = request[3]

            products[(hubID, day, locationID_hub)] += amounts
         
        result = []

        for (hubID, day, locationID_hub), amounts in products.items():
            result.append([hubID, day, locationID_hub, amounts])
        
        return result # hubID day location amount
        
def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    print(instance.Locations)
    Optimize(instance)
    WriteResults()
    print("Done.")