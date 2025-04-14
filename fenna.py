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
            hub = groupCustomersToHubs[request.Customer]
            day = request.Day
            demand = np.array(request[3:])

            products[(hub, day)] += demand
         

        return products # hubID day location amount
        

def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    optimization = Optimize(instance)
    WriteResults()
    print("Done.")