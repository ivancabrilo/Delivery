import argparse
from InstanceCO25 import InstanceCO22
import matplotlib.pyplot as plt

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
    # look at the nearest hub per customer 
    # add that demand to the nearest hub 
    # then you know demand of the hub
    # truck to the hub with that demand 
    print("Optimizing...")

    def hubProducts(groupCustomersToHubs, Requests): # assuming that the customer is equal to the location 
        # returns how many products must be delivered to given hub from the depot

        for day in Requests.desiredDay:
            return

        products = {} # hubID day location amount 

        return products
        

def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    optimization = Optimize(instance)
    WriteResults()
    print("Done.")