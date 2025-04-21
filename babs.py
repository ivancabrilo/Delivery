import argparse
from InstanceCO25 import InstanceCO22

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
            ID_request = request.ID
            day = request.desiredDay
            locationID_request = request.customerLocID
            amounts = request.amounts

            formatted.append([ID_request, day, locationID_request, amounts])

        return formatted


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
    
        
def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    instance = ReadInstance()
    print(instance.Requests[0].customerLocID)
    Optimize(instance)
    WriteResults()
    print("Done.")