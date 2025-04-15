def ReadFile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Assuming the first line contains the number of locations
    num_locations = int(lines[0].strip())
    locations = []

    # Read the locations from the file
    for i in range(1, num_locations + 1):
        location_data = list(map(int, lines[i].strip().split()))
        locations.append(location_data)

    return locations


import argparse
from typing import *
from collections import defaultdict



def eucilidian_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class Truck:
    def __init__(self, truck_id, capacity, milage, orders: List[Tuple[str, int]]):
        self.truck_id = truck_id
        self.capacity = capacity
        self.milage = milage
        self.current_load = 0
        self.orders = orders
        self.list_of_products = defaultdict(int)  # Default dictionary to store product quantities


    def load(self, amount):
          for product, amount in self.orders:
            if self.capacity >= amount:
                self.current_load += amount
                self.list_of_products[product] += amount
                self.orders.remove((product, amount))
                self.capacity -= amount
            continue
    
    def return_remaining_orders(self):
        return self.orders
    
    def return_carried_products(self):
        return self.list_of_products

    def unload(self, amount):
        if self.current_load >= amount:
            self.current_load -= amount
            return True
        else:
            return False
        
    def get_current_load(self):
        return self.current_load
    
    def calculate_milage(self,distance):
        self.milage -= distance
        return self.milage
        

class Hub:
    def __init__(self, hub_id, storage: defaultdict):
        self.hub_id = hub_id
        # this is where items from trucks are stored ["apple" : [5], "banana" : [10]]
        self.storage = storage 

    def order_products(self, ordered_procuts: defaultdict) -> bool: # load_on_vans
        for item, amount in ordered_procuts.items():
            if self.storage[item] >= amount:
                self.storage[item] -= amount
            else:
                return False


    def load_from_truck(self, products: defaultdict):
        for product, amount in products:
            self.storage[product] += amount

    # def unload(self, products: List[Tuple[str, int]]) -> bool:
    #     for product, amount in products:
    #         if self.products[product] >= amount:
    #             self.products[product] -= amount
    #         else:
    #             return False
    #     return True

    # do we want to give the list of items to hub or the amount per single item? list of items probably better
