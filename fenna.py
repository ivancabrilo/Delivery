import argparse
from InstanceCO25 import InstanceCO22

def ReadInstance():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="Path to the instance file")
    parser.add_argument("--filetype", help="File type (default: txt)", default="txt")

    args = parser.parse_args()
    instance = InstanceCO22(args.instance, filetype=args.filetype)

    if not instance.isValid():
        print("Invalid instance.")
        return
    
    print("Dataset name:", instance.Dataset)
    print("Number of days:", instance.Days)
    print("Number of locations:", len(instance.Locations))
    print("First location:", instance.Locations[0])
    print("Number of requests:", len(instance.Requests))
    print("First request:", instance.Requests[0])


def Optimize():
    print("Optimizing...")

def WriteResults():
    print("Writing results...")

if __name__ == "__main__":
    ReadInstance()
    Optimize()
    WriteResults()
    print("Done.")