from datetime import datetime, timedelta
import json
import numpy

def get_flight_info(origin_location, destination_location):
    "get the flight information between two given locations"

    flight_info = {
        "loc_origin" : origin_location,
        "loc_destination" : destination_location,
        "date" : str(datetime.now())
    }
    
    return json.dumps(flight_info)

def deploy_chenneye():
    "to deploy our backend chenneye"

    deployment_state = numpy.random.randint(0,2)
    current_time = str(datetime.now())
    print("The time is {}, and deployment of Chenneye has started......".format(current_time))
    return "Deployment failed" if deployment_state else "Deployment Success"

def get_fucntion(chosen_function, params):
    return deploy_chenneye() if chosen_function == "deploy_chenneye" else get_flight_info(**params)