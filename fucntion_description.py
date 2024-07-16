basic_function_description = [
    {
        "name" : "get_flight_info",
        "description" : "get the flight information between two given locations",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "origin_location" : {
                    "type" : "string",
                    "description" : "the depature location eg: MAA"
                    },
                "destination_location" : {
                    "type" : "string",
                    "description" : "the destination location eg: GOI"
                    }
                },
            "required" : ["origin_location", "destination_location"]
            }
        },
]


deploy_function_description = [
    {
        "name" : "deploy_chenneye",
        "description" : "deploying our backend chenneye",
    },

    # {
    #     "name" : "deploy_status",
    #     "description" : "check wheter the deployment is sucess or not",
    #     "parameters" : {
    #         "type" : "object",
    #         "properties" : {
    #             "status_message" : {
    #                 "type" : "string",
    #                 "description" : "Deployed successfully"
    #             }
    #         },
    #         "required" : ["status_message"]
    #     }
    # }
]