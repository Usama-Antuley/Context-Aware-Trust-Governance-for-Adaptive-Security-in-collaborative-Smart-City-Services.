# Registration
import json
import os
import hashlib
import time
from datetime import datetime
import requests
import multichain

# --- Define all 5 service entries ---
service1 = {
    "Service name": "CA",
    "Authentication": "123",
    "Authorization": ["service 2", "service 3", "service 4", "service 5"],
    "Access Control": ["0"]
}

service2 = {
    "Service name": "weather",
    "Authentication": "321",
    "Authorization": ["service 1", "service 3", "service 4", "service 5"],
    "Access Control": ["1"]
}

service3 = {
    "Service name": "transport",
    "Authentication": "678",
    "Authorization": ["service 1", "service 2", "service 4", "service 5"],
    "Access Control": ["2"]
}

service4 = {
    "Service name": "sf",
    "Authentication": "456",
    "Authorization": ["service 1", "service 2", "service 3", "service 5"],
    "Access Control": ["3"]
}

service5 = {
    "Service name": "bnk",
    "Authentication": "789",
    "Authorization": ["service 1", "service 2", "service 3", "service 4"],
    "Access Control": ["4"]
}

# --- Connect to Multichain ---
rpcuser = "multichainrpc"
rpcpassword = "2xYe2PpKbiCpuXfHVhJPDUiVuus3k1dinKfMriSCD6dx"
rpchost = "127.0.0.1"
rpcport = "9724"
mc = multichain.MultiChainClient(rpchost, rpcport, rpcuser, rpcpassword)

# --- Publish each service to the stream ---
stream_name = "Service Registration"

mc.publish(stream_name, "CA", {"json": service1})
mc.publish(stream_name, "weather", {"json": service2})
mc.publish(stream_name, "transport", {"json": service3})
mc.publish(stream_name, "sf", {"json": service4})
mc.publish(stream_name, "bnk", {"json": service5})

# --- Optional: Save last published service to local file ---
with open("output.json", "w") as file:
    json.dump(service5, file, indent=4)  # Now saves Bank service as last

# --- Retrieve the last 5 stream items ---
all_items = mc.liststreamitems(stream_name)
last_5_services = all_items[-5:]  # Changed from 4 to 5

# --- Extract and print Authentication values ---
print("\nAuthentication values from last 5 services:")
for idx, item in enumerate(last_5_services, 1):
    service_data = item['data']['json']
    print(f"Service {idx} - Name: {service_data['Service name']}, Authentication: {service_data['Authentication']}")