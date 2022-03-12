import ssl
import websocket
import json
import base64
import hmac
import hashlib
import os
import time
import pandas as pd
payload = {"request": "/v1/order/events","nonce": int(time.time()*1000)}
encoded_payload = json.dumps(payload).encode()
b64 = base64.b64encode(encoded_payload)

column_names = ['timestampms', 'socket_sequence','event_price','amount_remaining']
root_directory = os.getcwd()
directories = {'directory':root_directory}
asset = "ethusd"
def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_message(ws, message):
    #print(directories['directory'])
    message = pd.DataFrame(eval(message))
    #print(message)
    #print(message.columns)
    if int(message['socket_sequence'])>1:
        #if len(message)==1:
        event = message["events"].values[0]
        event_side = event['side']
        message['event_type'] = event['type']
        message['event_side'] = event['side']
        message['event_price'] = event['price']
        #message['amount'] = event['delta']
        message['amount_remaining'] = event['remaining']
        #message['reason'] = event['reason']
        message.reset_index()
        message = message.drop(columns = ['type', 'eventId', 'timestamp','event_type','event_side','events'])
        if event_side == 'bid':
            # print(message.iloc[0].values)
            message.to_csv(os.path.join(directories['directory'], asset + "bids" + ".csv"), mode='a', header=None)
        elif event_side == 'ask':
            message.to_csv(os.path.join(directories['directory'], asset + "asks" + ".csv"), mode='a', header=None)




def socket_run(sub_directory):
    directories['directory'] = sub_directory
    asset= "ethusd"
    new_files = 'y'
    if new_files=='y':
        df = pd.DataFrame(columns=column_names)
        df.to_csv(os.path.join(directories['directory'], asset+"bids"+".csv"))
        df.to_csv(os.path.join(directories['directory'], asset + "asks" + ".csv"))
    # from websocket import WebSocketApp


    ws = websocket.WebSocketApp(
        "wss://api.gemini.com/v1/marketdata/"+asset+"?top_of_book=true",
        on_message=on_message)

    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    print("End")
    print("rerun")
    re_run()
def re_run():
    socket_run(directories['directory'])