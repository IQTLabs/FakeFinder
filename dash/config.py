import os

APP_PATH = ""
#APP_PATH = "/performance"
if os.getenv("DASH_APP_NAME"):
    APP_PATH = "/{}".format(os.getenv("DASH_APP_NAME"))

PORT = int(os.getenv("PORT", 8050))

