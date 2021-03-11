# FakeFinder Dash-App

Dash application for interacting with FakeFinder API inference system.


# Installation

First setup a standalone virtual environment:
```
virtualenv -p /usr/local/bin/python3 venv
source venv/bin/activate
```

Next, clone the repo and install the required packages:
```
git clone https://github.com/IQTLabs/FakeFinder.git
cd dash
pip install -r requirements.txt
```

To run the app locally, one first must switch the commenting on the final two lines of `index.py`
and switch the commenting on the third and fourth lines of `config.py`.

Finally, launch the app process via `python index.py` 
and then point a browser to [http://127.0.0.1:8050/](http://127.0.0.1:8050/).
