# http://api.eia.gov/category/?api_key=YOUR_API_KEY_HERE[&category_id=nn][&out=xml|json]
import requests

api_url_crude_oil = "http://api.eia.gov/category/?api_key=BC7BF788DB89B9771F459FD4C84B95B9&category_id=1292190"


series_id = "http://api.eia.gov/series/?api_key=BC7BF788DB89B9771F459FD4C84B95B9&series_id=PET_IMPORTS.WORLD-US-ALL.A"
r = requests.get(series_id)

print r.text