from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://thumb.wikimedia.org/wikipedia/commons/thumb/0/0e/"
    "Hawaii_Hall%2C_University_of_Hawai%CA%BBi_west_side.jpg/"
    "960px-Hawaii_Hall%2C_University_of_Hawai%CA%BBi_west_side.jpg"
)

out = Path("images/hawaii_hall.jpg")
out.parent.mkdir(parents=True, exist_ok=True)
urlretrieve(URL, out)
print(f"Saved {out}")
