# Drink Atlas

IARC 425 · *Human Centered AI* · University of Tennessee, Knoxville

An environmental gallery where the room itself is the artwork. Pick any cocktail and
the room responds:

| Element | Driven by |
|---|---|
| 6 fans | the drink's preparation method (Shaken, Build, Stirred, Blend, Layer, Muddle) |
| 2 humidifiers | 30-day relative humidity where the drink's spirit comes from |
| Heat pump | the drink's ABV (stronger drink = warmer and wider) |
| 6 regional lights | brewery density by region, with a spotlight on the drink's region |

Live data comes from [24cocktails](https://24cocktails.com), [Open-Meteo](https://open-meteo.com)
and [Open Brewery DB](https://www.openbrewerydb.org). Machine positions are read from
`Exhibition Layout V1.3dm` (Rhino).

## Run it

You need [Python 3.10+](https://www.python.org/downloads/).

```bash
pip install -r requirements.txt
python app.py
```

On Windows you can also double-click `start_exhibition.bat`.

Then open <http://127.0.0.1:5000>. The first launch takes a few seconds while the
simulation compiles, and the first run downloads the full cocktail list (about a
minute); after that it is cached in `exports/`.

## Files

| File | Role |
|---|---|
| `app.py` | Pulls the live APIs, runs the airflow / heat / humidity simulation, serves the site |
| `index.html` | The gallery interface: room view, cocktail picker, graphs and workflow diagram |
| `Exhibition Layout V1.3dm` | Rhino file used as the placement reference for every device |
