"""
===============================================================================
 PROMPTING IMPACT / READING BIG DATA AT THE HUMAN SCALE  (IARC 425)
 Exhibition Template - Python Backend
===============================================================================

 This is the "brain" box in the workflow diagram:

     API  ->  PYTHON  ->  JSON  ->  GRASSHOPPER  ->  RHINO
                 |
                 v
              WEBSITE

 It does four jobs:
   1. Polls whatever APIs are registered in API_REGISTRY.
   2. Runs a real 2D incompressible Navier-Stokes simulation of the gallery
      air (Numba JIT compiled), plus photometric light and humidity fields.
   3. Applies IF-THEN rules that let an API value drive a building system.
   4. Serves the website AND writes one JSON per system for Grasshopper.

-------------------------------------------------------------------------------
 AI-AGENT ORIENTATION  (read this before editing - it is the whole map)
-------------------------------------------------------------------------------
 This file is written to be edited by an AI coding agent working from a
 student's plain-English prompt. Everything a student is likely to change
 lives inside a clearly delimited block that looks like this:

     # >>> AI-AGENT EDIT ZONE: <NAME> >>>
     ...
     # <<< END AI-AGENT EDIT ZONE: <NAME> <<<

 There are FOUR of them, in this order:
     1. API_REGISTRY   - which data sources exist
     2. PLACEMENTS     - where the systems sit in the room (simulate mode)
     3. RULES          - the IF-THEN links between an API and a system
     4. TUNING         - physics constants and display ranges

 Do not restructure the solver to satisfy a placement request. Placements,
 rules and APIs are DATA; the solver is machinery. If a student says
 "put the fan in the back left corner blowing right", that is a PLACEMENTS
 edit only.

 Coordinate system, because agents get this wrong:
     x runs 0 -> ROOM_W (12 m), left to right across the front of the room.
     y runs 0 -> ROOM_D (8 m), front (near the viewer) to back.
     rot is degrees, 0 = blowing toward +x (right), 90 = blowing toward +y
     (toward the back), 180 = left, 270 = toward the front. Counter-clockwise
     in plan is increasing degrees.
     Zone numbering matches the six ceiling can lights, left to right then
     front to back:   1 2 3   <- front row
                      4 5 6   <- back row
===============================================================================
"""

import base64
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(HERE, "exports")

# =============================================================================
# >>> AI-AGENT EDIT ZONE: API_REGISTRY >>>
# =============================================================================
# Every data source the exhibition can listen to. Add one dict per source.
#
# A "synthetic" source needs no network and is the safe classroom default.
# An "http" source is a real API. Only free, no-key endpoints are pre-filled
# so the template runs on first launch; swap in whatever API the student
# actually chose.
#
# Fields:
#   id      unique slug, referenced by RULES below
#   label   what the website shows
#   kind    "synthetic" | "http"
#   units   free text, shown next to the number
#   range   [lo, hi] expected range, used to scale the sparkline only
#   -- synthetic only --
#   shape   "wave" | "walk" | "spike" | "daily"
#   seed    int, makes the feed reproducible
#   -- http only --
#   url     the endpoint
#   path    dot-path into the JSON response; list indices are numbers,
#           e.g. "features.0.properties.mag"
#   every   seconds between polls (be polite, 60+ for public APIs)
#
# TO ADD A STUDENT'S OWN API: copy the CUSTOM_SLOT entry, give it a new id,
# paste the url, and set path to wherever the number lives in the response.
# =============================================================================

API_REGISTRY = [
    {
        "id": "synthetic_wave",
        "label": "Synthetic - slow tidal wave",
        "kind": "synthetic",
        "shape": "wave",
        "seed": 11,
        "units": "index",
        "range": [0, 100],
    },
    {
        "id": "synthetic_walk",
        "label": "Synthetic - random walk",
        "kind": "synthetic",
        "shape": "walk",
        "seed": 23,
        "units": "index",
        "range": [0, 100],
    },
    {
        "id": "synthetic_spike",
        "label": "Synthetic - sparse spikes",
        "kind": "synthetic",
        "shape": "spike",
        "seed": 37,
        "units": "events",
        "range": [0, 100],
    },
    {
        "id": "synthetic_daily",
        "label": "Synthetic - 24h cycle",
        "kind": "synthetic",
        "shape": "daily",
        "seed": 5,
        "units": "index",
        "range": [0, 100],
    },
    {
        "id": "knoxville_air_temp",
        "label": "Open-Meteo - Knoxville air temp",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=35.9606&longitude=-83.9207&current=temperature_2m"
        ),
        "path": "current.temperature_2m",
        "units": "degC",
        "range": [-10, 40],
        "every": 300,
    },
    {
        "id": "knoxville_humidity",
        "label": "Open-Meteo - Knoxville rel. humidity",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=35.9606&longitude=-83.9207&current=relative_humidity_2m"
        ),
        "path": "current.relative_humidity_2m",
        "units": "%",
        "range": [0, 100],
        "every": 300,
    },
    {
        "id": "quake_latest_mag",
        "label": "USGS - latest earthquake magnitude",
        "kind": "http",
        "url": (
            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
            "all_hour.geojson"
        ),
        "path": "features.0.properties.mag",
        "units": "Mw",
        "range": [0, 8],
        "every": 120,
    },
    # -------------------------------------------------------------------------
    # LIVE EXHIBITION FEEDS - one selected cocktail drives the room
    # Optional http fields used below:
    #   "reduce": "mean"      -> the path points at a list; use its average
    #   "reduce": "sum_keys"  -> the path points at a dict; add up the values
    #                            whose (lower-cased) key is in "keys"
    #   kind "selection"      -> not polled; filled from the cocktail picked in
    #                            the page's search bar (see COCKTAIL zone).
    #                            None = nothing selected.
    # -------------------------------------------------------------------------

    # A. THE SELECTED COCKTAIL (24cocktails.com, via the page's search bar)
    {"id": "cocktail_abv", "label": "Selected cocktail - ABV",
     "kind": "selection", "field": "abv", "units": "% ABV", "range": [0, 40]},
    {"id": "cocktail_spirit_rh", "label": "Selected cocktail - base-spirit origin 30-day RH",
     "kind": "selection", "field": "spirit_rh", "units": "%", "range": [0, 100]},
    {"id": "cocktail_method_shake", "label": "Selected cocktail - method is shake",
     "kind": "selection", "field": "method:shake", "units": "0/1", "range": [0, 1]},
    {"id": "cocktail_method_build", "label": "Selected cocktail - method is build",
     "kind": "selection", "field": "method:build", "units": "0/1", "range": [0, 1]},
    {"id": "cocktail_method_stir", "label": "Selected cocktail - method is stir",
     "kind": "selection", "field": "method:stir", "units": "0/1", "range": [0, 1]},
    {"id": "cocktail_method_blend", "label": "Selected cocktail - method is blend",
     "kind": "selection", "field": "method:blend", "units": "0/1", "range": [0, 1]},
    {"id": "cocktail_method_layer", "label": "Selected cocktail - method is layer",
     "kind": "selection", "field": "method:layer", "units": "0/1", "range": [0, 1]},
    {"id": "cocktail_method_muddle", "label": "Selected cocktail - method is muddle",
     "kind": "selection", "field": "method:muddle", "units": "0/1", "range": [0, 1]},
    {"id": "cocktail_region_nw", "label": "Selected cocktail - base spirit tie to Northwest",
     "kind": "selection", "field": "region:nw", "units": "0-1", "range": [0, 1]},
    {"id": "cocktail_region_nc", "label": "Selected cocktail - base spirit tie to North Central",
     "kind": "selection", "field": "region:nc", "units": "0-1", "range": [0, 1]},
    {"id": "cocktail_region_ne", "label": "Selected cocktail - base spirit tie to Northeast",
     "kind": "selection", "field": "region:ne", "units": "0-1", "range": [0, 1]},
    {"id": "cocktail_region_sw", "label": "Selected cocktail - base spirit tie to Southwest",
     "kind": "selection", "field": "region:sw", "units": "0-1", "range": [0, 1]},
    {"id": "cocktail_region_sc", "label": "Selected cocktail - base spirit tie to South Central",
     "kind": "selection", "field": "region:sc", "units": "0-1", "range": [0, 1]},
    {"id": "cocktail_region_se", "label": "Selected cocktail - base spirit tie to Southeast",
     "kind": "selection", "field": "region:se", "units": "0-1", "range": [0, 1]},

    # B. 30-day mean relative humidity at each base spirit's origin (Open-Meteo)
    {
        "id": "rh30_netherlands",
        "label": "Open-Meteo - Schiedam, Netherlands 30-day RH (gin)",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=51.92&longitude=4.40"
            "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
            "&timezone=auto"
        ),
        "path": "daily.relative_humidity_2m_mean", "reduce": "mean",
        "units": "%", "range": [0, 100], "every": 3600,
    },
    {
        "id": "rh30_poland",
        "label": "Open-Meteo - Warsaw, Poland 30-day RH (vodka)",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=52.23&longitude=21.01"
            "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
            "&timezone=auto"
        ),
        "path": "daily.relative_humidity_2m_mean", "reduce": "mean",
        "units": "%", "range": [0, 100], "every": 3600,
    },
    {
        "id": "rh30_barbados",
        "label": "Open-Meteo - Bridgetown, Barbados 30-day RH (rum)",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=13.10&longitude=-59.62"
            "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
            "&timezone=auto"
        ),
        "path": "daily.relative_humidity_2m_mean", "reduce": "mean",
        "units": "%", "range": [0, 100], "every": 3600,
    },
    {
        "id": "rh30_ireland",
        "label": "Open-Meteo - Dublin, Ireland 30-day RH (whiskey)",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=53.35&longitude=-6.26"
            "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
            "&timezone=auto"
        ),
        "path": "daily.relative_humidity_2m_mean", "reduce": "mean",
        "units": "%", "range": [0, 100], "every": 3600,
    },
    {
        "id": "rh30_mexico",
        "label": "Open-Meteo - Jalisco, Mexico 30-day RH (tequila)",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=20.66&longitude=-103.35"
            "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
            "&timezone=auto"
        ),
        "path": "daily.relative_humidity_2m_mean", "reduce": "mean",
        "units": "%", "range": [0, 100], "every": 3600,
    },
    {
        "id": "rh30_western_europe",
        "label": "Open-Meteo - Cognac, France 30-day RH (brandy)",
        "kind": "http",
        "url": (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=45.69&longitude=-0.33"
            "&daily=relative_humidity_2m_mean&past_days=30&forecast_days=0"
            "&timezone=auto"
        ),
        "path": "daily.relative_humidity_2m_mean", "reduce": "mean",
        "units": "%", "range": [0, 100], "every": 3600,
    },

    # C. LIGHTS - Open Brewery DB brewery count per U.S. region
    #    (map grid: back row = north, front row = south)
    {
        "id": "brew_northwest",
        "label": "Open Brewery DB - Northwest breweries",
        "kind": "http",
        "url": "https://api.openbrewerydb.org/v1/breweries/meta?by_country=united_states",
        "path": "by_state", "reduce": "sum_keys",
        "keys": ["washington", "oregon", "idaho", "montana", "wyoming", "alaska"],
        "units": "breweries", "range": [0, 2500], "every": 3600,
    },
    {
        "id": "brew_north_central",
        "label": "Open Brewery DB - North Central / Midwest breweries",
        "kind": "http",
        "url": "https://api.openbrewerydb.org/v1/breweries/meta?by_country=united_states",
        "path": "by_state", "reduce": "sum_keys",
        "keys": ["north dakota", "south dakota", "nebraska", "kansas",
                 "minnesota", "iowa", "missouri", "wisconsin", "illinois",
                 "michigan", "indiana", "ohio"],
        "units": "breweries", "range": [0, 2500], "every": 3600,
    },
    {
        "id": "brew_northeast",
        "label": "Open Brewery DB - Northeast breweries",
        "kind": "http",
        "url": "https://api.openbrewerydb.org/v1/breweries/meta?by_country=united_states",
        "path": "by_state", "reduce": "sum_keys",
        "keys": ["maine", "new hampshire", "vermont", "massachusetts",
                 "rhode island", "connecticut", "new york", "new jersey",
                 "pennsylvania"],
        "units": "breweries", "range": [0, 2500], "every": 3600,
    },
    {
        "id": "brew_southwest",
        "label": "Open Brewery DB - Southwest / Pacific breweries",
        "kind": "http",
        "url": "https://api.openbrewerydb.org/v1/breweries/meta?by_country=united_states",
        "path": "by_state", "reduce": "sum_keys",
        "keys": ["california", "nevada", "utah", "colorado", "arizona",
                 "new mexico", "hawaii"],
        "units": "breweries", "range": [0, 2500], "every": 3600,
    },
    {
        "id": "brew_south_central",
        "label": "Open Brewery DB - South Central breweries",
        "kind": "http",
        "url": "https://api.openbrewerydb.org/v1/breweries/meta?by_country=united_states",
        "path": "by_state", "reduce": "sum_keys",
        "keys": ["texas", "oklahoma", "arkansas", "louisiana"],
        "units": "breweries", "range": [0, 2500], "every": 3600,
    },
    {
        "id": "brew_southeast",
        "label": "Open Brewery DB - Southeast breweries",
        "kind": "http",
        "url": "https://api.openbrewerydb.org/v1/breweries/meta?by_country=united_states",
        "path": "by_state", "reduce": "sum_keys",
        "keys": ["delaware", "maryland", "district of columbia", "virginia",
                 "west virginia", "kentucky", "tennessee", "north carolina",
                 "south carolina", "georgia", "florida", "alabama",
                 "mississippi"],
        "units": "breweries", "range": [0, 2500], "every": 3600,
    },

    {
        "id": "CUSTOM_SLOT",
        "label": "Custom - paste your own endpoint",
        "kind": "http",
        "url": "",
        "path": "",
        "units": "",
        "range": [0, 100],
        "every": 120,
    },
]

# =============================================================================
# <<< END AI-AGENT EDIT ZONE: API_REGISTRY <<<
# =============================================================================


# =============================================================================
# >>> AI-AGENT EDIT ZONE: PLACEMENTS >>>
# =============================================================================
# Where the movable systems sit when the site is in SIMULATE mode.
#
# The student lays these out by dragging in PLACE mode, presses
# "Copy placement for Claude", and pastes the result at you. When that
# happens, REPLACE THIS WHOLE LIST with what they pasted and change nothing
# else in the file.
#
#   type   "heat_pump" | "humidifier" | "fan"
#   x, y   metres, see coordinate note at the top of the file
#   rot    degrees, direction the unit discharges air
#   power  0..1 capacity scalar
#   label  shown on the diagram
#
# The six can lights are NOT in this list. They are fixed to the ceiling
# grid because they are what divides the room into its six zones - see
# CAN_LIGHTS below. They are still individually selectable and rule-driven.
# =============================================================================

# Positions come from the points in "Exhibition Layout V1.3dm" (same 12 x 8 m
# room, same numbers: Rhino's Top-view +y = the back of the room here).
# With RHINO_SYNC on (TUNING zone) the x / y / rot below are refreshed from
# the .3dm every time app.py starts, so moving a point in Rhino moves the unit.
#   fans         6 points, left column = shake / build / stir (top->bottom),
#                right column = blend / layer / muddle; each faces the centre
#   heat pump    the single heat-pump point; "omni" = heat in all directions
#   humidifiers  top and bottom points, both blowing toward the centre
PLACEMENTS = [
    {"id": "fan_shake",  "type": "fan", "x": 1.00, "y": 7.00, "rot": 329, "power": 1.0, "label": "Fan - Shaken"},
    {"id": "fan_build",  "type": "fan", "x": 1.00, "y": 4.00, "rot": 0,   "power": 1.0, "label": "Fan - Build"},
    {"id": "fan_stir",   "type": "fan", "x": 1.00, "y": 1.00, "rot": 31,  "power": 1.0, "label": "Fan - Stirred"},
    {"id": "fan_blend",  "type": "fan", "x": 11.00, "y": 7.00, "rot": 211, "power": 1.0, "label": "Fan - Blend"},
    {"id": "fan_layer",  "type": "fan", "x": 11.00, "y": 4.00, "rot": 180, "power": 1.0, "label": "Fan - Layer"},
    {"id": "fan_muddle", "type": "fan", "x": 11.00, "y": 1.00, "rot": 149, "power": 1.0, "label": "Fan - Muddle"},

    {"id": "heatpump_abv", "type": "heat_pump", "x": 6.00, "y": 4.00, "rot": 0, "power": 1.0, "omni": True, "label": "Heat pump - ABV"},

    {"id": "humidifier_top",    "type": "humidifier", "x": 6.00, "y": 7.00, "rot": 270, "power": 1.0, "label": "Humidifier - top"},
    {"id": "humidifier_bottom", "type": "humidifier", "x": 6.00, "y": 1.00, "rot": 90,  "power": 1.0, "label": "Humidifier - bottom"},
]

# =============================================================================
# <<< END AI-AGENT EDIT ZONE: PLACEMENTS <<<
# =============================================================================


# =============================================================================
# >>> AI-AGENT EDIT ZONE: COCKTAIL >>>
# =============================================================================
# How the ONE selected cocktail is translated into the room. Pure data.
# =============================================================================

COCKTAIL_API = "https://24cocktails.com/api/v1"
CATALOG_MAX_AGE_DAYS = 7          # re-download the full drink list weekly

# Base spirit -> the Open-Meteo feed (API_REGISTRY) whose 30-day RH drives
# both humidifiers. Spirits not listed (other, na) -> humidifiers off.
SPIRIT_ORIGIN_RH = {
    "gin": "rh30_netherlands",
    "vodka": "rh30_poland",
    "rum": "rh30_barbados",
    "whiskey": "rh30_ireland",
    "tequila": "rh30_mexico",
    "brandy": "rh30_western_europe",
}
SPIRIT_ORIGIN_LABEL = {
    "gin": "Netherlands", "vodka": "Poland", "rum": "Barbados",
    "whiskey": "Ireland", "tequila": "Mexico", "brandy": "Western Europe",
}

# Fans: speed of the one fan that matches the selected method (m/s).
FAN_ACTIVE_MS = 4.5

# Heat pump: ABV -> degC 1:1 (rule). The heat field's radius and strength
# scale with ABV / HEATPUMP_FULL_ABV (30 % = the top of the current data;
# stronger drinks keep growing, capped at 1.5x).
HEATPUMP_FULL_ABV = 30.0
HEATPUMP_RADIUS_M = (0.35, 2.0)     # radius at 0 % ABV, at HEATPUMP_FULL_ABV
# Calmer thermostat: within +/- HEATPUMP_DEADBAND_C of the target it stops
# correcting, and its output eases toward the new value (time constant in
# seconds of room time) instead of jumping every frame.
HEATPUMP_DEADBAND_C = 0.5
HEATPUMP_SMOOTH_S = 20.0

# Lights: brewery density is the baseline, scaled into 0.10..LIGHT_BASELINE_MAX
# so there is always headroom left for the cocktail layer. SPOTLIGHT: each
# base spirit is tied to the U.S. regions most associated with making it;
# those lights get + LIGHT_COCKTAIL_BOOST x tie (0..1) on top of their
# baseline, and every other light dims to LIGHT_SPOTLIGHT_DIM x baseline.
LIGHT_BASELINE_MAX = 0.70
LIGHT_COCKTAIL_BOOST = 0.50       # spotlight: added to the spirit's regions
LIGHT_SPOTLIGHT_DIM = 0.60        # spotlight: other regions keep 60 % of baseline
# (spirits with no regional tie - "other", non-alcoholic - leave the brewery
#  map untouched: no spotlight, no dimming)
SPIRIT_REGION_TIE = {
    "whiskey": {"se": 1.0, "nc": 0.4, "ne": 0.3},  # Kentucky/Tennessee, Midwest rye
    "rum":     {"se": 1.0, "sc": 0.6},             # Florida/Caribbean, New Orleans
    "tequila": {"sw": 1.0, "sc": 0.7},             # border states, Texas
    "gin":     {"ne": 1.0, "nw": 0.5},             # Northeast, PNW craft gin
    "vodka":   {"nc": 1.0, "nw": 0.3},             # Midwest, PNW
    "brandy":  {"sw": 1.0, "nw": 0.3},             # California brandy
}

# =============================================================================
# <<< END AI-AGENT EDIT ZONE: COCKTAIL <<<
# =============================================================================


# =============================================================================
# >>> AI-AGENT EDIT ZONE: RULES >>>
# =============================================================================
# The IF-THEN layer. This is the actual argument of the project: a number
# from the world decides what the room feels like.
#
# One rule per system. Read it as:
#
#     IF  <api>  <op>  <value>   THEN setpoint = <then>   ELSE setpoint = <otherwise>
#
#   op          ">" | "<" | ">=" | "<=" | "between" | "outside"
#               | "map" | "rank" | "share_of_max"  (continuous, see below)
#   value       the threshold
#   value2      second threshold, only for "between" / "outside"
#   then        setpoint while the condition holds
#   otherwise   setpoint while it does not
#
# This list is the ONLY place rules exist. The website reports them and shows
# which branch is firing, but cannot edit them - there is no endpoint for that.
# A student who wants a new behaviour has to come here and write it, which is
# the point. If asked for "a rule that dims zone 3 when the tide is out", add a
# dict to this list; do not add a form to the page.
#
# What "setpoint" means depends on the system:
#     heat_pump   target air temperature, degC
#     humidifier  target relative humidity, %      (below current = dehumidify)
#     fan         discharge speed, m/s             (0 = off)
#     can_light   luminous intensity, 0..1
#
# A system with no rule just holds its default setpoint. Deleting a rule is
# always safe.
# =============================================================================

# ---------------------------------------------------------------------------
# LIVE EXHIBITION RULES - ONE selected cocktail drives the room.
# Nothing selected -> every "cocktail_*" api is None -> fans, heat pump and
# humidifiers go idle and the lights show only the brewery-density map.
#
# Continuous ops (in addition to the IF-THEN ops above):
#   "map"           linear: setpoint = otherwise at api == value, rising to
#                   then at api == value2 (clamped outside that range)
#   "rank"          ranks this api against the ids in "among" (largest = 1st)
#                   and uses by_rank[rank]
#   "share_of_max"  setpoint = then * api / max(among), never below otherwise;
#                   optional "boost_api" + "boost" ADD boost * boost_api on top
#                   (clamped to 1) - used to layer the cocktail on the lights
# Extra keys:
#   "fallback"        setpoint held while the api is None
#   "idle_when_none"  True -> the unit switches OFF (no output) while the api
#                     is None, instead of holding the fallback
# ---------------------------------------------------------------------------
_FAN_METHODS = ["shake", "build", "stir", "blend", "layer", "muddle"]
_BREWERIES = ["brew_northwest", "brew_north_central", "brew_northeast",
              "brew_southwest", "brew_south_central", "brew_southeast"]
# light_1..3 = front row (south: SW, S-Central, SE); light_4..6 = back row (north)
_LIGHTS = [("light_1", "brew_southwest", "sw"), ("light_2", "brew_south_central", "sc"),
           ("light_3", "brew_southeast", "se"), ("light_4", "brew_northwest", "nw"),
           ("light_5", "brew_north_central", "nc"), ("light_6", "brew_northeast", "ne")]

RULES = (
    # 1. SIX LIGHTS: baseline = region's breweries / busiest region's
    #    (always on), PLUS a spotlight from the selected cocktail's base
    #    spirit: its regions brighten, the rest dim.
    [{"system": sid, "api": api, "op": "share_of_max", "value": None,
      "value2": None, "among": _BREWERIES, "then": LIGHT_BASELINE_MAX,
      "otherwise": 0.10, "boost_api": "cocktail_region_" + reg, "boost": LIGHT_COCKTAIL_BOOST,
      "dim": LIGHT_SPOTLIGHT_DIM, "fallback": 0.45} for sid, api, reg in _LIGHTS]

    # 2. SIX FANS: the fan for the selected cocktail's method runs, the rest
    #    stay off.
    + [{"system": "fan_" + m, "api": "cocktail_method_" + m, "op": ">=",
        "value": 1, "value2": None, "then": FAN_ACTIVE_MS, "otherwise": 0.0,
        "fallback": 0.0} for m in _FAN_METHODS]

    # 3. HEAT PUMP: ABV maps straight to degC (11 % -> 11 C, 30 % -> 30 C).
    #    Radius and strength of the heat field also grow with ABV - see
    #    HEATPUMP_RADIUS_M / HEATPUMP_FULL_ABV in the COCKTAIL zone.
    + [{"system": "heatpump_abv", "api": "cocktail_abv", "op": "map",
        "value": 0, "value2": 100, "then": 100.0, "otherwise": 0.0,
        "fallback": 21.0, "idle_when_none": True}]

    # 4. HUMIDIFIERS (both): target RH = 30-day mean RH at the base spirit's
    #    origin. "other" / non-alcoholic -> None -> humidifiers off.
    + [{"system": sid, "api": "cocktail_spirit_rh", "op": "map",
        "value": 0, "value2": 100, "then": 100.0, "otherwise": 0.0,
        "fallback": 45.0, "idle_when_none": True}
       for sid in ("humidifier_top", "humidifier_bottom")]
)

# =============================================================================
# <<< END AI-AGENT EDIT ZONE: RULES <<<
# =============================================================================


# =============================================================================
# >>> AI-AGENT EDIT ZONE: TUNING >>>
# =============================================================================
# Room, physics and display constants. Safe to nudge; changing NX/NY changes
# solver cost roughly linearly with cell count.
# =============================================================================

# Read unit positions from the Rhino model's points at startup (see
# apply_rhino_layout). Set RHINO_SYNC = False to use PLACEMENTS as written.
RHINO_SYNC = True
RHINO_FILE = "Exhibition Layout V1.3dm"     # next to app.py

ROOM_W = 12.0          # m, along +x
ROOM_D = 8.0           # m, along +y
ROOM_H = 4.0           # m, floor to ceiling

NXI, NYI = 96, 64      # interior cells; 96/12 = 8 cells per metre
DX = ROOM_W / NXI      # 0.125 m, square cells

SIM_DT = 0.10          # s of simulated time per solver step
SUBSTEPS = 6           # solver steps per published frame
FRAME_HZ = 20.0        # published frames per second
# -> 0.10 * 6 * 20 = 12 s of room time per real second.

VISCOSITY = 1.5e-3     # m^2/s, eddy viscosity (not molecular - this is a room)
DIFF_TEMP = 2.0e-3     # m^2/s, turbulent thermal diffusivity
DIFF_HUM = 2.0e-3      # m^2/s, turbulent moisture diffusivity
BUOY_K = 0.030         # horizontal Boussinesq coefficient, see solver note
PROJ_ITERS = 40        # pressure Poisson SOR sweeps
SOR_OMEGA = 1.92       # over-relaxation, ~2/(1+sin(pi/NXI)) for this grid
DIFF_ITERS = 6         # diffusion Jacobi sweeps
DRAG = 0.06            # 1/s, bulk drag standing in for out-of-plane losses

OUTDOOR_T = 14.0       # degC, what leaks in through the envelope
OUTDOOR_RH = 55.0      # %
ENVELOPE_K = 0.010     # 1/s, wall coupling strength

AMBIENT_LUX = 6.0      # lux of spill light with every can light off
CANDELA_MAX = 4200.0   # cd per can light at intensity 1.0

# Fixed display ranges, so the gradient legend never jumps around.
FIELD_RANGES = {
    "temperature": [10.0, 32.0],
    "humidity": [10.0, 90.0],
    "wind": [0.0, 2.5],
    "light": [0.0, 400.0],
}

# The six ceiling can lights. Their centres ARE the six zone centres.
CAN_LIGHTS = [
    # x/y taken from the "lights" points in Exhibition Layout V1.3dm
    {"id": "light_1", "zone": 1, "x": 3.0, "y": 2.0},
    {"id": "light_2", "zone": 2, "x": 6.0, "y": 2.0},
    {"id": "light_3", "zone": 3, "x": 9.0, "y": 2.0},
    {"id": "light_4", "zone": 4, "x": 3.0, "y": 6.0},
    {"id": "light_5", "zone": 5, "x": 6.0, "y": 6.0},
    {"id": "light_6", "zone": 6, "x": 9.0, "y": 6.0},
]

DEFAULT_SETPOINT = {
    "heat_pump": 21.0,
    "humidifier": 45.0,
    "fan": 0.0,
    "can_light": 0.45,
}

# Plant capacity. Controllers are proportional and clamp to these.
HEATPUMP_W = 9000.0        # W thermal, either direction
HUMIDIFIER_GS = 0.0016     # kg water / s, either direction
FAN_MAX_MS = 6.0           # m/s discharge

# =============================================================================
# <<< END AI-AGENT EDIT ZONE: TUNING <<<
# =============================================================================


NX = NXI + 2               # padded grid, index 0 and N-1 are boundary cells
NY = NYI + 2

AIR_RHO = 1.2              # kg/m^3
AIR_CP = 1005.0            # J/kg.K
CELL_VOL = DX * DX * ROOM_H
CELL_MASS = AIR_RHO * CELL_VOL
CELL_HEAT_CAP = CELL_MASS * AIR_CP


# =============================================================================
# SOLVER  -  2D incompressible Navier-Stokes, staggered MAC grid, Numba JIT
# =============================================================================
#
# Per step:   forces  ->  advect  ->  diffuse  ->  project
#
#   advect    semi-Lagrangian backtrace, bilinear reconstruction (Stam)
#   diffuse   implicit, Gauss-Seidel relaxation of (I - a*lap)x = x0
#   project   pressure Poisson by SOR, then subtract the gradient
#
# The grid is STAGGERED (Marker-and-Cell):
#
#         v[j+1,i]
#      +-----^-----+
#      |           |
#  u[j,i]>  p,T,H  >u[j,i+1]        p, T, H, L, phi   at cell centres
#      |   [j,i]   |                u                 on vertical faces
#      +-----^-----+                v                 on horizontal faces
#         v[j,i]
#
# That choice is not cosmetic. It is what makes the projection exact - see
# the note on project(). Array shapes, with a one-cell ghost ring:
#     cell-centred   (NYI+2, NXI+2)
#     u              (NYI+2, NXI+3)
#     v              (NYI+3, NXI+2)
#
# Why these kernels are serial njit and not parallel=True: a frame is several
# hundred relaxation sweeps, and at 96x64 one sweep is about 25 us of
# arithmetic. Numba's thread dispatch costs more than that per region, so
# parallel=True measured 94 ms/frame against well under 30 ms serial on this
# machine. Small grids want one fast thread. Past roughly 256x256 that flips.
# =============================================================================


@njit(cache=True, fastmath=True, inline="always")
def sample(f, fi, fj):
    """Bilinear probe of an array at fractional index (fi, fj), clamped."""
    ny, nx = f.shape
    if fi < 0.0:
        fi = 0.0
    if fi > nx - 1.001:
        fi = nx - 1.001
    if fj < 0.0:
        fj = 0.0
    if fj > ny - 1.001:
        fj = ny - 1.001
    i0 = int(fi)
    j0 = int(fj)
    s = fi - i0
    t = fj - j0
    return (
        (1.0 - s) * (1.0 - t) * f[j0, i0]
        + s * (1.0 - t) * f[j0, i0 + 1]
        + (1.0 - s) * t * f[j0 + 1, i0]
        + s * t * f[j0 + 1, i0 + 1]
    )


@njit(cache=True, fastmath=True)
def bnd_u(u, nxi, nyi):
    """Left/right walls are solid; top/bottom are free-slip for u."""
    for j in range(u.shape[0]):
        u[j, 0] = 0.0
        u[j, 1] = 0.0
        u[j, nxi + 1] = 0.0
        u[j, nxi + 2] = 0.0
    for i in range(u.shape[1]):
        u[0, i] = u[1, i]
        u[nyi + 1, i] = u[nyi, i]


@njit(cache=True, fastmath=True)
def bnd_v(v, nxi, nyi):
    """Top/bottom walls are solid; left/right are free-slip for v."""
    for i in range(v.shape[1]):
        v[0, i] = 0.0
        v[1, i] = 0.0
        v[nyi + 1, i] = 0.0
        v[nyi + 2, i] = 0.0
    for j in range(v.shape[0]):
        v[j, 0] = v[j, 1]
        v[j, nxi + 1] = v[j, nxi]


@njit(cache=True, fastmath=True)
def bnd_c(f, nxi, nyi):
    """Zero-gradient ghost ring for any cell-centred field."""
    for i in range(1, nxi + 1):
        f[0, i] = f[1, i]
        f[nyi + 1, i] = f[nyi, i]
    for j in range(0, nyi + 2):
        f[j, 0] = f[j, 1]
        f[j, nxi + 1] = f[j, nxi]


@njit(cache=True, fastmath=True)
def advect_u(dst, u0, v0, dt, dx, nxi, nyi):
    for j in range(1, nyi + 1):
        for i in range(2, nxi + 1):
            x = (i - 1) * dx
            y = (j - 0.5) * dx
            uu = u0[j, i]
            vv = 0.25 * (v0[j, i - 1] + v0[j + 1, i - 1] + v0[j, i] + v0[j + 1, i])
            xb = x - dt * uu
            yb = y - dt * vv
            dst[j, i] = sample(u0, xb / dx + 1.0, yb / dx + 0.5)


@njit(cache=True, fastmath=True)
def advect_v(dst, u0, v0, dt, dx, nxi, nyi):
    for j in range(2, nyi + 1):
        for i in range(1, nxi + 1):
            x = (i - 0.5) * dx
            y = (j - 1) * dx
            vv = v0[j, i]
            uu = 0.25 * (u0[j - 1, i] + u0[j - 1, i + 1] + u0[j, i] + u0[j, i + 1])
            xb = x - dt * uu
            yb = y - dt * vv
            dst[j, i] = sample(v0, xb / dx + 0.5, yb / dx + 1.0)


@njit(cache=True, fastmath=True)
def advect_c(dst, s0, u, v, dt, dx, nxi, nyi):
    for j in range(1, nyi + 1):
        for i in range(1, nxi + 1):
            x = (i - 0.5) * dx
            y = (j - 0.5) * dx
            uu = 0.5 * (u[j, i] + u[j, i + 1])
            vv = 0.5 * (v[j, i] + v[j + 1, i])
            xb = x - dt * uu
            yb = y - dt * vv
            dst[j, i] = sample(s0, xb / dx + 0.5, yb / dx + 0.5)


@njit(cache=True, fastmath=True)
def diffuse_u(x, x0, a, iters, nxi, nyi):
    invc = 1.0 / (1.0 + 4.0 * a)
    for _ in range(iters):
        for j in range(1, nyi + 1):
            for i in range(2, nxi + 1):
                x[j, i] = (
                    x0[j, i]
                    + a * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i])
                ) * invc
        bnd_u(x, nxi, nyi)


@njit(cache=True, fastmath=True)
def diffuse_v(x, x0, a, iters, nxi, nyi):
    invc = 1.0 / (1.0 + 4.0 * a)
    for _ in range(iters):
        for j in range(2, nyi + 1):
            for i in range(1, nxi + 1):
                x[j, i] = (
                    x0[j, i]
                    + a * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i])
                ) * invc
        bnd_v(x, nxi, nyi)


@njit(cache=True, fastmath=True)
def diffuse_c(x, x0, a, iters, nxi, nyi):
    invc = 1.0 / (1.0 + 4.0 * a)
    for _ in range(iters):
        for j in range(1, nyi + 1):
            for i in range(1, nxi + 1):
                x[j, i] = (
                    x0[j, i]
                    + a * (x[j, i - 1] + x[j, i + 1] + x[j - 1, i] + x[j + 1, i])
                ) * invc
        bnd_c(x, nxi, nyi)


@njit(cache=True, fastmath=True)
def project(u, v, phi, div, dx, iters, omega, nxi, nyi):
    """
    Exact Hodge projection on the staggered grid.

    On a MAC grid the discrete divergence (a one-cell difference of face
    velocities) composed with the discrete gradient (a one-cell difference of
    cell-centred phi, landing back on faces) IS the standard five-point
    Laplacian. So solving lap(phi) = div(u*) and subtracting grad(phi) drives
    the divergence to the solver's tolerance rather than to a checkerboard
    floor. That identity is the whole reason for staggering; a collocated grid
    with central differences cannot do it, which is the well-known wart in
    Stam's original stable-fluids code.

    The Poisson solve is SOR, not plain Gauss-Seidel: same arithmetic per
    sweep, but GS needs O(N^2) sweeps on this operator and SOR near
    omega = 2/(1+sin(pi/N)) needs O(N). phi is deliberately NOT zeroed between
    calls; last frame's solution is a very good initial guess.

    Compatibility: every wall face is held at zero normal velocity, so the
    integral of div over the domain telescopes to exactly zero and the pure
    Neumann problem is solvable. phi is defined up to a constant, which the
    gradient discards.
    """
    inv_dx = 1.0 / dx
    for j in range(1, nyi + 1):
        for i in range(1, nxi + 1):
            div[j, i] = (
                (u[j, i + 1] - u[j, i]) + (v[j + 1, i] - v[j, i])
            ) * inv_dx

    dx2 = dx * dx
    om4 = omega * 0.25
    one_m = 1.0 - omega
    for _ in range(iters):
        bnd_c(phi, nxi, nyi)
        for j in range(1, nyi + 1):
            for i in range(1, nxi + 1):
                phi[j, i] = one_m * phi[j, i] + om4 * (
                    phi[j, i - 1] + phi[j, i + 1]
                    + phi[j - 1, i] + phi[j + 1, i]
                    - dx2 * div[j, i]
                )
    bnd_c(phi, nxi, nyi)

    for j in range(1, nyi + 1):
        for i in range(2, nxi + 1):
            u[j, i] -= (phi[j, i] - phi[j, i - 1]) * inv_dx
    for j in range(2, nyi + 1):
        for i in range(1, nxi + 1):
            v[j, i] -= (phi[j, i] - phi[j - 1, i]) * inv_dx
    bnd_u(u, nxi, nyi)
    bnd_v(v, nxi, nyi)


@njit(cache=True, fastmath=True)
def max_divergence(u, v, dx, nxi, nyi):
    """Diagnostic only: peak |div u| in 1/s, reported to the website."""
    inv_dx = 1.0 / dx
    m = 0.0
    for j in range(1, nyi + 1):
        for i in range(1, nxi + 1):
            d = abs((u[j, i + 1] - u[j, i]) + (v[j + 1, i] - v[j, i])) * inv_dx
            if d > m:
                m = d
    return m


@njit(cache=True, fastmath=True)
def buoyancy_and_drag(u, v, T, k, drag, dt, dx, nxi, nyi):
    """
    In-plane density force plus bulk drag.

    Honest scope, stated once: the domain is a HORIZONTAL slice through the
    gallery at occupant height, so vertical buoyancy is out of plane. What k
    models is the in-plane part - the horizontal density gradient that drives
    a gravity current, warm air spreading away from a warm source and cold air
    pooling toward a cold one. That is a Boussinesq reduction, not a 3D plume.
    drag stands in for the momentum this slice loses to floor, ceiling and the
    third dimension. Everything else in this solver is solved as written.
    """
    inv_dx = 1.0 / dx
    d = 1.0 - drag * dt
    for j in range(1, nyi + 1):
        for i in range(2, nxi + 1):
            u[j, i] = (u[j, i] - k * (T[j, i] - T[j, i - 1]) * inv_dx * dt) * d
    for j in range(2, nyi + 1):
        for i in range(1, nxi + 1):
            v[j, i] = (v[j, i] - k * (T[j, i] - T[j - 1, i]) * inv_dx * dt) * d
    bnd_u(u, nxi, nyi)
    bnd_v(v, nxi, nyi)


@njit(cache=True, fastmath=True)
def envelope_exchange(T, H, t_out, rh_out, k, dt, nxi, nyi):
    """Infiltration through the envelope. Strongest in the perimeter ring."""
    for j in range(1, nyi + 1):
        for i in range(1, nxi + 1):
            edge = i < 5 or i > nxi - 4 or j < 5 or j > nyi - 4
            w = k * dt * (3.0 if edge else 0.35)
            if w > 0.5:
                w = 0.5
            T[j, i] += (t_out - T[j, i]) * w
            H[j, i] += (rh_out - H[j, i]) * w


@njit(cache=True, fastmath=True)
def inject_momentum(u, v, cx, cy, rot_rad, speed, dx, radius_m, nxi, nyi):
    """Blend the faces around a unit toward its discharge velocity."""
    if speed == 0.0:
        return
    tu = math.cos(rot_rad) * speed
    tv = math.sin(rot_rad) * speed
    r = int(radius_m / dx) + 1
    rr = float(r * r)
    i0 = int(cx / dx) + 1
    j0 = int(cy / dx) + 1
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            d2 = float(di * di + dj * dj)
            if d2 > rr:
                continue
            w = math.exp(-d2 / (0.5 * rr + 1e-9))
            if w > 1.0:
                w = 1.0
            j = j0 + dj
            i = i0 + di
            if 1 <= j <= nyi and 2 <= i <= nxi:
                u[j, i] += (tu - u[j, i]) * w
            if 2 <= j <= nyi and 1 <= i <= nxi:
                v[j, i] += (tv - v[j, i]) * w


@njit(cache=True, fastmath=True)
def inject_scalar(T, H, cx, cy, heat_w, moist_gs, dt, dx,
                  cell_heat_cap, cell_mass, radius_m, nxi, nyi):
    """Distribute thermal and moisture output over a unit's footprint."""
    if heat_w == 0.0 and moist_gs == 0.0:
        return
    r = int(radius_m / dx) + 1
    rr = float(r * r)
    i0 = int(cx / dx) + 1
    j0 = int(cy / dx) + 1
    wsum = 0.0
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            d2 = float(di * di + dj * dj)
            if d2 <= rr:
                wsum += math.exp(-d2 / (0.5 * rr + 1e-9))
    if wsum <= 0.0:
        return
    for dj in range(-r, r + 1):
        j = j0 + dj
        if j < 1 or j > nyi:
            continue
        for di in range(-r, r + 1):
            i = i0 + di
            if i < 1 or i > nxi:
                continue
            d2 = float(di * di + dj * dj)
            if d2 > rr:
                continue
            w = math.exp(-d2 / (0.5 * rr + 1e-9)) / wsum
            if heat_w != 0.0:
                T[j, i] += (heat_w * w * dt) / cell_heat_cap
            if moist_gs != 0.0:
                # ~1 g water per kg dry air is ~6 %RH at room conditions
                H[j, i] += (moist_gs * w * dt / cell_mass) * 1000.0 * 6.0
                if H[j, i] < 2.0:
                    H[j, i] = 2.0
                if H[j, i] > 100.0:
                    H[j, i] = 100.0


@njit(cache=True, fastmath=True)
def compute_light(L, lx, ly, lint, dx, h, ambient, cd_max, nxi, nyi):
    """
    Point-source illuminance on the horizontal plane:
        E = I * cos(theta) / d^2,   cos(theta) = h / d
    Real photometry, not a painted radial gradient. Light is the one field in
    the room the air cannot carry, so it is never advected.
    """
    n = lx.shape[0]
    for j in range(1, nyi + 1):
        for i in range(1, nxi + 1):
            x = (i - 0.5) * dx
            y = (j - 0.5) * dx
            tot = ambient
            for k in range(n):
                ddx = x - lx[k]
                ddy = y - ly[k]
                d2 = ddx * ddx + ddy * ddy + h * h
                d = math.sqrt(d2)
                tot += (lint[k] * cd_max) * h / (d2 * d)
            L[j, i] = tot
    bnd_c(L, nxi, nyi)


@njit(cache=True, fastmath=True)
def speed_field(out, u, v, nxi, nyi):
    """Cell-centred wind speed from the surrounding face velocities."""
    for j in range(1, nyi + 1):
        for i in range(1, nxi + 1):
            uc = 0.5 * (u[j, i] + u[j, i + 1])
            vc = 0.5 * (v[j, i] + v[j + 1, i])
            out[j, i] = math.sqrt(uc * uc + vc * vc)
    bnd_c(out, nxi, nyi)


@njit(cache=True, fastmath=True)
def zone_means(f, nxi, nyi):
    """Mean of the interior over the 3x2 zone grid. Returns zones 1..6."""
    out = np.zeros(6, dtype=np.float64)
    cnt = np.zeros(6, dtype=np.float64)
    for j in range(1, nyi + 1):
        zr = 0 if (j - 1) < nyi // 2 else 1
        for i in range(1, nxi + 1):
            zc = (i - 1) * 3 // nxi
            if zc > 2:
                zc = 2
            z = zr * 3 + zc
            out[z] += f[j, i]
            cnt[z] += 1.0
    for z in range(6):
        if cnt[z] > 0.0:
            out[z] /= cnt[z]
    return out


@njit(cache=True, fastmath=True)
def to_uint8(dst, f, lo, hi, nxi, nyi):
    """Quantise the interior to bytes for transport. Row 0 = front of room."""
    inv = 255.0 / (hi - lo) if hi > lo else 0.0
    for j in range(0, nyi):
        for i in range(0, nxi):
            val = (f[j + 1, i + 1] - lo) * inv
            if val < 0.0:
                val = 0.0
            if val > 255.0:
                val = 255.0
            dst[j * nxi + i] = np.uint8(val)


# =============================================================================
# DATA SOURCES
# =============================================================================


def dig(obj, path):
    """Walk a dot-path into decoded JSON. Numeric parts index into lists."""
    cur = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def synthetic_value(shape, seed, t):
    """Deterministic offline feed. Same t always gives the same number."""
    rng = math.sin(seed * 12.9898) * 43758.5453
    ph = (rng - math.floor(rng)) * math.tau
    if shape == "wave":
        base = 50.0 + 34.0 * math.sin(t / 47.0 + ph)
        jitter = 6.0 * math.sin(t / 6.3 + ph * 2.0)
        return max(0.0, min(100.0, base + jitter))
    if shape == "walk":
        # sum of incommensurate sines: wanders like a walk, but replayable
        v = 50.0
        for k in range(1, 7):
            v += (26.0 / k) * math.sin(t / (9.0 * k) + ph * k)
        return max(0.0, min(100.0, v))
    if shape == "spike":
        q = math.sin(t / 3.1 + ph) * math.sin(t / 7.7 + ph * 3.0)
        q = q * q * q * q
        return max(0.0, min(100.0, 8.0 + 100.0 * q))
    if shape == "daily":
        # 24 h compressed into 240 s so a class can watch a whole day
        return 50.0 - 46.0 * math.cos(t * math.tau / 240.0 + ph)
    return 50.0


HTTP_HEADERS = {"User-Agent": "IARC425-ExhibitionTemplate/1.0"}


class Selection:
    """The one cocktail picked in the page's search bar (or None)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cocktail = None          # dict from the 24cocktails recipe api
        self.changed_at = None

    def set(self, rec):
        with self.lock:
            self.cocktail = rec
            self.changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def get(self):
        with self.lock:
            return None if self.cocktail is None else dict(self.cocktail)

    def field(self, name, apis):
        """Value of one "selection" api field, or None if nothing applies."""
        c = self.get()
        if c is None:
            return None
        base = (c.get("base") or "").lower()
        if name == "abv":
            return float(c.get("abv") or 0)
        if name == "spirit_rh":
            feed = SPIRIT_ORIGIN_RH.get(base)
            return None if feed is None else apis.get(feed)
        if name.startswith("method:"):
            return 1.0 if (c.get("method") or "").lower() == name[7:] else 0.0
        if name.startswith("region:"):
            ties = SPIRIT_REGION_TIE.get(base)
            if not ties:
                return None        # no spotlight for other / non-alcoholic
            return float(ties.get(name[7:], 0.0))
        return None

    def summary(self, apis):
        c = self.get()
        if c is None:
            return None
        base = (c.get("base") or "").lower()
        return {
            "slug": c.get("slug"), "name": c.get("name_en") or c.get("name"),
            "base": base, "abv": c.get("abv"), "method": c.get("method"),
            "glass": c.get("glass"), "categories": c.get("categories"),
            "ingredients": [i.get("name") if isinstance(i, dict) else i
                            for i in (c.get("ingredients") or [])],
            "photo": c.get("photo"), "url": c.get("url"),
            "spirit_origin": SPIRIT_ORIGIN_LABEL.get(base),
            "spirit_rh": self.field("spirit_rh", apis),
            "selected_utc": self.changed_at,
        }


class CocktailCatalog:
    """Every recipe on 24cocktails.com (about 7,200), for the dropdown.

    The first launch pages through the whole catalogue in the background
    (~2 minutes, polite to the API's rate limit) and saves it to
    exports/cocktail_catalog.json; later launches load that file instantly
    and refresh it in the background if it is older than CATALOG_MAX_AGE_DAYS.
    """

    KEEP = ("slug", "name_en", "base", "abv", "method", "glass", "ingredients")

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.items = []
        self.total = None
        self.complete = False
        self.status = "not loaded"

    @staticmethod
    def sort_key(c):
        import re
        name = c.get("name_en") or c.get("slug") or ""
        return re.sub(r"^[^0-9A-Za-z]+", "", name).lower()

    def load_cache(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            items = data["items"]
            with self.lock:
                self.items = sorted(items, key=self.sort_key)
                self.total = data.get("total", len(items))
                self.complete = True
                self.status = "loaded %d from cache" % len(items)
            age_days = (time.time() - os.path.getmtime(self.path)) / 86400.0
            return age_days
        except Exception:
            return None

    def fetch_all(self):
        import requests
        items, offset, total = [], 0, None
        try:
            while total is None or offset < total:
                r = requests.get(COCKTAIL_API + "/search",
                                 params={"q": "", "limit": 50, "offset": offset},
                                 timeout=15, headers=HTTP_HEADERS)
                r.raise_for_status()
                data = r.json()
                total = int(data.get("total", 0))
                page = data.get("results", [])
                if not page:
                    break
                items.extend({k: c.get(k) for k in self.KEEP} for c in page)
                offset += len(page)
                with self.lock:
                    if not self.complete:           # show progress live
                        self.items = sorted(items, key=self.sort_key)
                        self.total = total
                        self.status = "downloading %d / %d" % (len(items), total)
                time.sleep(0.6)                      # stay under 120 req/min
            with self.lock:
                self.items = sorted(items, key=self.sort_key)
                self.total = len(items)
                self.complete = True
                self.status = "loaded %d from 24cocktails.com" % len(items)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"total": len(items), "items": items}, fh)
        except Exception as exc:
            with self.lock:
                self.status = "download stopped (%s) - %d loaded" % (
                    type(exc).__name__, len(self.items))

    def start(self):
        age = self.load_cache()
        if age is None or age > CATALOG_MAX_AGE_DAYS:
            threading.Thread(target=self.fetch_all, daemon=True).start()

    def search(self, q):
        q = q.lower()
        with self.lock:
            items = list(self.items)
            info = {"total": self.total, "complete": self.complete,
                    "status": self.status, "loaded": len(self.items)}
        if q:
            def hit(c):
                if q in (c.get("name_en") or "").lower():
                    return True
                return any(q in (i or "").lower() for i in (c.get("ingredients") or []))
            items = [c for c in items if hit(c)]
        return items, info


def read_rhino_points(path):
    """Minimal reader for point objects in a Rhino 8 .3dm (no rhino3dm
    needed). Returns {layer_name_lower: [(x, y), ...]} or raises."""
    import re
    import struct
    with open(path, "rb") as fh:
        b = fh.read()
    if not b.startswith(b"3D Geometry File Format"):
        raise ValueError("not a .3dm file")

    def chunks(off, end):
        while off + 12 <= end:
            tc, = struct.unpack_from("<I", b, off)
            v, = struct.unpack_from("<q", b, off + 4)
            if tc & 0x80000000:
                yield tc, None
                off += 12
                continue
            if v < 0 or off + 12 + v > end:
                return
            yield tc, (off + 12, off + 12 + v)
            off += 12 + v

    top = {tc: r for tc, r in chunks(32, len(b))}
    names = []
    for tc, r in chunks(*top[0x10000011]):          # layer table
        if r:
            m = re.findall(rb"(?:[\x20-\x7e]\x00){2,}", b[r[0]:r[1]])
            names.append(m[0].decode("utf-16le").lower() if m else "")
    out = {}
    for tc, r in chunks(*top[0x10000013]):          # object table
        if tc != 0x20008070:
            continue
        data = att = None
        for t2, r2 in chunks(*r):
            if t2 == 0x27FFA:
                for t3, r3 in chunks(*r2):
                    if t3 == 0x2FFFC:
                        data = b[r3[0]:r3[1]]
            elif t2 == 0x2008072:
                att = b[r2[0]:r2[1]]
        # ON_Point: 1 version byte, 3 doubles, 4-byte crc
        if data is None or len(data) != 29 or att is None or len(att) < 21:
            continue
        lay, = struct.unpack_from("<i", att, 17)
        if 0 <= lay < len(names):
            x, y, _ = struct.unpack_from("<3d", data, 1)
            out.setdefault(names[lay], []).append((x, y))
    return out


def _face_centre(x, y):
    return round(math.degrees(math.atan2(ROOM_D / 2 - y, ROOM_W / 2 - x)) % 360, 1)


def apply_rhino_layout():
    """Overwrite PLACEMENTS / CAN_LIGHTS x, y, rot from the Rhino points."""
    path = os.path.join(HERE, RHINO_FILE)
    if not RHINO_SYNC:
        return "off"
    try:
        pts = read_rhino_points(path)
    except Exception as exc:
        return "not read (%s) - using PLACEMENTS as written" % exc
    by_id = {p["id"]: p for p in PLACEMENTS}
    notes = []
    fans = pts.get("fans", [])
    if len(fans) == 6:
        cx = ROOM_W / 2
        left = sorted([p for p in fans if p[0] < cx], key=lambda q: -q[1])
        right = sorted([p for p in fans if p[0] >= cx], key=lambda q: -q[1])
        if len(left) == 3 and len(right) == 3:
            for (x, y), m in zip(left + right, ["shake", "build", "stir",
                                               "blend", "layer", "muddle"]):
                p = by_id["fan_" + m]
                p["x"], p["y"], p["rot"] = round(x, 3), round(y, 3), _face_centre(x, y)
            notes.append("6 fans")
    hums = sorted(pts.get("humidifier", []), key=lambda q: -q[1])
    if len(hums) == 2:
        for (x, y), sid in zip(hums, ["humidifier_top", "humidifier_bottom"]):
            p = by_id[sid]
            p["x"], p["y"], p["rot"] = round(x, 3), round(y, 3), _face_centre(x, y)
        notes.append("2 humidifiers")
    hp = pts.get("heat pump", [])
    if hp:
        by_id["heatpump_abv"]["x"], by_id["heatpump_abv"]["y"] = [round(v, 3) for v in hp[0]]
        notes.append("heat pump")
    lights = pts.get("lights", [])
    if len(lights) == 6:
        ymid = sum(q[1] for q in lights) / 6.0
        front = sorted([q for q in lights if q[1] < ymid], key=lambda q: q[0])
        back = sorted([q for q in lights if q[1] >= ymid], key=lambda q: q[0])
        if len(front) == 3 and len(back) == 3:
            for c, (x, y) in zip(CAN_LIGHTS, front + back):
                c["x"], c["y"] = round(x, 3), round(y, 3)
            notes.append("6 lights")
    return "synced " + ", ".join(notes) if notes else "no matching points found"


def reduce_value(raw, a):
    """Turn what the dot-path found into one number (see API_REGISTRY notes)."""
    how = a.get("reduce")
    if how == "mean":
        vals = [float(v) for v in raw if v is not None]
        return sum(vals) / len(vals)
    if how == "sum_keys":
        keys = set(k.lower() for k in a["keys"])
        return float(sum(v for k, v in raw.items() if k.lower() in keys))
    return float(raw)


def fetch_cocktail_abv(requests, names):
    """Average ABV of every 24cocktails recipe whose English name contains one
    of `names` (so variants count too). Recipes listed at 0 % are unknown and
    skipped."""
    total, n = 0.0, 0
    for name in names:
        r = requests.get("https://24cocktails.com/api/v1/search",
                         params={"q": name, "limit": 50},
                         timeout=10, headers=HTTP_HEADERS)
        r.raise_for_status()
        for rec in r.json().get("results", []):
            abv = rec.get("abv") or 0
            if abv > 0 and name.lower() in (rec.get("name_en") or "").lower():
                total += abv
                n += 1
    if n == 0:
        raise ValueError("no recipes matched")
    return total / n


class ApiPool:
    """Holds the current value of every registered source."""

    def __init__(self, registry):
        self.registry = registry
        self.lock = threading.Lock()
        self.values = {}
        self.t0 = time.time()
        for a in registry:
            self.values[a["id"]] = {
                "value": None,
                "at": None,
                "status": "waiting",
                "history": deque(maxlen=300),
            }

    def snapshot(self):
        with self.lock:
            return {
                k: {
                    "value": v["value"],
                    "at": v["at"],
                    "status": v["status"],
                    "history": list(v["history"]),
                }
                for k, v in self.values.items()
            }

    def get(self, api_id):
        with self.lock:
            rec = self.values.get(api_id)
            return None if rec is None else rec["value"]

    def _store(self, api_id, value, status):
        with self.lock:
            rec = self.values[api_id]
            rec["value"] = value
            rec["status"] = status
            rec["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if value is not None:
                rec["history"].append(round(float(value), 4))

    def tick_synthetic(self):
        t = time.time() - self.t0
        for a in self.registry:
            if a["kind"] == "synthetic":
                self._store(a["id"], synthetic_value(a["shape"], a["seed"], t), "ok")

    def tick_selection(self, selection):
        for a in self.registry:
            if a["kind"] == "selection":
                v = selection.field(a["field"], self)
                self._store(a["id"], v, "ok" if v is not None else "no cocktail")

    def run_http(self):
        """One background thread polls every http source on its own interval."""
        import requests

        polled = ("http", "cocktail_abv")
        next_due = {a["id"]: 0.0 for a in self.registry if a["kind"] in polled}
        while True:
            for a in self.registry:
                if a["kind"] not in polled:
                    continue
                now = time.time()
                if a["kind"] == "http" and not a.get("url"):
                    self._store(a["id"], None, "no url set")
                    next_due[a["id"]] = now + 30.0
                    continue
                if now < next_due[a["id"]]:
                    continue
                try:
                    if a["kind"] == "cocktail_abv":
                        val = fetch_cocktail_abv(requests, a["names"])
                    else:
                        r = requests.get(a["url"], timeout=10, headers=HTTP_HEADERS)
                        r.raise_for_status()
                        val = reduce_value(dig(r.json(), a["path"]), a)
                    self._store(a["id"], val, "ok")
                except Exception as exc:
                    self._store(a["id"], None, "error: %s" % type(exc).__name__)
                next_due[a["id"]] = now + float(a.get("every", 120))
            time.sleep(1.0)


# =============================================================================
# RULE ENGINE
# =============================================================================


def rule_text(rule):
    """Human-readable form of one rule's condition, used in the exports."""
    op = rule["op"]
    if op == "map":
        return "%s mapped %g..%g -> %g..%g" % (
            rule["api"], rule["value"], rule["value2"],
            rule["otherwise"], rule["then"])
    if op == "rank":
        return "%s ranked among %s -> %s" % (
            rule["api"], ", ".join(rule["among"]), rule["by_rank"])
    if op == "share_of_max":
        txt = "%s / max(%s) x %g, min %g" % (
            rule["api"], ", ".join(rule["among"]), rule["then"],
            rule["otherwise"])
        if rule.get("boost_api"):
            txt += " + %g x %s (else x %g)" % (rule.get("boost", 0.0),
                                                rule["boost_api"],
                                                rule.get("dim", 1.0))
        return txt
    if rule["op"] in ("between", "outside"):
        lo, hi = sorted([rule["value"],
                         rule["value2"] if rule["value2"] is not None
                         else rule["value"]])
        return "%s %s %g and %g" % (rule["api"], rule["op"], lo, hi)
    return "%s %s %g" % (rule["api"], rule["op"], rule["value"])


def eval_rule(rule, value, lookup=None):
    """Returns (setpoint, condition_is_true_or_None).

    lookup(api_id) -> current value; needed by "rank" and "share_of_max",
    which compare this api against others. For the continuous ops the
    condition is reported as True whenever live data is driving the system.
    """
    if value is None:
        return rule.get("fallback", rule["otherwise"]), None
    op = rule["op"]
    if op == "map":
        lo, hi = float(rule["value"]), float(rule["value2"])
        t = 0.0 if hi == lo else (value - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
        return rule["otherwise"] + t * (rule["then"] - rule["otherwise"]), True
    if op in ("rank", "share_of_max"):
        others = [lookup(k) if lookup else None for k in rule["among"]]
        if any(v is None for v in others):
            return rule.get("fallback", rule["otherwise"]), None
        if op == "rank":
            rank = sum(1 for v in others if v > value)
            table = rule["by_rank"]
            return table[min(rank, len(table) - 1)], True
        peak = max(others) or 1.0
        sp = max(rule["otherwise"], rule["then"] * value / peak)
        if rule.get("boost_api") and lookup:
            tie = lookup(rule["boost_api"])
            if tie is not None:                      # a spotlight is active
                if tie > 0:
                    sp = min(1.0, sp + rule.get("boost", 0.0) * tie)
                else:
                    sp = sp * rule.get("dim", 1.0)
        return sp, True
    a = rule["value"]
    b = rule.get("value2")
    if op == ">":
        ok = value > a
    elif op == "<":
        ok = value < a
    elif op == ">=":
        ok = value >= a
    elif op == "<=":
        ok = value <= a
    elif op == "between":
        lo, hi = sorted([a, b if b is not None else a])
        ok = lo <= value <= hi
    elif op == "outside":
        lo, hi = sorted([a, b if b is not None else a])
        ok = value < lo or value > hi
    else:
        ok = False
    return (rule["then"] if ok else rule["otherwise"]), ok


# =============================================================================
# SIMULATION
# =============================================================================


class Gallery:
    def __init__(self, apis):
        self.apis = apis
        self.lock = threading.Lock()
        self.mode = "simulate"
        self.speed = 1.0
        self.sim_time = 0.0
        self.frame = 0
        self.step_ms = 0.0

        cc = lambda: np.zeros((NY, NX), dtype=np.float64)        # noqa: E731
        fu = lambda: np.zeros((NY, NX + 1), dtype=np.float64)    # noqa: E731
        fv = lambda: np.zeros((NY + 1, NX), dtype=np.float64)    # noqa: E731
        self.u, self.u0 = fu(), fu()
        self.v, self.v0 = fv(), fv()
        self.T, self.T0 = cc(), cc()
        self.H, self.H0 = cc(), cc()
        self.L, self.spd = cc(), cc()
        self.phi, self.div = cc(), cc()
        self.T[:] = 21.0
        self.H[:] = 45.0
        self.max_div = 0.0

        self.lx = np.array([c["x"] for c in CAN_LIGHTS])
        self.ly = np.array([c["y"] for c in CAN_LIGHTS])
        self.lint = np.full(len(CAN_LIGHTS), DEFAULT_SETPOINT["can_light"])

        self.systems = []
        self.rules = {}
        self.history = {}
        self.last_hist = 0.0
        self.load(PLACEMENTS, RULES)

    # -- configuration ----------------------------------------------------

    def load(self, placements, rules):
        systems = []
        for p in placements:
            systems.append(
                {
                    "id": p["id"],
                    "type": p["type"],
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "rot": float(p.get("rot", 0.0)),
                    "power": float(p.get("power", 1.0)),
                    "label": p.get("label", p["id"]),
                    "movable": True,
                    "omni": bool(p.get("omni", False)),
                    "drive": 1.0,      # 0 = switched off by idle_when_none
                    "radius": 0.55,    # heat-field radius (omni heat pump)
                    "setpoint": DEFAULT_SETPOINT[p["type"]],
                    "output": 0.0,
                }
            )
        for c in CAN_LIGHTS:
            systems.append(
                {
                    "id": c["id"],
                    "type": "can_light",
                    "x": c["x"],
                    "y": c["y"],
                    "rot": 0.0,
                    "power": 1.0,
                    "label": "Can light %d" % c["zone"],
                    "zone": c["zone"],
                    "movable": False,
                    "omni": False,
                    "drive": 1.0,
                    "radius": 0.0,
                    "setpoint": DEFAULT_SETPOINT["can_light"],
                    "output": DEFAULT_SETPOINT["can_light"],
                }
            )
        self.systems = systems
        self.rules = {r["system"]: dict(r) for r in rules}
        self.history = {
            s["id"]: deque(maxlen=240) for s in systems
        }

    def by_id(self, sid):
        for s in self.systems:
            if s["id"] == sid:
                return s
        return None

    # -- one published frame ----------------------------------------------

    def step(self):
        t_start = time.perf_counter()
        dt = SIM_DT * self.speed

        for s in self.systems:
            rule = self.rules.get(s["id"])
            if rule:
                val = self.apis.get(rule["api"])
                sp, _ = eval_rule(rule, val, self.apis.get)
                s["setpoint"] = float(sp)
                s["drive"] = 0.0 if (val is None and rule.get("idle_when_none")) else 1.0
            if s["omni"]:
                k = max(0.0, min(1.5, s["setpoint"] / HEATPUMP_FULL_ABV)) * s["drive"]
                r0, r1 = HEATPUMP_RADIUS_M
                s["radius"] = r0 + (r1 - r0) * k if s["drive"] else 0.0

        for _ in range(SUBSTEPS):
            self._substep(dt)
            self.sim_time += dt

        self.lint[:] = [
            self.by_id(c["id"])["output"] for c in CAN_LIGHTS
        ]
        compute_light(
            self.L, self.lx, self.ly, self.lint, DX, ROOM_H,
            AMBIENT_LUX, CANDELA_MAX, NXI, NYI,
        )
        speed_field(self.spd, self.u, self.v, NXI, NYI)
        self.max_div = max_divergence(self.u, self.v, DX, NXI, NYI)

        now = time.time()
        if now - self.last_hist > 0.35:
            self.last_hist = now
            self._sample_history()

        self.frame += 1
        self.step_ms = (time.perf_counter() - t_start) * 1000.0

    def _substep(self, dt):
        # --- plant: proportional control toward each setpoint -------------
        for sysm in self.systems:
            if sysm["type"] == "can_light":
                sysm["output"] = max(0.0, min(1.0, sysm["setpoint"] * sysm["power"]))
                continue

            i = max(4, min(NXI - 3, int(sysm["x"] / DX) + 1))
            j = max(4, min(NYI - 3, int(sysm["y"] / DX) + 1))
            rot = math.radians(sysm["rot"])

            if sysm["type"] == "heat_pump" and sysm["omni"]:
                # heat in ALL directions: no jet, a disc of heat whose radius
                # and strength scale with ABV (radius set in step())
                if sysm["drive"] <= 0.0:
                    sysm["output"] = 0.0
                    continue
                rad = max(0.3, sysm["radius"])
                ri = max(2, int(rad / DX))
                local = float(self.T[max(1, j - ri):j + ri + 1,
                                     max(1, i - ri):i + ri + 1].mean())
                err = sysm["setpoint"] - local
                if abs(err) < HEATPUMP_DEADBAND_C:
                    err = 0.0
                strength = max(0.15, min(1.5, sysm["setpoint"] / HEATPUMP_FULL_ABV))
                target = (max(-1.0, min(1.0, err / 2.5)) * HEATPUMP_W
                          * sysm["power"] * strength)
                ease = min(1.0, dt / HEATPUMP_SMOOTH_S)
                q = sysm["output"] + (target - sysm["output"]) * ease
                sysm["output"] = q
                inject_scalar(self.T, self.H, sysm["x"], sysm["y"], q, 0.0,
                              dt, DX, CELL_HEAT_CAP, CELL_MASS, rad, NXI, NYI)
                continue

            if sysm["type"] == "heat_pump":
                local = float(self.T[j - 3:j + 4, i - 3:i + 4].mean())
                err = sysm["setpoint"] - local
                q = max(-1.0, min(1.0, err / 2.5)) * HEATPUMP_W * sysm["power"]
                sysm["output"] = q
                # a heat pump is also a fan: it throws its own air
                inject_momentum(self.u, self.v, sysm["x"], sysm["y"], rot,
                                1.6 * sysm["power"], DX, 0.55, NXI, NYI)
                inject_scalar(self.T, self.H, sysm["x"], sysm["y"], q, 0.0,
                              dt, DX, CELL_HEAT_CAP, CELL_MASS, 0.55, NXI, NYI)

            elif sysm["type"] == "humidifier":
                if sysm["drive"] <= 0.0:
                    sysm["output"] = 0.0
                    continue
                local = float(self.H[j - 3:j + 4, i - 3:i + 4].mean())
                err = sysm["setpoint"] - local
                g = max(-1.0, min(1.0, err / 8.0)) * HUMIDIFIER_GS * sysm["power"]
                sysm["output"] = g
                inject_momentum(self.u, self.v, sysm["x"], sysm["y"], rot,
                                0.9 * sysm["power"], DX, 0.45, NXI, NYI)
                inject_scalar(self.T, self.H, sysm["x"], sysm["y"], 0.0, g,
                              dt, DX, CELL_HEAT_CAP, CELL_MASS, 0.45, NXI, NYI)

            elif sysm["type"] == "fan":
                spd = max(0.0, min(FAN_MAX_MS, sysm["setpoint"])) * sysm["power"]
                sysm["output"] = spd
                inject_momentum(self.u, self.v, sysm["x"], sysm["y"], rot,
                                spd, DX, 0.5, NXI, NYI)

        bnd_u(self.u, NXI, NYI)
        bnd_v(self.v, NXI, NYI)

        # --- momentum: force, advect, diffuse, project --------------------
        buoyancy_and_drag(self.u, self.v, self.T, BUOY_K, DRAG, dt, DX, NXI, NYI)

        self.u0[:] = self.u
        self.v0[:] = self.v
        advect_u(self.u, self.u0, self.v0, dt, DX, NXI, NYI)
        advect_v(self.v, self.u0, self.v0, dt, DX, NXI, NYI)
        bnd_u(self.u, NXI, NYI)
        bnd_v(self.v, NXI, NYI)

        a_v = VISCOSITY * dt / (DX * DX)
        self.u0[:] = self.u
        self.v0[:] = self.v
        diffuse_u(self.u, self.u0, a_v, DIFF_ITERS, NXI, NYI)
        diffuse_v(self.v, self.v0, a_v, DIFF_ITERS, NXI, NYI)

        project(self.u, self.v, self.phi, self.div, DX,
                PROJ_ITERS, SOR_OMEGA, NXI, NYI)

        # --- transported scalars ------------------------------------------
        self.T0[:] = self.T
        advect_c(self.T, self.T0, self.u, self.v, dt, DX, NXI, NYI)
        bnd_c(self.T, NXI, NYI)
        self.T0[:] = self.T
        diffuse_c(self.T, self.T0, DIFF_TEMP * dt / (DX * DX), DIFF_ITERS, NXI, NYI)

        self.H0[:] = self.H
        advect_c(self.H, self.H0, self.u, self.v, dt, DX, NXI, NYI)
        bnd_c(self.H, NXI, NYI)
        self.H0[:] = self.H
        diffuse_c(self.H, self.H0, DIFF_HUM * dt / (DX * DX), DIFF_ITERS, NXI, NYI)

        envelope_exchange(self.T, self.H, OUTDOOR_T, OUTDOOR_RH,
                          ENVELOPE_K, dt, NXI, NYI)
        np.clip(self.H, 2.0, 100.0, out=self.H)
        np.clip(self.T, -20.0, 60.0, out=self.T)
        bnd_c(self.T, NXI, NYI)
        bnd_c(self.H, NXI, NYI)

    def _sample_history(self):
        zt = zone_means(self.T, NXI, NYI)
        zh = zone_means(self.H, NXI, NYI)
        zs = zone_means(self.spd, NXI, NYI)
        zl = zone_means(self.L, NXI, NYI)
        for s in self.systems:
            if s["type"] == "heat_pump":
                val = self._probe(self.T, s)
            elif s["type"] == "humidifier":
                val = self._probe(self.H, s)
            elif s["type"] == "fan":
                val = self._probe(self.spd, s)
            else:
                val = float(zl[s["zone"] - 1])
            self.history[s["id"]].append(round(val, 3))
        self._zone_now = {
            "temperature": [round(float(x), 2) for x in zt],
            "humidity": [round(float(x), 2) for x in zh],
            "wind": [round(float(x), 3) for x in zs],
            "light": [round(float(x), 1) for x in zl],
        }

    def _probe(self, f, s):
        i = max(4, min(NX - 5, int(s["x"] / DX) + 1))
        j = max(4, min(NY - 5, int(s["y"] / DX) + 1))
        return float(f[j - 4:j + 5, i - 4:i + 5].mean())

    # -- what the website asks for ----------------------------------------

    def field_by_name(self, name):
        return {
            "temperature": self.T,
            "humidity": self.H,
            "wind": self.spd,
            "light": self.L,
        }[name]

    def frame_payload(self, field_name):
        with self.lock:
            f = self.field_by_name(field_name)
            lo, hi = FIELD_RANGES[field_name]
            buf = np.empty(NXI * NYI, dtype=np.uint8)
            to_uint8(buf, f, lo, hi, NXI, NYI)
            interior = f[1:-1, 1:-1]
            zones = getattr(self, "_zone_now", None) or {
                k: [0] * 6 for k in FIELD_RANGES
            }
            apis = self.apis.snapshot()

            sysout = []
            for s in self.systems:
                rule = self.rules.get(s["id"])
                cond = None
                if rule:
                    _, cond = eval_rule(rule, apis.get(rule["api"], {}).get("value"),
                                        lambda k: apis.get(k, {}).get("value"))
                sysout.append(
                    {
                        "id": s["id"],
                        "type": s["type"],
                        "label": s["label"],
                        "x": round(s["x"], 3),
                        "y": round(s["y"], 3),
                        "rot": s["rot"],
                        "power": s["power"],
                        "zone": s.get("zone"),
                        "setpoint": round(s["setpoint"], 3),
                        "output": round(s["output"], 5),
                        "reading": round(
                            self.history[s["id"]][-1] if self.history[s["id"]] else 0.0,
                            3,
                        ),
                        "history": list(self.history[s["id"]]),
                        "rule": rule,
                        "rule_true": cond,
                        "omni": s.get("omni", False),
                        "drive": s.get("drive", 1.0),
                        "radius": round(s.get("radius", 0.0), 3),
                    }
                )

            return {
                "frame": self.frame,
                "sim_time_s": round(self.sim_time, 1),
                "step_ms": round(self.step_ms, 2),
                "max_div": round(self.max_div, 6),
                "mode": self.mode,
                "speed": self.speed,
                "field": field_name,
                "range": [lo, hi],
                "nx": NXI,
                "ny": NYI,
                "min": round(float(interior.min()), 3),
                "max": round(float(interior.max()), 3),
                "mean": round(float(interior.mean()), 3),
                "data": base64.b64encode(buf.tobytes()).decode("ascii"),
                "zones": zones,
                "systems": sysout,
                "apis": apis,
                "cocktail": selection.summary(self.apis),
            }

    # -- exports ------------------------------------------------------------

    def export_payload(self):
        with self.lock:
            apis = self.apis.snapshot()
            zones = getattr(self, "_zone_now", None) or {
                k: [0] * 6 for k in FIELD_RANGES
            }
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            docs = {}
            for s in self.systems:
                rule = self.rules.get(s["id"])
                cond = None
                api_val = None
                if rule:
                    api_val = apis.get(rule["api"], {}).get("value")
                    _, cond = eval_rule(rule, api_val,
                                        lambda k: apis.get(k, {}).get("value"))
                zone = s.get("zone") or self.zone_of(s["x"], s["y"])
                docs[s["id"]] = {
                    "schema": "iarc425.system.v1",
                    "exported_utc": stamp,
                    "frame": self.frame,
                    "sim_time_s": round(self.sim_time, 1),
                    "system_id": s["id"],
                    "system_type": s["type"],
                    "label": s["label"],
                    "position_m": {"x": round(s["x"], 4), "y": round(s["y"], 4),
                                   "z": ROOM_H if s["type"] == "can_light" else 1.2},
                    "rotation_deg": s["rot"],
                    "power_scalar": s["power"],
                    "active": s.get("drive", 1.0) > 0 and (
                        s["type"] != "fan" or s["setpoint"] > 0),
                    "heat_radius_m": round(s.get("radius", 0.0), 3) if s.get("omni") else None,
                    "zone": zone,
                    "setpoint": round(s["setpoint"], 4),
                    "output": round(s["output"], 6),
                    "output_units": {
                        "heat_pump": "W_thermal",
                        "humidifier": "kg_per_s",
                        "fan": "m_per_s",
                        "can_light": "intensity_0_1",
                    }[s["type"]],
                    "reading": round(
                        self.history[s["id"]][-1] if self.history[s["id"]] else 0.0, 4
                    ),
                    "linked_api": None if not rule else {
                        "id": rule["api"],
                        "value": api_val,
                        "condition": rule_text(rule),
                        "condition_met": cond,
                        "then": rule["then"],
                        "otherwise": rule["otherwise"],
                    },
                    "series": list(self.history[s["id"]]),
                }
            room = {
                "schema": "iarc425.room.v1",
                "exported_utc": stamp,
                "room_m": {"w": ROOM_W, "d": ROOM_D, "h": ROOM_H},
                "zone_grid": {"cols": 3, "rows": 2},
                "zone_centres_m": [
                    {"zone": c["zone"], "x": round(c["x"], 4), "y": round(c["y"], 4)}
                    for c in CAN_LIGHTS
                ],
                "zone_values": zones,
                "field_ranges": FIELD_RANGES,
                "selected_cocktail": selection.summary(self.apis),
            }
            return docs, room

    def zone_of(self, x, y):
        col = min(2, max(0, int(x / (ROOM_W / 3.0))))
        row = 0 if y < ROOM_D / 2.0 else 1
        return row * 3 + col + 1


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__, static_folder=None)
apis = ApiPool(API_REGISTRY)
selection = Selection()
catalog = CocktailCatalog(os.path.join(EXPORT_DIR, "cocktail_catalog.json"))
RHINO_STATUS = apply_rhino_layout()
gallery = Gallery(apis)


@app.after_request
def relax_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/config")
def api_config():
    """Everything the page needs once at boot."""
    return jsonify(
        {
            "room": {"w": ROOM_W, "d": ROOM_D, "h": ROOM_H},
            "grid": {"nx": NXI, "ny": NYI, "dx": DX},
            "fields": FIELD_RANGES,
            "apis": [
                {k: a[k] for k in ("id", "label", "kind", "units", "range")}
                for a in API_REGISTRY
            ],
            "can_lights": CAN_LIGHTS,
            "defaults": DEFAULT_SETPOINT,
            "fan_max": FAN_MAX_MS,
            "placements": PLACEMENTS,
            "rules": RULES,
            "sim": {
                "dt": SIM_DT,
                "substeps": SUBSTEPS,
                "hz": FRAME_HZ,
                "room_s_per_real_s": round(SIM_DT * SUBSTEPS * FRAME_HZ, 2),
            },
        }
    )


@app.get("/api/frame")
def api_frame():
    field = request.args.get("field", "temperature")
    if field not in FIELD_RANGES:
        return jsonify({"error": "unknown field"}), 400
    return jsonify(gallery.frame_payload(field))


@app.post("/api/mode")
def api_mode():
    body = request.get_json(force=True)
    m = body.get("mode")
    if m not in ("place", "simulate"):
        return jsonify({"error": "mode must be place or simulate"}), 400
    with gallery.lock:
        gallery.mode = m
    return jsonify({"mode": m})


@app.post("/api/speed")
def api_speed():
    body = request.get_json(force=True)
    with gallery.lock:
        gallery.speed = max(0.0, min(4.0, float(body.get("speed", 1.0))))
    return jsonify({"speed": gallery.speed})


@app.post("/api/systems")
def api_systems():
    """
    Place mode writes the live layout here. Movable units may be moved,
    rotated and powered; can lights may only be re-powered.
    """
    body = request.get_json(force=True)
    with gallery.lock:
        for item in body.get("systems", []):
            s = gallery.by_id(item.get("id"))
            if s is None:
                continue
            if s["movable"]:
                s["x"] = max(0.15, min(ROOM_W - 0.15, float(item.get("x", s["x"]))))
                s["y"] = max(0.15, min(ROOM_D - 0.15, float(item.get("y", s["y"]))))
                s["rot"] = float(item.get("rot", s["rot"])) % 360.0
            s["power"] = max(0.0, min(1.0, float(item.get("power", s["power"]))))
            if "label" in item:
                s["label"] = str(item["label"])[:48]
    return jsonify({"ok": True})


@app.get("/api/placement")
def api_placement():
    """
    The 'Copy placement for Claude' payload. Returns a ready-to-paste Python
    block that replaces the two AI-AGENT edit zones in this file.
    """
    with gallery.lock:
        movable = [s for s in gallery.systems if s["movable"]]
        lines = [
            "# ==== PASTE OVER THE PLACEMENTS EDIT ZONE IN app.py ====",
            "PLACEMENTS = [",
        ]
        for s in movable:
            lines.append(
                '    {"id": "%s", "type": "%s", "x": %.2f, "y": %.2f, '
                '"rot": %d, "power": %.2f,%s "label": "%s"},'
                % (s["id"], s["type"], s["x"], s["y"], round(s["rot"]),
                   s["power"], ' "omni": True,' if s.get("omni") else "",
                   s["label"])
            )
        lines.append("]")
        lines.append("")
        lines.append("# ==== PASTE OVER THE RULES EDIT ZONE IN app.py ====")
        lines.append("RULES = [")
        for r in gallery.rules.values():
            lines.append("    %r," % (r,))
        lines.append("]")
        lines.append("")
        lines.append("# Can lights are fixed to the ceiling zone grid at:")
        for c in CAN_LIGHTS:
            lines.append(
                "#   %s  zone %d  x=%.2f  y=%.2f  power=%.2f"
                % (c["id"], c["zone"], c["x"], c["y"],
                   gallery.by_id(c["id"])["power"])
            )
        return Response("\n".join(lines), mimetype="text/plain")


@app.get("/api/cocktails")
def api_cocktails():
    """Dropdown + search bar. Uses the full local catalogue when it is
    loaded; while it is still downloading, a typed search also asks
    24cocktails directly so results appear immediately."""
    import requests
    q = request.args.get("q", "").strip()
    keep = ("slug", "name_en", "base", "abv", "method")
    items, info = catalog.search(q)
    if info["complete"] or not q:
        return jsonify(dict(info, results=[{k: c.get(k) for k in keep}
                                           for c in items]))
    try:
        r = requests.get(COCKTAIL_API + "/search",
                         params={"q": q, "limit": 50}, timeout=10,
                         headers=HTTP_HEADERS)
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = {"results": []}          # offline: show what is loaded so far
    seen = set(c["slug"] for c in items)
    extra = [c for c in data.get("results", []) if c.get("slug") not in seen]
    merged = sorted(items + extra, key=CocktailCatalog.sort_key)
    return jsonify(dict(info, results=[{k: c.get(k) for k in keep}
                                       for c in merged]))


@app.post("/api/cocktail")
def api_cocktail_select():
    """Select one cocktail by slug, or {"slug": null} to clear."""
    import requests
    body = request.get_json(force=True) or {}
    slug = body.get("slug")
    if not slug:
        selection.set(None)
    else:
        try:
            r = requests.get(COCKTAIL_API + "/recipe/" + slug, timeout=10,
                             headers=HTTP_HEADERS)
            r.raise_for_status()
            selection.set(r.json())
        except Exception as exc:
            return jsonify({"error": "could not load %s: %s"
                            % (slug, type(exc).__name__)}), 502
    apis.tick_selection(selection)
    summary = selection.summary(apis)
    try:   # hand-off file for Grasshopper / the Rhino script
        os.makedirs(EXPORT_DIR, exist_ok=True)
        with open(os.path.join(EXPORT_DIR, "selected_cocktail.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"selected_cocktail": summary}, fh, indent=2)
    except OSError:
        pass
    return jsonify({"cocktail": summary})


@app.get("/api/state")
def api_state():
    """Compact live state for the Rhino script: what each unit is doing."""
    with gallery.lock:
        units = [{
            "id": s["id"], "type": s["type"], "label": s["label"],
            "x": round(s["x"], 3), "y": round(s["y"], 3), "rot": s["rot"],
            "setpoint": round(s["setpoint"], 3), "output": round(s["output"], 5),
            "active": s.get("drive", 1.0) > 0 and (s["type"] != "fan" or s["setpoint"] > 0),
            "radius": round(s.get("radius", 0.0), 3),
            "api": (gallery.rules.get(s["id"]) or {}).get("api"),
            "api_value": apis.get((gallery.rules.get(s["id"]) or {}).get("api")),
        } for s in gallery.systems]
    return jsonify({"cocktail": selection.summary(apis),
                    "changed_utc": selection.changed_at,
                    "units": units, "rhino_sync": RHINO_STATUS})


@app.get("/api/export")
def api_export_preview():
    docs, room = gallery.export_payload()
    return jsonify({"room": room, "systems": docs})


@app.post("/api/export")
def api_export_write():
    """One JSON per system on disk, for the Grasshopper file-reader + timer."""
    docs, room = gallery.export_payload()
    os.makedirs(EXPORT_DIR, exist_ok=True)
    written = []
    for sid, doc in docs.items():
        path = os.path.join(EXPORT_DIR, "%s.json" % sid)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        written.append(os.path.basename(path))
    for name, payload in (("_room.json", room),
                          ("_all_systems.json", docs)):
        path = os.path.join(EXPORT_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        written.append(name)
    return jsonify({"ok": True, "dir": EXPORT_DIR, "files": written})


@app.post("/api/reset")
def api_reset():
    """Full restart of the piece: no cocktail, still air at 21 C / 45 %RH,
    every unit off (lights back to the brewery baseline), running at 1x."""
    selection.set(None)
    apis.tick_selection(selection)
    with gallery.lock:
        gallery.u[:] = 0.0
        gallery.v[:] = 0.0
        gallery.phi[:] = 0.0
        gallery.T[:] = 21.0
        gallery.H[:] = 45.0
        gallery.sim_time = 0.0
        gallery.speed = 1.0
        for s in gallery.systems:
            if s["type"] != "can_light":
                s["output"] = 0.0
        for k in gallery.history:
            gallery.history[k].clear()
    try:
        with open(os.path.join(EXPORT_DIR, "selected_cocktail.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"selected_cocktail": None}, fh, indent=2)
    except OSError:
        pass
    return jsonify({"ok": True, "speed": 1.0})


# =============================================================================
# BACKGROUND LOOPS
# =============================================================================


def sim_loop():
    period = 1.0 / FRAME_HZ
    while True:
        t = time.perf_counter()
        apis.tick_synthetic()
        apis.tick_selection(selection)
        with gallery.lock:
            gallery.step()
        rest = period - (time.perf_counter() - t)
        if rest > 0:
            time.sleep(rest)


def warm_up():
    """Force Numba to compile before the first request, not during it."""
    t = time.perf_counter()
    with gallery.lock:
        gallery.step()
    return (time.perf_counter() - t) * 1000.0


def main():
    print("=" * 70)
    print(" IARC 425  Exhibition Template  -  Python backend")
    print("=" * 70)
    print(" grid          %d x %d cells at %.3f m" % (NXI, NYI, DX))
    print(" room          %.1f x %.1f x %.1f m, 6 zones" % (ROOM_W, ROOM_D, ROOM_H))
    print(" room time     %.1f s per real second"
          % (SIM_DT * SUBSTEPS * FRAME_HZ))
    print(" rhino layout  %s" % RHINO_STATUS)
    print(" compiling Numba kernels ...", flush=True)
    ms = warm_up()
    print(" first step    %.0f ms (compile). steady state next." % ms)
    ms2 = warm_up()
    print(" steady step   %.2f ms per frame" % ms2)

    threading.Thread(target=sim_loop, daemon=True).start()
    threading.Thread(target=apis.run_http, daemon=True).start()
    catalog.start()

    print("")
    print("  open  ->  http://127.0.0.1:5000")
    print("")
    app.run(host="127.0.0.1", port=5000, threaded=True,
            debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
