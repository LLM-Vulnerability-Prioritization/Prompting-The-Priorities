# System Modules
import os
from dotenv import load_dotenv
import ssvc
import json
import re
import requests
import gc
import pytz
import ast
from termcolor import cprint
from datetime import datetime

# Data Processing Modules
import pandas as pd
import numpy as np
from math import pi

# Concurrent task processing modules
import multiprocessing
from functools import partial

# Model Evaluation Modules
from sklearn.metrics import precision_score, recall_score, accuracy_score, fbeta_score, average_precision_score, matthews_corrcoef, f1_score
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

# Dashboarding Modules
from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DataTable, TableColumn, FactorRange, Range1d, HoverTool, Tabs, TabPanel, LinearColorMapper, ColorBar, Div, GroupingInfo, NumberFormatter, Legend
from bokeh.palettes import RdYlGn, Category10
from bokeh.transform import transform, factor_cmap
from bokeh.layouts import column, row, gridplot

# Env vars
load_dotenv()
openrouter_url = os.getenv('OPENROUTER_URL')
openrouter_key = os.getenv('OPENROUTER_KEY')