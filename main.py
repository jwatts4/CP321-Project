import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings

warnings.filterwarnings('ignore', category=FutureWarning) # supress PyArrow warning from pandas

df = pd.read_csv("data/region_hospital_icu_covid_data.csv")

# Convert to datetime
df["date"] = pd.to_datetime(df["date"])

# Set date as the dataframe's index
df.set_index('date', inplace=True)

integer_columns = ['icu_current_covid', 'icu_current_covid_vented', 
               'hospitalizations', 'icu_crci_total', 'icu_crci_total_vented', 
               'icu_former_covid', 'icu_former_covid_vented']

regions = df["oh_region"].unique()

# Convert all numeric types to int64
for column in integer_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int64")

key_dates = {
    '2020-12-01': 'Emergence of Alpha variant',
    '2021-05-01': 'Emergence of Beta variant',
    '2021-08-01': 'Emergence of Gamma variant',
    '2021-11-01': 'Emergence of Delta variant',
    '2022-02-01': 'Emergence of Omicron variant',
    '2020-12-14': 'First Pfizer–BioNTech vaccine administered in Canada',
    '2020-12-23': 'Health Canada authorizes Moderna vaccine',
    '2021-02-26': 'Oxford–AstraZeneca vaccine authorized by Health Canada',
    '2021-03-05': 'Janssen (Johnson & Johnson) vaccine authorized by Health Canada',
    '2021-05-05': 'Pfizer vaccine authorized for adolescents 12 to 15 in Canada',
    '2021-09-16': 'Moderna and Pfizer vaccines receive full approval for use in individuals aged 12 and older',
    '2021-11-19': 'Health Canada approves Pfizer-BioNTech vaccine for children aged 5 to 11',
    '2021-12-14': 'One year since the first COVID-19 vaccine was administered in Canada',
    '2022-01-28': 'Truckers protest in Ottawa against mandatory vaccination',
    '2022-07-14': 'Health Canada approves Moderna vaccine for children aged 6 months to 5 years',
    '2022-09-01': 'Moderna bivalent vaccine targeting original and Omicron BA.1 variant authorized by Health Canada',
    '2022-10-07': 'Pfizer bivalent vaccine targeting Omicron BA.4 and BA.5 subvariants approved by Health Canada',
    '2023-09-12': 'Health Canada authorizes a new Moderna vaccine based on Omicron XBB.1.5 subvariant',
}

# For each field in the data, create a separate figure
for field in integer_columns:
    fig = go.Figure()

    # Group together all the rows with the same date (should be 6 per day)
    # Then sum them together.
    # reset index?
    total_data = df.groupby('date')[field].sum().reset_index()

    # Add trace for the total
    fig.add_trace(go.Scatter(x=total_data['date'],
                             y=total_data[field],
                             mode='lines',
                             name=f'Total {field.replace("_", " ").title()}',
                             line=dict(color='#3a86ff')))

    # Add a dot for each key date on the total line
    for key_date, description in key_dates.items():
        if pd.to_datetime(key_date) in total_data['date'].values:
            key_date_val = total_data[total_data['date'] == pd.to_datetime(key_date)][field].values[0]
            formatted_date = pd.to_datetime(key_date).strftime('%b %Y')  # E.g., 'Jan 2022'
            hover_text = f"{formatted_date}: {description}"
            fig.add_trace(go.Scatter(x=[key_date],
                                     y=[key_date_val],
                                     mode='markers',
                                     marker=dict(size=12, color='#780000'),
                                     hoverinfo='text',
                                     text=hover_text,
                                     showlegend=False))

    # Update layout for each field's figure
    fig.update_layout(title=f"Time Series of Total {field.replace('_', ' ').title()} in Ontario",
                      xaxis_title="Date",
                      yaxis_title=f"Total {field.replace('_', ' ').title()}",
                      hovermode="closest")

    # Show figure
    fig.show()

from statsmodels.tsa.seasonal import seasonal_decompose

total_ontario_hospitalizations = df.groupby('date')['hospitalizations'].sum()
total_data = total_ontario_hospitalizations.reset_index()

# Perform seasonal decomposition
decomposition = seasonal_decompose(total_ontario_hospitalizations, model='additive', period=150)

# Extract the trend component
trend = decomposition.trend.reset_index()

# Create the figure
fig = go.Figure()

# Add the trend data to the figure
fig.add_trace(go.Scatter(x=trend['date'], y=trend['trend'], mode='lines', name='Trend', line=dict(color='#780000')))

# Add line chart for current field
fig.add_trace(go.Scatter(x=total_data['date'],
                         y=total_data['hospitalizations'],
                         mode='lines',
                         name=f'Total {field.replace("_", " ").title()}',
                         line=dict(color='#3a86ff')))

# Update the layout
fig.update_layout(title='Trend of Hospitalizations in Ontario',
                  xaxis_title='Date',
                  yaxis_title='Number of Hospitalizations')

# Show the figure
fig.show()

