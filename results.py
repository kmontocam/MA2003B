import pandas as pd
import numpy as np
import janitor
import unidecode
import pickle
import os
import operator
import functools
import itertools
import matplotlib.pyplot as plt
from typing import Callable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans

def shared_structure(db_name: str) -> list:
    """Returns sheets in dict of DataFrames that conain the same fields"""

    df_col_names = [[key, get_col_names(val)]
                        for key, val in databases[db_name].items()]

    grouped = list()
    for _, g in itertools.groupby(df_col_names, operator.itemgetter(1)):
        group = list(g)
        if len(group) > 1:  # Can't be performed with comprehension since g is a generator/iterator
            grouped.append(group)

    if len(grouped) == 1:
        return [sheet_name for sheet_name, _ in functools.reduce(operator.iconcat, grouped, [])]
        # Expecting there is only one shared_structure across the DataFrames
    else:
        return ...  # Not necessary

def f_identifier(dict_): return identifier.get(dict_, True)

def munge(df: pd.core.frame.DataFrame, fields: list):
    return (
        df
        .loc[:, fields]
        .pipe(janitor.rename_columns, new_column_names={'parametro': 'factor',
                                                        'Fecha': 'date'})
        .pipe(janitor.process_text, column_name='factor', string_function='strip')
        .pivot(index='date', columns='factor', values=ids)
        .convert_dtypes(convert_integer=False)
        .apply(lambda field: field.where(field > 0, pd.NA) if pd.api.types.is_numeric_dtype(field)
               else field.map(f_identifier))  # Remove negative values from numeric fields
                                              # and map str typed variables with a dict
        .pipe(janitor.clean_names, strip_underscores='r')
    )

def fa_linear_combinations(df: pd.core.frame.DataFrame, n_components=15):

    factor_analysis = FactorAnalysis(n_components=n_components)
    factor_analysis.fit_transform(df)
  
    factors = pd.DataFrame(factor_analysis.components_, columns=df.columns).T

    return factors

if 'databases.pkl' not in os.listdir('./raw-data/'):
    from quickstart.loader import XlsxDriveLoader

    Loads = XlsxDriveLoader()  # Drive Folder is hardcoded in module, since this is not prone to change
    databases = Loads.content  # Process takes approximately ~3 minutes to run for the first time.
                               # Then the file will be stored in the data/
else:
    with open('./raw-data/databases.pkl', 'rb') as file:
        databases = pickle.load(file)
    print("Loaded data from .pkl file")
    

(db_name_anotaciones, db_name_meteoorologicas,
 db_name_meteoorologicas, db_name_contaminantes) = databases.keys()


get_col_names: Callable[[pd.core.frame.DataFrame], pd.core.indexes.base.Index] = (
    lambda df: sorted(list(df.columns)))

raw_contaminantes = pd.concat([databases[db_name_contaminantes][sheet]
                               for sheet in shared_structure(db_name_contaminantes)])

raw_meteorologicas = pd.concat([databases[db_name_meteoorologicas][sheet]
                                for sheet in shared_structure(db_name_meteoorologicas)])

identifier = (
    databases[db_name_anotaciones]['LEEME']
    .loc[:, ['Flag', 'Hora']]
    .set_index('Flag')
    .dropna(axis='index')
    .squeeze()
    .apply(lambda string: unidecode.unidecode(string).strip().lower())
    .apply(lambda validity: True if validity == 'valida' else False)
    .to_dict()
)

identifier |= {'x': False}  # Record does not appear on DataFrame

# Dictionary for mesaurement units

measurement_units = (
    databases[db_name_anotaciones]['Hoja1']
    .iloc[23:41, [0, 2]]
    .dropna()
    .drop(33)
    .pivot(columns='Notas a considerar:', values='Unnamed: 2')
    .pipe(janitor.clean_names, remove_special=True)
    .mode()
    .squeeze()
    .to_dict()
)

id = 'SO2' # San Pedro identifier
ids = [f'{id}', f'{id} b'] 

fields = ['parametro', 'Fecha']
fields.extend(ids)

contaminantes = munge(raw_contaminantes, fields)
meteorologicas = munge(raw_meteorologicas, fields)

df_ = contaminantes.join(other=meteorologicas,
                         how='outer')

vals, flags = df_.columns.droplevel(1).unique()

sp = (
    df_
    .apply(lambda row: row[vals].where(row[flags].astype('bool')), axis=1) # check validity
    .replace({pd.NA: np.nan}) # .astype() does not operate when having diferent dtypes for
                              # missing values
    .astype(float)
    .drop_duplicates()
)

sp = sp.loc[(sp.index > '2019-06-01') &
             (sp.index < '2020-06-01'), :]

id = 'CE' # Centro identifier
ids = [f'{id}', f'{id} b'] 

fields = ['parametro', 'Fecha']
fields.extend(ids)

contaminantes = munge(raw_contaminantes, fields)
meteorologicas = munge(raw_meteorologicas, fields)

df_ = contaminantes.join(other=meteorologicas,
                         how='outer')

vals, flags = df_.columns.droplevel(1).unique()

centro = (
    df_
    .apply(lambda row: row[vals].where(row[flags].astype('bool')), axis=1) # check validity
    .replace({pd.NA: np.nan}) # .astype() does not operate when having diferent dtypes for
                              # missing values
    .astype(float)
    .drop_duplicates()
)

sp = (
      sp
      .pipe(janitor.transform_column, column_name='rainf',
            function=lambda rainf: rainf.where(
                  (rainf.index < '2019-8-25') | (rainf.index > '2019-9-6'), np.nan),
            elementwise=False)
      .pipe(janitor.transform_column, column_name='wsr',
            function=lambda wsr: wsr.where(wsr.index < '2021-3-23', np.nan),
            elementwise=False)
)

limits = pd.Series({
    'co': 26,
    'no': 210,
    'no2': 210,
    'nox': 210,
    'o3': 90,
    'pm10': 70,
    'pm2_5': 41,
    'so2': 75,
    'prs': 760,
    'rainf': 130,
    'rh': 100,
    'sr': 1,
    'tout': 45,
    'wdr': 360,
    'wsr': 117,
}) # Maximum value that variables can take according to the Mexican normatives of air quality

sp = sp.apply(lambda row: row.where(row < limits, np.nan), axis=1)
centro = centro.apply(lambda row: row.where(row < limits, np.nan), axis=1)

# Imputate missing values from centro

sp = (
    sp
    .pipe(janitor.fill_empty, column_names='no', value=centro.no)
    .pipe(janitor.fill_empty, column_names='no2', value=centro.no2)
    .pipe(janitor.fill_empty, column_names='nox', value=centro.nox)
    .pipe(janitor.fill_empty, column_names='rainf', value=0) # There was no rain
)

# Rolling statistic and first days with mean

df = (
    sp
    .fillna(sp.rolling(24*7*12, min_periods=1, closed='both').mean())
    .fillna(sp.mean())
)

# Results

scaler = StandardScaler()

scaled_df = pd.DataFrame(scaler.fit_transform(df),
                         columns=df.columns,
                         index=df.index)

factors = fa_linear_combinations(scaled_df)

factor_analysis = FactorAnalysis(n_components=15)
lc = pd.DataFrame(factor_analysis.fit_transform(scaled_df),
                  index=scaled_df.index)


# 1st result
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(x=lc[0], y=lc[1], c='#ADD8E6')

for k, v in lc.loc[lc.iloc[:, 0] > 3.6, 0:1].iterrows():
    ax.annotate(k.strftime('%b-%d-%H:00'), v+0.05)

fig.suptitle('Registros con altas concentraciones de nitrógenos', fontsize=18)
fig.tight_layout()
_ = fig.text(0, 0, s='Varianza explicada por los primeros 2 factores: $69.5\%$')

# 2nd result
fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(lc.index, lc.iloc[:, 0], c='#555555')
_ = fig.suptitle('Primer factor (contentración de nitrógenos)', fontsize=20)

# 3rd result

daily_scaled_df = scaled_df.groupby(pd.Grouper(freq='d')).mean()
factors = fa_linear_combinations(daily_scaled_df)

lc = pd.DataFrame(FactorAnalysis(n_components=15).fit_transform(daily_scaled_df),
                  index=daily_scaled_df.index).loc[:, [0, 1, 2]]

# Clustering

kmeans = KMeans(n_clusters=2).fit(lc)
labels = kmeans.predict(lc)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

is_cold = np.array(labels == 0)
is_hot = np.array(labels == 1)

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
           c='black', s=50, label="Centro", alpha=1)

ax.scatter(lc.loc[is_cold, 0], lc.loc[is_cold, 1], lc.loc[is_cold, 2],
           c='#7EBACE', s=50, label='Frío', marker='2')
ax.scatter(lc.loc[is_hot, 0], lc.loc[is_hot, 1], lc.loc[is_hot, 2],
           c='#Ef983C', s=50, label='Caliente', marker='*')

ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.legend()

fig.suptitle(
    'Factores agrupados según su promedio\nen indicadores durante el día', fontsize=20)
_ = fig.text(
    0.5, 0.1, s='Varianza explicada por los primeros 3 factores: $90.4\%$')

# 4th result

monthly_scaled_df = scaled_df.groupby(pd.Grouper(freq='M')).mean()
factors = fa_linear_combinations(monthly_scaled_df)

factor_analysis = FactorAnalysis(n_components=15)
lc = pd.DataFrame(factor_analysis.fit_transform(monthly_scaled_df),
                  index=monthly_scaled_df.index).loc[:, [0, 1]]

# Clustering

kmeans = KMeans(n_clusters=3).fit(lc)
labels = kmeans.predict(lc)
centroids = kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(centroids[:, 0], centroids[:, 1],
           c="black", s=20, label="Centro", alpha=1)

for k, v in lc.iterrows():
    ax.annotate(k.strftime('%b'), v+0.05)

ax.scatter(x=lc[0], y=lc[1], c=labels, cmap='plasma')

fig.suptitle('Factores agrupados según su promedio\nen indicadores durante el mes', fontsize=18)
#fig.tight_layout()
fig.subplots_adjust(top=0.85, bottom=0.1)
fig.legend()
_ = fig.text(0, 0, s='Varianza explicada por los primeros 2 factores: $81.9\%$')

