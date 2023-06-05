# eng1n3
- - - 

### Description
Package that can create feature instances. At present there is one engine name `EnginePandas` which basically reads features from a file into a pandas dataframe and/or Numpy array. See the [notebooks](https://github.com/t0kk35/eng1n3/tree/main/notebooks) directory for examples. 

The output of the engine can subsequently be used as input to Neural Net Models created by the [m0d3ls](https://github.com/t0kk35/m0d3l) package.

### Example Usage
#### Pandas Engine
```
# Define Features
card = ft.FeatureSource('Card', ft.FEATURE_TYPE_STRING)
merchant = ft.FeatureSource('Merchant', ft.FEATURE_TYPE_STRING)

# Create TensorDefinition.
td = ft.TensorDefinition('Features', [card, merchant])

# Build a Pandas DataFrame from the TensorDefinition
with en.EnginePandas(num_threads=1) as e:
    df = e.df_from_csv(td, './file.csv', inference=False)
```

### Requirements
#### PandasEngine
- Pandas
- Numpy
- Numba