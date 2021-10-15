# Code Directory

## Mandatory Task
### Directory
- `MiniModels.ipynb`: contains all functions of the mini models 
  -   `Moving Average`
  -   `Region Maximisation`
  -   `Loss Removal`
  -   `Stationary Maximisation`
  -   `Action Shift`
- `Battery.py`: Contains the battery class that is needed for `PeriodMaximisation.py`
- `PeriodMaximisation.py`: Period Maximisation model (MAKE SURE `Battery.py` IS IN THE SAME FOLDER AS THERE IS DEPENDENCIES TOWARDS IT).
- `Final Optimisation Algorithm.ipynb`: Main Function to run the optimisation algorithm (JUST USE THIS ONE, IF THERE IS A CHANGE IN BATTERY SPECIFICATIONS CHANGE IT IN THE THREE FILES ABOVE).

### Instructions
1. Make sure `MiniModels.ipynb`, `Battery.py`, `PeriodMaximisation.py` and `Final Optimisation Algorithm.ipynb` are in the same directory.
2. Open the `Final Optimisation Algorithm.ipynb`.
3. Read the instructions in `Final Optimisation Algorithm.ipynb`.

## Bonus Task
### Directory
- `Run Bonus Task.ipynb`: notebook to run the mandatory task's algorithm onn predicted price obtained from `Vector Autoregression (Log).ipynb` 
- `Vector Autoregression (Log).ipynb`: contains test data price predictions with VAR using log 
- `ARMA.ipynb`: contains test data price predictions using ARIMA
- `MiniModels.ipynb`: contains all functions of the mini models to be run in the bonus task
- `OptimisationAlgs.ipynb`: contains all functions to run the main mandatory task algorithm (required for bonus task)

### Instructions
1. To predict prices using the final model, open the file `Vector Autoregression (Log).ipynb` and run the file.
2. To run the models for bonus task (VAR and ARMA Model) on mandatory task's algorithm, open `Run Bonus Task.ipynb`. This will run the predicted price on the mandatory task's model.

## Misc
- `Data Preprocessing.ipynb`: this file separates train and test period of `market_data.xlsx` and saves it to `preprocessed_data` folder.