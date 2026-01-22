# FAF
Feature-Adaptive Framework for Few-Shot Time Series Forecasting
## Requirements
* conda create -n FAF python=3.10
* pip install -r requirements.txt
## Original Dataset
https://www.iso-ne.com/isoexpress/web/reports/load-and-demand<br>
https://www.kaggle.com/datasets/varsharam/walmart-sales-dataset-of-45stores<br>
https://www.kaggle.com/datasets/saloni1712/co2-emissions<br>
https://www.kaggle.com/datasets/stealthtechnologies/gdp-growth-of-european-countries<br>
https://www.kaggle.com/datasets/vanvalkenberg/historicalweatherdataforindiancities<br>
## Experiments
### The First Step
`conda activate FAF`<br>
`cd FAF`<br>
### The Second Step
`python train.py --dataset ELECTRICI`<br>
`python train.py --dataset WALMART`<br>
`python train.py --dataset CO2`<br>
`python train.py --dataset GDP`<br>
`python train.py --dataset INDIAN`<br>

