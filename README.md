# OutSpot


## Efficient and Robust KPI Outlier Detection for Large-Scale Datacenters

OutSpot is an efficient and robust unsupervised outlier detection framework, which can detect both subsequence outlier and outlier time series for large-scale datacenters. OutSpot applies the hierarchical agglomerative clustering (HAC) method to cluster KPIs based on their patterns. For learning both the historical pattern of each KPI and the pattern of all KPIs in the same period, it  then encodes the clustering information into the generative model using the conditional variational autoencoder (CVAE) method. Additionally, OutSpot compares the reconstructed and original KPI shapes to determine whether a KPI is an outlier. 

## Getting Started

#### Clone the repo

```
git clone https://github.com/OutlierDetection-OutSpot/Code.git
```

#### Get data from github and unzip 

```
git lfs clone https://github.com/OutlierDetection-OutSpot/Dataset.git && cd Dataset && unzip data.zip  && cd  ../
```

#### Create a virtual environment for Python3.8.10

```
conda create -n py38 python=3.8.10
conda activate py38
```

#### Install dependencies

```
pip install -r requirements.txt
```

#### Run the code in the ./code directory

```
cd code
python run.py
python evaluate.py
```

If you want to change the default configuration, you can edit `DefaultConfig` in `code/config.py`


## Result

After running the programmings, you can get the output in the file directory that you set in the `code/config.py`. For each kpi of each machine, you can get their anomaly score.

* All of kpis's anomaly score are in `{config.result_dir}/score.xlsx`
* Trained model is in `code/model_weight/cvae/my_model_weight`
* Logs are in `{config.log_dir}`
* The threshold,  accuracy, and recall for achieving the best score will be output on the console  

