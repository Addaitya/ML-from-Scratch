# ML from scratch

1. In this repository I implemented popular machine learning algorithm from scratch.
2. I have used python and numpy for the implementation. But I used sklearn for loading dataset and other pre processing tasks.



## Libraries to install
First you need to setup enviornment for this. Use any one of following: 
<details>
    <summary>Conda enviornment</summary>
    1. Install mini-conda from [here](https://docs.anaconda.com/miniconda/)
    <br>
    2. Run in terminal:

    conda create --name ml_from_scratch 
    conda deactivate
    conda activate ml_from_scratch

</details>

<details>
    <summary>pip build-in enviornment</summary>
    1. Choose the folder where you download the repo.
    <br>
    2. Run following: 

    python3 -m venv ml_from_scratch
    source ml_from_scratch/bin/activate  # if didn't work remove source
</details>
<br>

Install following dependecies:

```
pip install numpy scikit-learn 
```

You may also need jupyter lab to open notebooks(.ipynb file)
```
pip install jupyterlab
jupyter lab
```
## ML algorithms in this repo

1. Decesion tree
2. Multiple linear regression