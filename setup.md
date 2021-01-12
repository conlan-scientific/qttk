
#### Setting up Anaconda Environment

```
conda create --name qttk python=3.7
conda activate qttk
conda install numpy pandas matplotlib scipy scikit-learn
conda install ipython

# In case you need to patch ipython
pip install -U jedi==0.17.2 parso==0.7.1
```

#### Common git commands

```

git clone https://github.com/conlan-scientific/qttk

# Pull updates 
git pull

git add -A .
git commit -m "your message"
git push origin master
```


