# MosquitoCNN

Open a terminal client.

Install anaconda

https://www.anaconda.com/download/#linux

Open your bashrc

``` console
$ sudo gedit ./bashrc
```

Add the line if doesn't exist

export PATH=/home/eduardo/anaconda3/bin:$PATH

Check anaconda version

``` console
$ conda -V
```

In the terminal client enter
``` console
$ conda update conda
```

Update any packages if necessary by typing y to proceed.

In the terminal client enter the following where your envname is the name you want to call your environment, and replace x.x with the Python version you wish to use. (To see a list of available python versions first, type conda search "^python$" and press enter.)

``` console
$ conda create -n PI2 python=3.5 anaconda
```

Activate virtual environment

``` console
$ source activate PI2
```

Install dependences

``` console
$ conda install -c anaconda scipy 
$ conda install -c conda-forge keras tensorflow
```