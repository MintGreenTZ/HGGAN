# HGGAN
Hand grasping generation based on conditional GAN.

## Environment

My install commands are

```shell
conda create -n hggan python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install opencv-python
pip install git+https://github.com/hassony2/manopth.git
pip install git+https://github.com/hassony2/libyana@v0.2.0
conda install tqdm
pip install chumpy
pip install open3d-python
conda install matplotlib
```

In the commands above you can change the version of cuda to satisfy your local environment.

You can also set up the environment by conda:

```shell
conda env create -f environment.yaml -n hggan
```

and activate by

```shell
conda activate hggan
```

