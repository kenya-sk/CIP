# CIP

## Workflow
### Installation of openCV
```
pyenv install anaconda3-4.2.0  #should not be python3.6
pyenv local anaconda3-4.2.0
conda install -c https://conda.anaconda.org/menpo opencv3
```

### Procedure
1. Set the names of the input images as follows. Pre_Data{level (2 digits)}_t{time (3 digits)}_page_{page (4 digits)}.tif
1. Set the absolute path for the image directory in `config/config.ini`
1. `./stabilize.py`
1. `./cumulative_flow.py`
1. `./dot_product.py`
1. `./detect.py`

### Detail
1. stabilize.py
    * calculates movement of images to stabilize them by sparse optical flow
1. cumulative_flow.py
    * cumulates dense optical flow
1. dot_product.py
    * caluculates dot product of the cumulative flow
1. detect.py
    * detect division events using DBSCAN clustering algorithm, and outputs answer file(.csv)

### Helper
* `tr_image_movie.py` converts series of `tif`s into a `mp4`. When given the answer file(.csv), it will output a movie with red frames surrounding the detected cell division events.
```
(cp output.csv ${BASEDIR}/Pre_DATA??/Pre_Data??_Answer.csv)
./tr_image_movie.py arg1(filepath to output video) arg2(input image level) arg3(time or page)
```

---

## development policy
* shared repository model
  * <https://help.github.com/articles/about-collaborative-development-models/>
  * <http://blog.qnyp.com/2013/05/28/pull-request-for-github-beginners/>
* delete branches that are associated with merged pull requests

## info
<http://alcon.itlab.org/>

2017/08/04: entry<br>
2017/09/29: submission<br>
2017/10/20: submission(paper)<br>
2017/12/07: party<br>
