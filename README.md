# CIP

## workflow
### installation of openCV
```
pyenv install anaconda3-4.2.0  #should not be python3.6
pyenv local anaconda3-4.2.0
conda install -c https://conda.anaconda.org/menpo opencv3
```

### configuration
Enter the absolute path for the data directory in `config/config.ini`
e.g)/hoge/pre_image

### procedure
```
1. Set the names of the input images as follows. Pre_Data{level (2 digits)}_t{time (3 digits)}_page_{page (4 digits)}.tif
2. Write the directory name of the input images in config/config.ini
3. ./stabilize.py
4. ./cumulative_flow.py
5. ./dot_product.py
6. ./detect.py
```

### detail
1. ciputil.py
    * loads configuration data
2. tr_image_movie.py
    * converts a series of `.tif`s into `.mp4` in designated order (time-continuous or depth-continuous)
3. stabilize.py
    * calculates movement of images to stabilize them by sparse optical flow
4. cumulative_flow.py
    * cumulates dense optical flow
5. dot_product.py
    * caluculates dot product of the cumulative flow
6. detect.py
    * detect division event to use DBSCAN clustering algorithm, and outputs answer file(.csv)

### helper
tr_image_movie.py can convert series of tifs into mp4. When given the output .csv of output.py, the detected cell division event will be surrounded by a red frame. put output.csv in directory of image data.
./tr_image_movie.py arg1(filepath to output video) arg2(input image level) arg3(time or page)

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
