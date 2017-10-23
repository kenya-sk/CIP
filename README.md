# CIP

## workflow
### install openCV
```
pyenv install anaconda3-4.2.0  #should not be python3.6
pyenv local anaconda3-4.2.0
conda install -c https://conda.anaconda.org/menpo opencv3
```
### configure
Enter absolute path for data directory in `config.txt`
e.g)/hoge/pre_image

### exec
1. ciputil.py
    * load configuration deta
2. tr_image_movie.py
    * converts a series of `.tif`s into `.mp4` in a designated order.
3. stabilize.py
    * correcting image using a  sparse optical flow
4. cumulative_flow.py
    * cumulative dense optical flow 
5. dot_product.py
    * caluculate dot product of the cululative flow
6. output.py
    * output answer file(.csv). using DBSCAN clustering 

### procedure
```
1. write configuration to config/config.ini
2. execute stabilize.py. save numpy file about fixDirection_arr  file in the set config.ini file.
3. execute cumulative_flow.py. save numpy file about cmlFlow_arr  in the set config.ini file.
4. execute dot_product.py. save numpy file about dotProduct_arr in the set config.ini file.
5. execute output.py. export output.csv in current directory. 
```

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

