# tferrorgeab
tensorflow error unable to get element as bytes

I think this works with python 2 and 3
and with any recent version of tensorflow,
but returns:

tensorflow.python.framework.errors_impl.InternalError: Unable to get element as bytes.

Will update with fix when I get it.


Run with:

python mlturn1a1ssa.py


The directory in this repo is the output of a run, it can be deleted, or not, for re-run.


UPDATE: hardcoding the data in this line:

  g.train_data = dfin.values

to this instead makes it run:

  g.train_data = np.array([[0]*4])
  
 so it is something with the dataframe .values transformation, 
 even though the data looks ok when printed.
