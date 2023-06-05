# Report on Code

- Link to Repository - https://github.com/bergstartup/wcs_project

## Steps to Run

The code for our research question is in the `modules` folder and the code for the packages to be run on the Brane Platform are in the `project` folder.

The data should be extracted and placed in the `data` folder. It is not unzipped now as we cannot push more than 100MB files in the git push command.

The outputs should ideally have been generated in the `outputs` folder.

## Experience with Brane

We did not have a good experience with Brane, as we were not able to successfully set up our pipeline as hoped. We encountered many issues from the beginning:

- Setting up the worker node in a different location. We had trouble in setting up the worker node in a different VM and running our example application on the different node. We kept encountering different errors.
- Our major roadblock was with setting up the pipeline for our data processing code. We kept encountering many different kinds of errors, which we were not able to figure out.
- When we called the preprocessing function(module) in our code, it always started terminating with the error `Internal package call was forcefully stopped with signal 9`.
- There is no syntax error in the code, but it terminates without giving any more information.
- We debugged our code by commenting out each line, and it stopped working at line 27 of our code in `preprocess.py`. We were not able to figure out anything after this, as there was no apaprent reason on why it was failing and we were stuck here.
- Other files were also stuck as we were not able to proceed with our pipeline without the preprocessing package.
- We also believe that our complication arose due to the handling of large number of input files. It could have been easier if we had to process only one dataset and pass it as an intermediate file along the pipeline.
