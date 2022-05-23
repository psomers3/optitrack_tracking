## Usage
The majority of this code needs to be run through blender and its built-in python environment. This can be done with the following command from the terminal:

```bash
blender --python my_script.py
```

where `my_script.py` is the script you want to run. For most cases in this repository, `render.py` is the script used to render/process the recorded data in blender.
If you don't wnat the blender gui to open up (i.e. when rendering is actually done) then you may pass the flag `-b` before `--python` to run blender headless.

In order to pass flags to your python script, follow the `my_script.py` entry with a singular `--` entry followed by all flags required by python. See the `render.py`
 script to see how to parse these. An example usage might be:
 ```bash
 blender -b --python render.py -- C:/render_path --render
 ```

# Installing packages in blender python
To get blender python working, it is easiest to first create an alias in your *.bashrc* file:

```bash
alias blender_python="path/to/blender 3.1/3.1/python/bin/python"
```
***NOTE: For installing packages in the blender python, you will need to open a terminal with admin rights when blender is installed in the default Program Files location!***

The blender python does not come with pip installed, but this can be added by using the following command in a terminal:
```bash
blender_python -m ensurepip --upgrade
```

It is best to install this repository as an external module so that the blender python can find it and change happen as expected. This is done by executing the following
command **from this repository directory:**
```bash
blender_python -m pip install -e .
```
