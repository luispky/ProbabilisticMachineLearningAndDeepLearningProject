# I am in the scripts folder of the project directory
# I want to import the save_plot function form the
# src folder of the project directory in the test_folders_creation.py file
# Let's write code to do that

import sys
import os

# Add the path to the src folder to the system path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import the save_plot function from the src folder
from test_folders_creation import save_plot

# Call the save_plot function
save_plot()

