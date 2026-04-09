#!/usr/bin/python

import warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from pyecore.resources import ResourceSet, Resource, URI
from io import StringIO
import sys
# Step 3: Create a ResourceSet
rset = ResourceSet()

resource = rset.get_resource(URI('./metamodel/pacman.ecore'))
mm_root = resource.contents[0]
rset.metamodel_registry[mm_root.nsURI] = mm_root

model_block = ''
termination_marker = '###MODEL_END###'
start_marker = '###MODEL_START###'

print("Initialized!")

line = input()

while line != 'exit':
    if line == start_marker:
        model_block = ''
        line = sys.stdin.readline().strip()
        while line != termination_marker:
            model_block += line + '\n'
            line = sys.stdin.readline().strip()

   # exec(code_block)
    

    # Execute the function

    #result = fitness_from_qc(overall, settings, params=True)

    # Send the result back to Java

    with StringIO(model_block) as model_stream:
        resource.load(model_stream)
        
    print(resource.contents[0])
    sys.stdout.flush()  # Flush the output buffer

    # Read the next input from Java

    line = sys.stdin.readline().strip()



# Now, you can access the root of your model
#root_element = resource.contents[0]  # Replace with the actual structure of your model


# Do something with your model, e.g., print its contents
#print(root_element)