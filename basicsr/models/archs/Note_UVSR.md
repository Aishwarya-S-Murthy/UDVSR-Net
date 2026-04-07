basicsr\\models\\archs\\\_\_init\_\_.py

uses scandir to find any file ending in \_arch.py. Since your new name UVSR\_temporalSimple\_arch.py still ends in \_arch.py, this file will automatically import it without you typing a single line.

So make sure there is only one file that ends with \_arch.py in this folder.





**Simplev3 and SingleScaleHybridFreqTile Difference**

1\. Skip Path Implementation

File 1 (Simple V3): Uses a Feature-level Skip Path. It takes the output of the initial feature extraction (conv\_in), upsamples it using the learned UpsampleBlock, and adds it to the main path's upsampled features.



File 2 (Single Scale Hybrid): Uses an Image-level Residual Skip Path. It takes the raw 3-channel input frames and performs a standard Bicubic Upsampling. This upsampled image is added to the network's output at the very end as a global residual.



2\. Refinement Strategy

File 1 (Simple V3): Features two distinct refinement stages. RefinementBlock1 occurs before upsampling, and RefinementBlock2 occurs after the skip path and main path have been combined in the high-resolution space.



File 2 (Single Scale Hybrid): Features only one RefinementBlock which occurs immediately after the temporal/spatial fusion but before upsampling.



3\. Final Output Projection

File 1 (Simple V3): The final conv\_out (3x3 convolution) is applied to the output of the second refinement block to produce the final image.



File 2 (Single Scale Hybrid): The final conv\_out is used to project the deep features (64 channels) down to image channels (3 channels) before adding the bicubic upsampled skip path.

