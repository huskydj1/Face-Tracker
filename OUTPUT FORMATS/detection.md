# Detection Format (using Biubug's Pytorch Implementation of RetinaFace):

## Input: 
Google Colab: Face_Tracker:Colab => Pytorch_RetinaFace => data => FaceTrackerImages => val

wider_val.txt:
/<folder name inside of "images">/<file name>.jpg
e.g. /8--voccamp/0.jpg

=> images:
=> /<video index>--<video name>
        <frame #>.jpg (stores one frame of input video)
        e.g. 1518.jpg

## Output:
Google Colab: Face_Tracker:Colab => Pytorch_RetinaFace => widerface_evaluate => widerface_txt

=> /<video index>--<video name>
        <frame #>.txt (stores information from one input frame):

        line 0: Frame #
        line 1: # of Detected Faces
        (for each face):
        xmin ymin xlen ylen score
        x y x5(per landmark)
