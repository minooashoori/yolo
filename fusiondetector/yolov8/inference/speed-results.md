
```bash
root@e5682eae0e84:/workspace# python3 speed-test.py 
Loading /models/yolov8s_half.torchscript for TorchScript Runtime inference...
Starting FPS test
FPS test ended
BS :32
nb_workers :2
Total duration : 26.101537704467773
NB images :  16000
FPS :  612.9907050365579
```
```bash
root@e5682eae0e84:/workspace# python3 speed-test.py 
Loading /models/yolov8s_half.torchscript for TorchScript Runtime inference...
Starting FPS test
FPS test ended
BS :16
nb_workers :2
Total duration : 14.047119140625
NB images :  8000
FPS :  569.5117924118393

```