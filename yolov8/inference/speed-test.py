import numpy as np
import threading
import time
import torch
import torch.nn as nn
from pathlib import Path
import json
import subprocess
from collections import OrderedDict, namedtuple

class Detector(nn.Module):
    """
    Detector class.
    Loads a deep learning model and uses it to detect items in images.
    """
    def __init__(self,
                weights='yolov8n.pt',
                device=torch.device('cuda:0'),
                fp16=False):
        super().__init__()
        w = str(weights)
        jit, engine, onnx = self._model_type(w)
        model, metadata = None, None

        cuda = torch.cuda.is_available() and device.type != 'cpu'
        if cuda and not any([jit, engine]):
            device = torch.device('cpu')
            cuda = False

        if jit:
            model, metadata = self._load_jit(w, device, fp16, cuda)
        elif onnx:
            session, metadata, output_names = self._load_onnx(w, device, fp16, cuda)
        elif engine:
            model, metadata, output_names, context, bindings, binding_addrs, batch_size, fp16, dynamic = self._load_engine(w, device, fp16, cuda)
        else:
            raise NotImplementedError('Inference backend not found')

        if metadata:
            for k, v in metadata.items():
                if k in ('stride', 'batch'):
                    metadata[k] = int(v)
                elif k in ('imgsz', 'names', 'kpt_shape') and isinstance(v, str):
                    metadata[k] = eval(v)
            stride = metadata['stride']
            task = metadata['task']
            batch = metadata['batch']
            imgsz = metadata['imgsz']
            names = metadata['names']

        self.__dict__.update(locals())  # assign all variables to self

    @staticmethod
    def _load_jit(w, device, fp16, cuda):
        print(f'Loading {w} for TorchScript Runtime inference...')
        extra_files = {'config.txt': ''}  # model metadata
        model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
        model.half() if fp16 else model.float()
        if extra_files['config.txt']:  # load metadata dict
            metadata = json.loads(extra_files['config.txt'], object_hook=lambda x: dict(x.items()))
        return model, metadata

    @staticmethod
    def _load_onnx(w, device, fp16, cuda):

        print(f'Loading {w} for ONNX Runtime inference...')
        try:
            import onnxruntime
        except ImportError:
            if cuda:
                subprocess.check_output(f"pip install --no-cache onnx", shell=True).decode()
                subprocess.check_output(f"pip install --no-cache onnxruntime-gpu", shell=True).decode()
            else:
                subprocess.check_output(f"pip install --no-cache onnx", shell=True).decode()
                subprocess.check_output(f"pip install --no-cache onnxruntime", shell=True).decode()

        import onnxruntime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        metadata = session.get_modelmeta().custom_metadata_map  # metadata

        return session, metadata, output_names

    @staticmethod
    def _load_engine(w, device, f16, cuda):

        print(f'Loading {w} for TensorRT inference...')
        try:
            import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
        except ImportError:
            subprocess.check_output(f"pip install --no-cache nvidia-tensorrt -U --index-url https://pypi.ngc.nvidia.com ", shell=True).decode()
        import tensorrt as trt  # noqa

        # ensure version is 7.0.0 or later for TensorRT 7
        trt_version = trt.__version__
        trt_version = tuple(int(x) for x in trt_version.split('.')[:3])
        if trt_version < (7, 0, 0):
            raise Exception('TensorRT 7.0.0 or later required for newer .trt models')


        if device.type == 'cpu':
            device = torch.device('cuda:0')

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        # Read file
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
            metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
            model = runtime.deserialize_cuda_engine(f.read())  # read engine
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size

        return  model, metadata, output_names, context, bindings, binding_addrs, batch_size, fp16, dynamic

    def forward(self, im):

        """
        Runs inference on the YOLOv8 MultiBackend model.

        Args:
            im (torch.Tensor): The image tensor to perform inference on.
        """

        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.jit:  # TorchScript
            y = self.model(im)
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        else:
            raise NotImplementedError('Inference backend not found')

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    @staticmethod
    def _model_type(w):

        sf = ["torchscript", "engine", "onnx"]
        suffix = Path(w).suffix[1:].lower()
        types = [suffix == s for s in sf]
        return types

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='Testvitesse TRT')
    # parser.add_argument('path_top', type=str)
    # parser.add_argument('nb_workers', type=int)
    # args = parser.parse_args()

    detector = Detector(weights="/home/ec2-user/dev/yolo/runs/detect/train69/weights/last.torchscript",
                            device=torch.device('cuda:0'),
                            fp16=True)
    batchsize = 32
    nb_iter_per_worker = 250
    nb_workers = 2
    total_imgs = nb_workers*nb_iter_per_worker*batchsize

    test_img = np.array(np.random.uniform(0,255, [batchsize, 3, 416,416]), dtype=np.float32)
    # send it to torch
    test_img = torch.from_numpy(test_img).half().cuda()

    print("Model First batch...")
    for _ in range(2):
        out = detector(test_img)

    print(out.shape)
    def worker_thread():
        for i in range(nb_iter_per_worker):
            detector(test_img)
    workers = [threading.Thread(target = worker_thread) for _ in range(nb_workers)]

    print("Starting FPS test")
    start = time.time()
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    end = time.time()

    print("FPS test ended")
    print("BS :"+str(batchsize))
    print("nb_workers :"+str(nb_workers))
    print("Total duration :", end-start)
    print("NB images : ", total_imgs)
    print("FPS : ", total_imgs / (end-start))