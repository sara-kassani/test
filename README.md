Traceback (most recent call last):
  File "train.py", line 128, in <module>
    main()
  File "train.py", line 123, in main
    num_classes=num_classes
  File "/home/paperspace/0.HC_Russion_DataAug/utils.py", line 68, in train
    outputs = model(inputs)
  File "/home/paperspace/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 491, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/paperspace/anaconda3/lib/python3.6/site-packages/torch/nn/parallel/data_parallel.py", line 112, in forward
    return self.module(*inputs[0], **kwargs[0])
  File "/home/paperspace/anaconda3/lib/python3.6/site-packages/torch/nn/modules/module.py", line 491, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/paperspace/0.HC_Russion_DataAug/models.py", line 388, in forward
    x_out = up(torch.cat([x_out, x_skip], 1))
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 1. Got 67 and 64 in dimension 2 at /opt/conda/conda-bld/pytorch_1525909934016/work/aten/src/THC/generic/THCTensorMath.cu:111
