<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# High-resolution networks for Semantic Segmentation

---

High-resolution networks show promising results on semantic segmentation. Here is a brief guide to dive into the details referring to the official code.

#### Network structure
* network config

```python
    config = CN()
    config.DATASET = CN()
    config.DATASET.NUM_CLASSES = n_classes

    config.MODEL = CN()
    config.MODEL.NAME = 'seg_hrnet'
    config.MODEL.PRETRAINED = ''
    config.MODEL.EXTRA = CN(new_allowed=True)

    // high_resoluton_net related params for segmentation
    HIGH_RESOLUTION_NET = CN()
    HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
    HIGH_RESOLUTION_NET.STEM_INPLANES = 64
    HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
    HIGH_RESOLUTION_NET.WITH_HEAD = True

    HIGH_RESOLUTION_NET.STAGE2 = CN()
    HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
    HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
    HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
    HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [48, 96]
    HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
    HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

    HIGH_RESOLUTION_NET.STAGE3 = CN()
    HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 4
    HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
    HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
    HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [48, 96, 192]
    HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
    HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'
```

* two basic convs with stride=2: conv1\conv2
* layer1: four Bottleneck defined in resnet
  * self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)
* transition1+stage2
  * transition1:use the output from the last stage to generate the input of the current stage, the main operation is use conv to make the number of channels to be the same.
  * stage2: a stage contains one or multiple MODULES(HighResolutionModule).
  * a HighResolutionModule contains multiple branches and one fuse_layer (down-or-up sample use interpolation and fuse)
  * a branch contains multiple blocks
* transition2+stage3