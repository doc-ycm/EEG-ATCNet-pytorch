# EEG-ATCNet-pytorch

这是我复现的 [EEG-ATCNet](https://ieeexplore.ieee.org/document/9852687) 的用于运动想象分类的 PyTorch 代码。原项目使用 TensorFlow 实现，我仔细完善地复刻了原模型。

但是，加入多头注意力层后，pytorch的模型性能较原模型性能差很多，目前考虑是mha实现机制的问题。

将本代码放置在[原项目](https://github.com/ChangdeDu/BraVL) 根目录下就可运行。
