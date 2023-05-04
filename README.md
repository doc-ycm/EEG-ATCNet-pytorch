# EEG-ATCNet-pytorch

这是我复现的 [BraVL](https://github.com/ChangdeDu/BraVL) 的用于运动想象分类的 PyTorch 代码。原项目使用 TensorFlow 实现，我仔细完善地复刻了原模型。
但是，加入多头注意力层后结果就，pytorch的模型性能较原模型性能差很多，目前考虑是mha实现机制的问题。