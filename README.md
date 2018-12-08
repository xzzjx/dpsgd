# deep learning with differential privacy using pytorch

## Idea

1. 动态调节sigma和C
2. 使用stochastic weight average
3. 使用self-ensemble

## Todo

- [x] 完成fully connected network mnist
- [ ] 完成autoencoder mnist，对mnist做PCA降维处理。保存模型，加载自己数据
- [ ] 完成per example gradient
- [x] 查看不同层的梯度，以及总梯度的l2norm, l2norm的方差变化的情况

## Tricks

1. 使用[torch.utils.data.WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)来实现resampling