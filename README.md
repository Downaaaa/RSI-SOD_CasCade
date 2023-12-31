# RSI-SOD_CasCade
其中Unet-model就是类unet的结构设计。CasCadeUR.py是整个模型定义的文件  
当前比较好的指标以及自己代码指标的对比  
|             |   Sα↑  | max F_β↑| mean F_β↑| adp F_β↑| max E_ξ↑| mean E_ξ ↑| adp E_ξ ↑|  mae↓  |
| ---------   | ------ | ------  | ------   | ------  | -----   | -----     |-----     |-----   |
| GeleNet-pvt | 0.9376 | 0.8923  | 0.8781   | 0.8641  | 0.9828  | 0.9765    | 0.9762   | 0.0064 |
| 第五版本     | 0.9329 | 0.8811  | 0.8682   | 0.8443  | 0.9743  | 0.9691    | 0.9384   | 0.0063 |
| 23-11-1     | 0.9164 | 0.8567  | 0.8411   | 0.8173  | 0.9639  | 0.9592    | 0.9329   | 0.0086 |
| 23-11-8     |        |         |          |         |         |           |          |        |
|             |        |         |          |         |         |           |          |        |
|             |        |         |          |         |         |           |          |        |
 
GeleNet-pvt：G. Li, Z. Bai, Z. Liu, X. Zhang and H. Ling, "Salient Object Detection in Optical Remote Sensing Images Driven by Transformer," in IEEE Transactions on Image Processing, vol. 32, pp. 5257-5269, 2023, doi: 10.1109/TIP.2023.3314285.   (2023 TIP)  
[github地址](https://github.com/MathLee/GeleNet/tree/main)  

评价指标所用代码包如下所示，利用该包还可以实现一些曲线图的对比，相对比较的方便。具体的使用可以参考下面的地址：    
[评价指标](https://github.com/lartpang/PySODMetrics)  
  
## 更新日志   
##### 2023-11-1 update  
主要更新：  
1、将第一层级联结果进行了注意力的更换，新加入的Attention.py文件就是新的一种注意力  
2、将第一层的Decoder做了更改，目的就是想要提供更多的信息， DecoderUnit.py文件就是其主要的改动  
3、将第二层的级联网络改为了第五版代码的精简版（没有加边缘、骨干信息两个分支）也既是BBRF.py文件的代码  
结果如上表‘23-11-1’所示（没有训练完epoch-42的结果），好像相比上周没有什么变化
  
##### 2023-11-3 update
主要更新：  
1、将第一层的backbone换成了ResNet18  
2、将损失做了更改，将第一层的结果改为交叉熵损失。  

##### 2023-11-8 update
主要更新：  
1、将第一层级联结构换成GeleNet-PVT。同时将第二层的级联结构变的简单化。  
2、将optimizer分成两个部分进行不同学习率的学习，同时第一层网络引入了预训练权重。结果表现非常的好    
3、还有的更改就是将第二层级联结果的backbone单独改为resnet18，同时decoder设计为类似于FPN的结构。  


# 2023-12-13 update  
主要更新：
重新上传了文件夹CasCadeNew，主要就是[参考论文](https://github.com/ZhenglinZhou/STAR/blob/master/lib/backbone/stackedHGNetV1.py)的网络设计级联网络
