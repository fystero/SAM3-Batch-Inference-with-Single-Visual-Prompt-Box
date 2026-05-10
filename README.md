# SAM3视觉提示批量推理

**为SAM3模型新增了视觉提示的批量推理功能**

## 效益

现在你只需要准备好几张图，并**在这几张图上绘制几个提示框**，就能完成**整个数据集**的推理(初始化图片越多检测效果越佳)

特别适用于无法使用语言提示进行描述的数据集

## **原理**

每张图片推理完成时，将本次推理的结果转为输入，从geometry encoders中取出提示特征，将其保存，推理下张图片时，将其取出插入到geometry encoders中，做为视觉提示供后续decoders使用

## 使用流程

**1、部署好SAM3的运行环境**

**2、用labelme等软件标注提示框**

<img title="" src="file:./assets/1.png" alt="" width="380">

准备好1-5张图放至./datasets/init目录下，每张图标注1-5个矩形框并将标签设置为1，将标签文件保存在该目录下

****3、将待检测的文件放在./datasets/images目录下**

**4、运行vision_prompt_batch_infer.py文件***

在./datasets/outputs目录下查看推理结果*

![](./assets/2.png)
