# 肺炎检测

## 工程使用方法
1.打开终端，克隆该仓库到本地当前目录

```
git clone https://github.com/Jackson-coder/CheXNet.git
```
2.将数据集放入dataset文件夹中

3.进入工程目录
```
cd CheXNet
```
4.进入本机工作虚拟环境
```
conda activate <环境名>
```
5.训练模型
```
python train.py
```
6.用训练得到的模型进行预测
```
python predict.py
```
