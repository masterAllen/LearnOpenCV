1. 下载数据集
Gehler-Shi dataset(http://www.cs.sfu.ca/~colour/data/shi_gehler)，它是由两台单反相机拍摄的共 568 张图片。

png_all: 网站给的数据集四个压缩包中所有的 PNG 图片。
groundtruth_568：网站中给的 groundtruth 压缩包解压的文件。

2. 训练
learn_color_balance.py: https://github.com/opencv/opencv_contrib/tree/master/modules/xphoto/samples/learn_color_balance.py

python .\learn_color_balance.py -i .\png_all -g .\groundtruth_568\real_illum_568..mat -r 0,378 --num_trees 30 --max_tree_depth 6 --num_augmented 0

3. 测试
color_balance_benchmark.py: https://github.com/opencv/opencv_contrib/tree/master/modules/xphoto/samples/color_balance_benchmark.py
color_balance_benchmark.py 有一定修改

python color_balance_benchmark.py -a GT,grayworld,learning_based:color_balance_model.yml -m . -i .\png_all -g .\groundtruth_568\real_illum_568..mat -r 400,450 -d "img"

img: 示例结果