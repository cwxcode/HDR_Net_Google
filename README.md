# HDR_Net_Google


参考
=============================================
https://github.com/mgharbi/hdrnet


文件说明
=============================================
hdrnet目录下：
run.py  测试
train.py  训练，参数调节
models.py  包含3种模型可供选择
data_pipeline.py  包含3种训练数据导入方式可供选择

pretrained_models目录下：（如需要更多预训练模型，可联系作者）
预训练的模型

sample_data目录下：
存放数据集


安装依赖环境
=============================================
pip install numpy  （安装最新版本numpy）
pip install cython
pip install opencv-python
pip install imageio

cd hdrnet
pip install -r requirements.txt  （python 2.7, tensorflow-gpu 1.1.0）


编译
=============================================
cd hdrnet
make
在hdrnet目录下生成build文件夹（该代码已经编译好）


训练
=============================================
FiveK数据集，共5000张图片，其中4500训练，500测试。

训练数据要求tfrecords或jpg或png格式。

./hdrnet/bin/train.py <checkpoint_dir> <path/to_training_data/filelist.txt>

python hdrnet/bin/train.py pretrained_models/hdrp sample_data/identity/filelist.txt
训练的过程不会自己停下来，需要自己定义，loss小于多少就定下来，epoch达到多少就停下来。

tensorboard --logdir pretrained_models/hdrp
按照tensorboard提示的地址，就可以访问训练的中间过程了

#!/bin/bash

cm=1

CUDA_VISIBLE_DEVICES=$1 python hdrnet/bin/train.py \
        --learning_rate 1e-4 \
        --batch_size 4 \
        --data_pipeline HDRpDataPipeline \
        --model_name HDRNetPointwiseNNGuide \
        --nobatch_norm \
        --output_resolution 256 256 \
        --channel_multiplier $cm \
        --data_dir data/hdrp/filelist.txt \
        --eval_data_dir data/hdrp/filelist.txt \
        --checkpoint_dir output/checkpoints/hdrp_256_nn_cm$cm

CUDA_VISIBLE_DEVICES="1" python hdrnet/bin/train.py --batch_size 4 --data_pipeline HDRpDataPipeline --model_name HDRNetPointwiseNNGuide --nobatch_norm \
        --output_resol --output_resolution 256 256 --data_dir data/hdrp/filelist.txt --eval_data_dir data/hdrp/filelist.txt --checkpoint_dir output/checkpoints/hdrp


输入输出
=============================================
测试图片放在input目录下，得到的输出在output目录下。

支持的输入格式：png|jpeg|jpg|tif|tiff

输出图片保存：
scipy.misc.imsave()只能保存8bit
cv2.imwrite()保存有色偏
imageio.imwrite()只可以保存16bit的tif格式图片


问题
=============================================
The pre-trained HDR+ model expects as input a specially formatted 16-bit linear input.
加载使用hdrp模型存在问题，没有适用的16bit线性图片。

If you run our HDR+ model on an sRGB input, it may produce uncanny colors.

备用：使用faces模型


运行
=============================================
hdrnet/bin/run.py  修改路径sys.path.insert(0, "/home/chenwx/HDR_Net_Google-master")

命令：
CUDA_VISIBLE_DEVICES="1" python hdrnet/bin/run.py pretrained_models/faces sample_data/input sample_data/output
