#安装依赖

git clone https://github.com/fangxu622/face_seeta

cd face_seeta 

git clone https://github.com/fangxu622/SeetaFaceEngine (seetafaceEngine 在face_seeta目录下)

> on linux

```bash
cd SeetaFaceEngine/
mkdir Release; cd Release
cmake ..
make
```

# installation

```bash
cd ./face_seeta
python setup.py install
```

使用

python face_dect.py -vp /path/xx.mp4

可选参数

-sf 2 , 跳跃检测，每隔2帧 检测一次，越大卡顿效果越明显

-s 0.5 取值范围 0～1,值越大，精度越高，速度越慢，值越小，精度越低，速度越快

-m 模型路径，默认值：./faceFront.bin 与执行文件同目录
