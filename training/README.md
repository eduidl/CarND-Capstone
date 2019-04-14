## Dataset

```sh
$ wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip
$ unzip *.zip -d data
```

## Finetuned Model

- ssd_mobilenet_v2_coco

```sh
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
$ tar xavf *.tar.gz
```

## Training

```sh
$ docker build -t <image_name> .
$ docker run --runtime=nvidia -it --rm -v $(pwd):/workspace/tl <image_name> bash
#---docker container---
$ cd /workspace/tl
# training
$ python ../models/research/object_detection/legacy/train.py --logtostderr \
      --train_dir=./model --pipeline_config_path=config/ssd_mobilenet_v2_coco.config
# export frozen graph
$ python ../models/research/object_detection/export_inference_graph.py \
      --input_type=image_tensor --pipeline_config_path=config/ssd_mobilenet_v2_coco.config \
      --trained_checkpoint_prefix=model/model.ckpt-20000 --output_directory=frozen_model
```
