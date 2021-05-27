# imbalanced_classification

## Setup
You can setup the enviroment using commands bellow
```
$ docker build -t [image_name] -f docker/Dockerfile .
$ docker run -d -p [port_num]:[port_num] --rm -itd -e TZ=Asia/Tokyo --gpus all --shm-size=32gb -v /home/user_name/:/home/ --name [container_name] -u root [image_name]
$ docker exec -it [container_name] bash
```

## Training and Test
You can Training and Test using commands bellow
```
$ python train.py --cfg_file [cfg_file] --seed [seed] --run_name [run_name]

# [cfg_file] = ./config/convae.yaml or ./config/cnn_classifier.yaml or ./config/conv_classifier.yaml
# [seed] = int
# [run_name] = str
```
## Visualize Results
You can visualize results using tensorboard
- learning curve
- confusion matrix
- input, decoded images
```
$ tensorboard --host 0.0.0.0 --logdir [path_to_log] --port [port_num]
```
![image](https://user-images.githubusercontent.com/40286449/119701383-3b559f80-be8f-11eb-96c6-cd8ab4880720.png)
