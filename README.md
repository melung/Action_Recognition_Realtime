# ST-GCN기반 실시간 행동 인지 코드

## Prerequisites
- Python3 (over 3.9) cause of multiprocessing shared memory
- [PyTorch](http://pytorch.org/)
- Other Python libraries can be installed by `pip install -r requirements.txt`


### Installation
``` shell
cd torchlight; python setup.py install; cd ..

```

## 실시간 행동인지 Demo
```shell
#모든 Slave 컴퓨터에서 example_for_realtime_action_recognition.py코드 실행 후
python test_realtime.py recognition -c <config파일 경로> 

예시 :python test_realtime.py recognition -c config/st_gcn/ours/test.yaml
```


## 학습을 위한 Data Preparation
```
python tools/our_gendata.py --data_path <행동데이터 셋 경로> --out_folder <Preparation 데이터 저장경로>
```

## 학습 Training
```
python main.py recognition -c <config파일 경로> [--work_dir <work folder>]
예시 :python main.py recognition -c config/st_gcn/ours/train.yaml

```

## Citation
```
> **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

```