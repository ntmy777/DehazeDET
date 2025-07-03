# DehazeDET: Deep Image Dehazing for Improving Object Detection 

## Overview

DehazeDET is a novel dual-branch architecture that integrates the DSEU model (https://github.com/yeanwei97/DSEU) for image restoration and the Mix-YOLONet detection network for object detection into a joint parallel framework.

## Features

- **Training**: Train the model using `train.py`.
- **Testing**: Test the model's performance using `train.py --test`.
- **Inference**: Perform inference on a single image or an entire folder of images.
- **Data Conversion**: Convert various datasets (VOC, Foggy Driving, RTTS) to the required format.

## Dataset
- **VOC-FOG**: Download this model at https://drive.google.com/file/d/1bLUtwrKwzPwLI3yZBFZYw4BnINpxCfVp/view?usp=sharing (from https://github.com/yz-wang/TogetherNet) .
- **RTTS**: Download this model at https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2.
- **Foggy Driving**: Download this model at https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/. 

## Usage

### Training

To train the model, run:
```bash
python train.py
```

### Testing

To test the model, run:
```bash
python train.py --test --data your_dataset
```

### Inference

To perform inference on a single image, run:
```bash
python train.py --inference --image_path path_to_your_image
```

To perform inference on an entire folder of images, run:
```bash
python train.py --inference_test --model_path your_model --inference_input your_input_image
```

### Data Conversion

Convert VOCFOG dataset:
```bash
python voctoyolo.py 
```

Convert Foggy Driving dataset:
```bash
python foggytovoc.py 
```

Convert RTTS dataset:
```bash
python rttstovoc.py 
```

## Configuration

Modify the `args.yaml` file to change training parameters and other settings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Acknowledgements

- DSEU by yeanwei97(https://github.com/yeanwei97/DSEU)
- Mix-YOLONet