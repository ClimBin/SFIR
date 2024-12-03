# SFIR
The official implementation code for the paper "SFIR: Optimizing Spatial and Frequency Domains for Image Restoration"

## 1. Training the Model

To train the model, you can use the following command:

```bash
python main.py --mode train --data_dir path/xxx
```

Replace `path/xxx` with the path to your training dataset directory. This command will start the model training process and train the model using the provided dataset.

## 2. Testing the Model

To test a pre-trained model, use the following command:

```bash
python main.py --mode test --data_dir path/xxx --test_model path_to_ckpt
```

- `path/xxx`: The directory path to your test dataset.
- `path_to_ckpt`: The path to the checkpoint file of the pre-trained model.

This command will load the specified model and evaluate its performance on the test dataset.

## 3. Results

We have provided the pre-trained models and results for the various restoration tasks discussed in the paper. You can access them through the following link: [xxx link](#).

Feel free to download the pre-trained models and directly test them, or refer to the results for further insights.

## 4. Acknowledgments

Our code is based on the [IRNeXt]([IRNeXt](https://github.com/c-yn/IRNeXt)) architecture. We want to express our sincere thanks to the authors for their outstanding work.
