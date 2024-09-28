# COVID 19 Detection via CNN on Chest X-Rays
![home](https://raw.githubusercontent.com/kevin-wijaya/resources/main/images/covid-19-detection-via-cnn-on-chest-x-rays/home.png)

## Table of Contents
+ [About](#about)
+ [Tech Stack](#techstack)
+ [Getting Started](#getting_started)
+ [Usage](#usage)
+ [Reports](#reports)
+ [Screenshots](#screenshots)
+ [Author](#author)

## About <a name = "about"></a>
C19DCCXR: COVID-19 Detection via CNN on Chest X-Rays is a project where I utilize Convolutional Neural Networks (CNN) to detect COVID-19 infections using chest X-ray images. The dataset for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset), which includes X-ray images of patients with COVID-19, Viral Pneumonia, and Normal. CNNs are employed to automatically extract key features from the X-ray images, allowing the model to identify patterns associated with the infection.

This is my first project in image classification, where I also developed an end-to-end web-based version of the system. The website is accessible via a local network and features a minimalist, user-friendly interface that allows users to upload X-ray images and receive diagnostic results efficiently.

## Tech Stack <a name = "techstack"></a>

- Web Application: JQuery, Tailwind CSS, Axios, Webpack, Flask
- Experiment: Numpy, Pandas, Scikit-learn, PIL, OpenCV, Torch, Torchvision

## Getting Started <a name = "getting_started"></a>

These instructions will guide you through installing the project on your local machine for testing purposes. There are two methods of installation, with Docker or manually using Linux or MacOS commands.

### Requirements

This project requires Python 3.10.


### Installation (Docker)

Clone this repository
``` sh
git clone https://github.com/kevin-wijaya/COVID-19-Detection-via-CNN-on-Chest-X-Rays.git
```

Change the directory to the `cloned repository`
``` sh
cd COVID-19-Detection-via-CNN-on-Chest-X-Rays/
```

Run docker compose
``` sh
docker compose up --build
```

Open your web browser and go to the following URL
``` python
# http://localhost:8001
```

### Installation (Linux or MacOS)

Clone this repository
``` sh
git clone https://github.com/kevin-wijaya/COVID-19-Detection-via-CNN-on-Chest-X-Rays.git
```

Change the directory to the cloned repository and then navigate to the `server` directory
``` sh
cd COVID-19-Detection-via-CNN-on-Chest-X-Rays/server/
```

Initialize the python environment to ensure isolation
``` sh
python -m venv .venv
```

Activate the python environment
``` sh
source .venv/bin/activate
```

Install prerequisite python packages
``` sh
 pip install --no-cache-dir -r requirements.txt
```

Run the Flask server
``` sh
env FLASK_APP=./src/app.py:serve flask run --debug --port=8000 --host=0.0.0.0
```

Open new terminal and change the directory to the cloned repository and then navigate to the `client` directory
``` sh
# replace the /path/to/your/ with the path where your cloned repository is located
cd /path/to/your/COVID-19-Detection-via-CNN-on-Chest-X-Rays/client/
```

Change the directory to the production folder
``` sh
cd .dist/
```

Run the Python HTTP server
``` sh
python -m http.server 8001 --bind 0.0.0.0
```

Open your web browser and go to the following URL
``` python
# http://localhost:8001
```

## Usage <a name = "usage"></a>

To use this web application is easy, follow these 3 steps:

1. **Upload Image**: Upload your image by either dragging and dropping it into the designated area or by browsing your files.
2. **Classify**: Click the "Classify" button, and the system will provide the diagnostic result based on the uploaded image.

## Reports <a name = "reports"></a>

Below are a graphic  and table presenting the evaluation metrics from the experiments conducted:

### Graph of Training Accuracy and Loss Over Epochs

![training-validation](https://raw.githubusercontent.com/kevin-wijaya/resources/main/images/covid-19-detection-via-cnn-on-chest-x-rays/training-validation.png)


### Table of Training Loss and Accuracy Over Epochs
<table>
    <tr>
        <th>Epoch</th>
        <th>Train Loss</th>
        <th>Train Accuracy (%)</th>
        <th>Val Loss</th>
        <th>Val Accuracy (%)</th>
    </tr>
    <tr>
            <td>1</td>
            <td>3.791473</td>
            <td>44.22</td>
            <td>1.193621</td>
            <td>56.06</td>
        </tr>
        <tr>
            <td>2</td>
            <td>0.500658</td>
            <td>80.08</td>
            <td>0.338603</td>
            <td>78.79</td>
        </tr>
        <tr>
            <td>3</td>
            <td>0.177622</td>
            <td>91.24</td>
            <td>0.453176</td>
            <td>89.39</td>
        </tr>
        <tr>
            <td>4</td>
            <td>0.133550</td>
            <td>95.62</td>
            <td>0.299129</td>
            <td>87.88</td>
        </tr>
        <tr>
            <td>5</td>
            <td>0.063360</td>
            <td>97.61</td>
            <td>0.308526</td>
            <td>84.85</td>
        </tr>
        <tr>
            <td>6</td>
            <td>0.043542</td>
            <td>98.01</td>
            <td>0.425044</td>
            <td>89.39</td>
        </tr>
        <tr>
            <td>7</td>
            <td>0.021633</td>
            <td>99.20</td>
            <td>0.309680</td>
            <td>92.42</td>
        </tr>
        <tr>
            <td>8</td>
            <td>0.022873</td>
            <td>99.20</td>
            <td>0.985550</td>
            <td>87.88</td>
        </tr>
        <tr>
            <td>9</td>
            <td>0.031532</td>
            <td>98.80</td>
            <td>0.444390</td>
            <td>90.91</td>
        </tr>
        <tr>
            <td>10</td>
            <td>0.014693</td>
            <td>99.60</td>
            <td>0.667717</td>
            <td>92.42</td>
        </tr>
        <tr>
            <td>11</td>
            <td>0.032846</td>
            <td>99.60</td>
            <td>0.359913</td>
            <td>89.39</td>
        </tr>
        <tr>
            <td>12</td>
            <td>0.013810</td>
            <td>99.60</td>
            <td>0.500779</td>
            <td>90.91</td>
        </tr>
        <tr>
            <td>13</td>
            <td>0.029297</td>
            <td>99.60</td>
            <td>0.393918</td>
            <td>90.91</td>
        </tr>
        <tr>
            <td>14</td>
            <td>0.006256</td>
            <td>100.00</td>
            <td>0.391421</td>
            <td>87.88</td>
        </tr>
        <tr>
            <td>15</td>
            <td>0.005283</td>
            <td>99.60</td>
            <td>0.591877</td>
            <td>92.42</td>
        </tr>
</table>

### Classification Reports

<table>
    <tr>
        <th>Class</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
        <th>Support</th>
    </tr>
    <tr>
        <td>Covid</td>
        <td>0.93</td>
        <td>0.96</td>
        <td>0.94</td>
        <td>26</td>
    </tr>
    <tr>
        <td>Normal</td>
        <td>0.87</td>
        <td>1.00</td>
        <td>0.93</td>
        <td>20</td>
    </tr>
    <tr>
        <td>Viral Pneumonia</td>
        <td>1.00</td>
        <td>0.80</td>
        <td>0.89</td>
        <td>20</td>
    </tr>
    <tr>
        <td><strong>Accuracy</strong></td>
        <td colspan="3"><strong>0.92</strong></td>
        <td><strong>66</strong></td>
    </tr>
    <tr>
        <td><strong>Macro Avg</strong></td>
        <td>0.93</td>
        <td>0.92</td>
        <td>0.92</td>
        <td>66</td>
    </tr>
    <tr>
        <td><strong>Weighted Avg</strong></td>
        <td>0.93</td>
        <td>0.92</td>
        <td>0.92</td>
        <td>66</td>
    </tr>
</table>

## Screenshots <a name = "screenshots"></a>

Here are some screenshots of the application:

![home](https://raw.githubusercontent.com/kevin-wijaya/resources/main/images/covid-19-detection-via-cnn-on-chest-x-rays/home.png)

![drop-image](https://raw.githubusercontent.com/kevin-wijaya/resources/main/images/covid-19-detection-via-cnn-on-chest-x-rays/drop-image.png)

![classify](https://raw.githubusercontent.com/kevin-wijaya/resources/main/images/covid-19-detection-via-cnn-on-chest-x-rays/classify.png)

## Author <a name = "author"></a>
- **Kevin Wijaya** 