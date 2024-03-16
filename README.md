# vinyl-record-indexing

Use computer vision to index your vinyl record collection.

## Demo

https://github.com/capjamesg/vinyl-record-indexing/assets/37276661/327f42fa-bace-4f17-b9b1-7e60f5bea524

## Getting Started

First, clone this repository and install the required dependencies:

```
git clone https://github.com/capjamesg/vinyl-record-indexing
cd vinyl-record-indexing
```

Next, [follow the MobileCLIP installation instructions](https://github.com/apple/ml-mobileclip) to install MobileCLIP, on which this project depends.

You will need an OpenAI API key to use this project. Register for an OpenAI API key, then export it into your environment using the following command:

```
export OPENAI_API_KEY=""
```

To start indexing your vinyl record collection, run the following command in the root project directory:

```
python3 app.py
```

When you run this command, a window will appear showing the feed from your webcam. In the top left corner, the prompt most similar to the current frame, as well as a counter showing how many records have been identified in the video feed, will show.

To start indexing your collection, place a vinyl in front of your camera until the `Vinyls recorded` counter increments. Repeat this process for all vinyls you want to index.

Then, open your palm (like you would if you were giving someone a high-five) and hold it until the camera stops. Opening your palm is a control sequence to indicate you have no more records to index.

Your camera will stop and all unique images will be sent to the OpenAI GPT-4 with Vision API for processing. The results, featuring the name of each vinyl record and the artist who wrote it, will be saved in a file called `results.csv`.

## License

This project is licensed under an [MIT license](LICENSE).

Refer to the [MobileCLIP license](https://github.com/apple/ml-mobileclip?tab=License-1-ov-file) for terms of use of MobileCLIP, on which this project depends. Of note, you can swap MobileCLIP for any CLIP-like model (i.e. the original CLIP model from OpenAI, which is licensed under an MIT license), although this will involve manually changing this script to work with your chosen model.
