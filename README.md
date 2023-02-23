# CloudIC AI - ISS Code Submission

This repository contains the code that will be submitted to the Astro Pi Team to be run on the ISS. It's part of the CloudIC AI project, which is about analysing clouds in images taken with a raspberry pi on the ISS. The code is written in Python 3 and to be run on a Raspberry Pi 4 with a Raspberry Pi HQ cameral, running Astro Pi's Flight OS.

## Goal

The goal for this code is to reliably take images of the Earth from the ISS in a 3 hour window, while not exceeding a 3 GB data storage limit. To maximize the number of images, the program filters out night images and applies a rough mask which filters out everything but clouds.

## Implementation

The code is split into three main files: 'main.py', 'camera.py' and 'image_processing.py'. The 'main.py' file contains the main loop of the program, which starts two separate threads for taking images and processing them. The 'main.py' also catches and logs any exceptions that might occur and restarts the program if needed. The 'camera.py' file contains the code for the image taking thread, which also filters out night images and makes sure the maximum file size is not exceeded. Lastly the 'image_processing.py' file contains the code for the image processing thread, which is responsible for computing a rough cloud mask and applying it, which reduces the file size of the image.

## Usage

1. Clone the repository
2. Run 'python3 main.py'
