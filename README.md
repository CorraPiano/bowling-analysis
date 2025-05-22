# Trajectory and Spin Detection in Bowling Throws

<div align="center">
   <img src="https://img.shields.io/badge/Language-Python-brightgreen" alt="Language" />
   <img src="https://img.shields.io/badge/Platform-VSCode-blue" alt="Platform" />
   <img src="https://img.shields.io/badge/Powered_by-OpenCV-red" alt="By" />
   <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</div>

## Project Overview

This project focuses on detecting the **trajectory** and **spin** of a bowling ball using computer vision techniques. Leveraging Python, OpenCV, and 3D reconstruction, the system processes video recordings to extract detailed motion information for performance analysis or further research.

## Video results

Example outputs from the analysis pipeline:

https://github.com/user-attachments/assets/90463700-498f-4987-9d5a-1572e3613d1d

https://github.com/user-attachments/assets/69ed840a-b4c7-4e7b-ab15-2530a4a06a34

https://github.com/user-attachments/assets/b03cc3f3-2def-4e3e-a547-c772e84172c6

## Techniques Used

- Lines and ball detection using OpenCV
- Frame extraction and video preprocessing
- Lane and background segmentation
- 3D trajectory reconstruction
- Spin and rotation estimation using Optical Flow
- Visualization and result compilation

## Repository structure

```bash
├── data/                          # Raw and processed data
│   ├── auxiliary_data/            # Shared resources (e.g., circle_positions/)
│   ├── recording_1/               # Data for recording 1
│   ├── recording_2/               # Data for recording 2
│   ├── recording_3/               # Data for recording 3
│   └── recording_4/               # Data for recording 4

├── documents/                     # Notes and reference materials
│   ├── papers/                    # Research papers
│   ├── Links.txt                  # Useful links
│   └── Notes.txt                  # Project notes

├── notebook/                      # Jupyter notebooks for analysis
│   ├── ball_detection/            # Ball detection notebooks
│   ├── lane_detection/            # Lane detection notebooks
│   ├── reconstruction/            # 3D reconstruction notebooks
│   ├── spin/                      # Spin analysis notebooks
│   ├── trajectory/                # Trajectory analysis notebooks
│   ├── various_tests/             # Miscellaneous tests
│   └── video_creation/            # Video creation notebooks

├── output_data/                   # Output videos and results
│   ├── recording_1/               # Outputs for recording 1
│   ├── recording_2/               # Outputs for recording 2
│   ├── recording_3/               # Outputs for recording 3
│   ├── recording_4/               # Outputs for recording 4
│   └── templates/                 # Output templates

├── src/                           # Source code
│   ├── app.py                     # Web app entry point
│   ├── main.py                    # Main script to run the project
│   ├── ball_detection/            # Ball detection logic
│   ├── lane_detection/            # Lane detection logic
│   ├── reconstruction/            # 3D reconstruction logic
│   ├── spin/                      # Spin analysis logic
│   ├── trajectory/                # Trajectory analysis logic
│   └── utility/                   # Utility functions

├── README.md                      # Project overview and instructions
└── requirements.txt               # Python dependencies
```

## Hot to run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bowling-trajectory-spin.git
   cd bowling-trajectory-spin
   ```
   Replace `your-username` with your GitHub username.

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
    ```
    Install FFmpeg from the link https://ffmpeg.org/download.html
3. Navigate to the `src` directory:
   ```bash
   cd src
   ```
4. Run the main script:
   ```bash
   python main.py
   ```
   Alternatively, you can launch the web app (requires [Streamlit](https://streamlit.io/)):
   ```bash
   streamlit run app.py
   ```
   This will open an interactive web interface in your browser, allowing you to execute the full bowling analysis pipeline without using the command line.
5. Follow the prompts to select the recording and adjust parameters as needed.
6. Use Jupyter notebooks in the `notebook/` directory for analysis and visualization.

## Contacts

For questions, feedback, or collaboration, feel free to reach out:

- **Davide Corradina**  
  Mail: [davi.corra@libero.it](mailto:davi.corra@libero.it)  
  GitHub: [CorraPiano](https://github.com/CorraPiano)

- **Michele Fassini**  
  Mail: [michele.fassini@icloud.com](mailto:michele.fassini@icloud.com)  
  GitHub: [MicheleFassini](https://github.com/MicheleFassini)

## Mentions

The videos to test the program were taken from YouTube.
